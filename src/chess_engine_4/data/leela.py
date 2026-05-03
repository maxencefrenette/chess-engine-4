"""Simplified reader for LCZero binary training records stored inside tar files."""

from __future__ import annotations

import glob
import gzip
import os
import struct
import tarfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

DEFAULT_DATA_ENV_VAR = "CHESS_ENGINE_4_DATA_PATH"

POLICY_SIZE = 1858
PLANE_COUNT = 104
BOARD_SIZE = 8

V5_RECORD_SIZE = 8308
V6_RECORD_SIZE = 8356

_U32 = struct.Struct("<I")
_V6_TARGETS = struct.Struct("<fffff")


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    """One LCZero training position in model-ready form."""

    planes: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    plies_left: np.float32
    visits: int
    version: int
    input_format: int


@dataclass(frozen=True, slots=True)
class LeelaBatch:
    """A mini-batch produced from Leela training data."""

    planes: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    plies_left: torch.Tensor


class LeelaTarDataset(IterableDataset[TrainingRecord]):
    """Stream LCZero v5/v6 records from tar files.

    The expected input is one or more plain `.tar` files. Tar members may be raw
    packed records or gzip streams containing packed records.
    """

    def __init__(
        self,
        paths: Sequence[Path | str] | Path | str | None = None,
        *,
        env_var: str = DEFAULT_DATA_ENV_VAR,
        shuffle_files: bool = False,
        max_records: int | None = None,
    ) -> None:
        self.paths = resolve_data_paths(paths, env_var=env_var)
        self.shuffle_files = shuffle_files
        self.max_records = max_records

    def __iter__(self) -> Iterator[TrainingRecord]:
        emitted = 0
        paths = list(self.paths)
        if self.shuffle_files:
            np.random.default_rng().shuffle(paths)

        for path in paths:
            for record in iter_tar_records(path):
                yield record
                emitted += 1
                if self.max_records is not None and emitted >= self.max_records:
                    return


def resolve_data_paths(
    paths: Sequence[Path | str] | Path | str | None,
    *,
    env_var: str = DEFAULT_DATA_ENV_VAR,
) -> list[Path]:
    """Resolve explicit paths or the configured environment variable into tar files."""

    raw_paths: list[str]
    if paths is None:
        value = os.environ.get(env_var)
        if not value:
            raise ValueError(f"Set {env_var} to a Leela data .tar file, directory, or glob.")
        raw_paths = value.split(os.pathsep)
    elif isinstance(paths, str | Path):
        raw_paths = [str(paths)]
    else:
        raw_paths = [str(path) for path in paths]

    resolved: list[Path] = []
    for raw_path in raw_paths:
        matches = [Path(match) for match in sorted(glob.glob(raw_path))]
        if not _looks_like_glob(raw_path):
            matches = [Path(raw_path)]
        for match in matches:
            if match.is_dir():
                resolved.extend(sorted(match.glob("*.tar")))
            elif match.exists():
                resolved.append(match)

    if not resolved:
        raise FileNotFoundError(f"No Leela tar files found from: {raw_paths}")
    return resolved


def iter_tar_records(path: Path) -> Iterator[TrainingRecord]:
    """Yield parsed training records from every regular member in a tar file."""

    with tarfile.open(path, mode="r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            if _is_gzip(payload):
                payload = gzip.decompress(payload)
            yield from iter_records(payload)


def iter_records(payload: bytes) -> Iterator[TrainingRecord]:
    """Parse consecutive packed LCZero records from a byte payload."""

    offset = 0
    size = len(payload)
    while offset + 4 <= size:
        version = _U32.unpack_from(payload, offset)[0]
        record_size = _record_size(version)
        if record_size is None or offset + record_size > size:
            return
        yield parse_record(memoryview(payload)[offset : offset + record_size])
        offset += record_size


def parse_record(record: memoryview) -> TrainingRecord:
    """Parse one packed v5/v6 LCZero training record."""

    version = _U32.unpack_from(record, 0)[0]
    record_size = _record_size(version)
    if record_size is None or len(record) != record_size:
        raise ValueError(f"Unsupported or malformed Leela training record version: {version}")

    input_format = _U32.unpack_from(record, 4)[0]
    policy = np.frombuffer(record, dtype="<f4", count=POLICY_SIZE, offset=8).astype(
        np.float32,
        copy=True,
    )
    packed_planes = np.frombuffer(record, dtype="<u8", count=PLANE_COUNT, offset=7440)
    planes = unpack_bitplanes(packed_planes)

    if version >= 6:
        result_q, result_d, _played_q, _played_d, plies_left = _V6_TARGETS.unpack_from(record, 8308)
        visits = _U32.unpack_from(record, 8340)[0]
        value = wdl_from_q_d(result_q, result_d)
    else:
        result = struct.unpack_from("<b", record, 8279)[0]
        plies_left = struct.unpack_from("<f", record, 8304)[0]
        visits = 0
        value = np.array([(result + 1.0) / 2.0], dtype=np.float32)

    return TrainingRecord(
        planes=planes,
        policy=policy,
        value=value,
        plies_left=np.float32(plies_left),
        visits=visits,
        version=version,
        input_format=input_format,
    )


def unpack_bitplanes(packed_planes: np.ndarray) -> np.ndarray:
    """Convert 104 packed uint64 bitboards to float32 planes shaped [104, 8, 8]."""

    bytes_view = packed_planes.astype("<u8", copy=False).view(np.uint8).reshape(PLANE_COUNT, 8)
    bits = np.unpackbits(bytes_view, bitorder="little", axis=1)
    return bits.reshape(PLANE_COUNT, BOARD_SIZE, BOARD_SIZE).astype(np.float32, copy=False)


def wdl_from_q_d(q: float, d: float) -> np.ndarray:
    """Convert LCZero Q/D targets into [win, draw, loss] probabilities."""

    win = (q + 1.0 - d) / 2.0
    loss = 1.0 - d - win
    return np.array([win, d, loss], dtype=np.float32)


def collate_records(records: Sequence[TrainingRecord]) -> LeelaBatch:
    """Collate records into tensors suitable for a PyTorch training step."""

    return LeelaBatch(
        planes=torch.from_numpy(np.stack([record.planes for record in records])),
        policy=torch.from_numpy(np.stack([record.policy for record in records])),
        value=torch.from_numpy(np.stack([record.value for record in records])),
        plies_left=torch.tensor([record.plies_left for record in records], dtype=torch.float32),
    )


def _record_size(version: int) -> int | None:
    if version == 5:
        return V5_RECORD_SIZE
    if version == 6:
        return V6_RECORD_SIZE
    return None


def _is_gzip(payload: bytes) -> bool:
    return len(payload) >= 2 and payload[0] == 0x1F and payload[1] == 0x8B


def _looks_like_glob(value: str) -> bool:
    return any(char in value for char in "*?[")
