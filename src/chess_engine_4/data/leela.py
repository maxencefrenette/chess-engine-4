"""Vectorized reader for LCZero v6 training records stored inside tar files."""

from __future__ import annotations

import glob
import gzip
import os
import tarfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import IterableDataset, get_worker_info

DEFAULT_DATA_ENV_VAR = "CHESS_ENGINE_4_DATA_PATH"

POLICY_SIZE = 1858
INPUT_PLANE_COUNT = 112
HISTORY_PLANE_COUNT = 104
BOARD_SIZE = 8
VALUE_TYPE_COUNT = 6
VALUE_FIELDS = 3
V6_RECORD_SIZE = 8356

LEELA_V6_DTYPE = np.dtype(
    [
        ("version", "<u4"),
        ("input_format", "<u4"),
        ("probabilities", "<f4", (POLICY_SIZE,)),
        ("planes", "<u8", (HISTORY_PLANE_COUNT,)),
        ("castling_us_ooo", "u1"),
        ("castling_us_oo", "u1"),
        ("castling_them_ooo", "u1"),
        ("castling_them_oo", "u1"),
        ("side_to_move_or_enpassant", "u1"),
        ("rule50_count", "u1"),
        ("invariance_info", "u1"),
        ("dummy", "u1"),
        ("root_q", "<f4"),
        ("best_q", "<f4"),
        ("root_d", "<f4"),
        ("best_d", "<f4"),
        ("root_m", "<f4"),
        ("best_m", "<f4"),
        ("plies_left", "<f4"),
        ("result_q", "<f4"),
        ("result_d", "<f4"),
        ("played_q", "<f4"),
        ("played_d", "<f4"),
        ("played_m", "<f4"),
        ("orig_q", "<f4"),
        ("orig_d", "<f4"),
        ("orig_m", "<f4"),
        ("visits", "<u4"),
        ("played_idx", "<u2"),
        ("best_idx", "<u2"),
        ("policy_kld", "<f4"),
        ("reserved", "<u4"),
    ]
)

if LEELA_V6_DTYPE.itemsize != V6_RECORD_SIZE:
    raise RuntimeError(f"LCZero v6 dtype has size {LEELA_V6_DTYPE.itemsize}, expected 8356.")


@dataclass(frozen=True, slots=True)
class LeelaBatch:
    """One LCZero-shaped training batch."""

    planes: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.planes, self.policy, self.value


@dataclass(frozen=True, slots=True)
class CompactLeelaBatch:
    """One training batch with compact CPU-side input planes."""

    binary_planes: torch.Tensor
    plane_scalars: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.binary_planes, self.plane_scalars, self.policy, self.value


class LeelaTarDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Yield LCZero-shaped tensor batches from v6 tar training data.

    Output contract:
    - planes: float32 tensor shaped [batch, 112, 8, 8]
    - policy: float32 tensor shaped [batch, 1858], preserving LCZero's -1
      illegal-move sentinel values
    - value: float32 tensor shaped [batch, 6, 3] with rows:
      result, best, played, orig, root, short-term

    This intentionally does no shuffling, rescoring, deblundering, or position
    sampling. Tar members are treated as chunks and converted with vectorized
    NumPy operations before becoming PyTorch tensors.
    """

    def __init__(
        self,
        paths: Sequence[Path | str] | Path | str | None = None,
        *,
        batch_size: int,
        env_var: str = DEFAULT_DATA_ENV_VAR,
        max_records: int | None = None,
        drop_last: bool = False,
        compact_planes: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.paths = resolve_data_paths(paths, env_var=env_var)
        self.batch_size = batch_size
        self.max_records = max_records
        self.drop_last = drop_last
        self.compact_planes = compact_planes

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        emitted = 0
        pending: list[np.ndarray] = []
        pending_count = 0

        for frames in iter_tar_frame_chunks(_worker_paths(self.paths)):
            if self.max_records is not None:
                remaining = self.max_records - emitted - pending_count
                if remaining <= 0:
                    break
                frames = frames[:remaining]
                if len(frames) == 0:
                    break

            pending.append(frames)
            pending_count += len(frames)

            while pending_count >= self.batch_size:
                batch_frames, pending, pending_count = _take_batch(
                    pending,
                    pending_count,
                    self.batch_size,
                )
                emitted += len(batch_frames)
                yield tensors_from_frames(
                    batch_frames,
                    compact_planes=self.compact_planes,
                ).as_tuple()

        if pending_count > 0 and not self.drop_last:
            batch_frames = np.concatenate(pending) if len(pending) > 1 else pending[0]
            yield tensors_from_frames(batch_frames, compact_planes=self.compact_planes).as_tuple()


def resolve_data_paths(
    paths: Sequence[Path | str] | Path | str | None,
    *,
    env_var: str = DEFAULT_DATA_ENV_VAR,
) -> list[Path]:
    """Resolve explicit paths or the configured environment variable into tar files."""

    load_dotenv(dotenv_path=Path.cwd() / ".env")

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


def iter_tar_frame_chunks(paths: Sequence[Path]) -> Iterator[np.ndarray]:
    """Yield structured v6 frame arrays from every regular member in tar files."""

    for path in paths:
        with tarfile.open(path, mode="r:*") as tar:
            for member in tar:
                if not member.isfile() or Path(member.name).name == "LICENSE":
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                if _is_gzip(payload):
                    payload = gzip.decompress(payload)
                yield parse_frame_chunk(payload, source=f"{path}:{member.name}")


def parse_frame_chunk(payload: bytes, *, source: str = "<buffer>") -> np.ndarray:
    """Parse one decompressed LCZero v6 chunk into a structured NumPy array."""

    if len(payload) == 0:
        return np.empty(0, dtype=LEELA_V6_DTYPE)
    if len(payload) % V6_RECORD_SIZE != 0:
        raise ValueError(
            f"{source} has {len(payload)} bytes, not a multiple of {V6_RECORD_SIZE}."
        )

    frames = np.frombuffer(payload, dtype=LEELA_V6_DTYPE)
    if len(frames) and not np.all(frames["version"] == 6):
        versions = sorted({int(version) for version in frames["version"]})
        raise ValueError(f"{source} contains unsupported LCZero versions: {versions}.")
    return frames


def tensors_from_frames(
    frames: np.ndarray,
    *,
    compact_planes: bool = False,
) -> LeelaBatch | CompactLeelaBatch:
    """Convert a structured v6 frame array into LCZero-shaped PyTorch tensors."""

    if compact_planes:
        return CompactLeelaBatch(
            binary_planes=torch.from_numpy(binary_planes_from_frames(frames)),
            plane_scalars=torch.from_numpy(plane_scalars_from_frames(frames)),
            policy=torch.from_numpy(np.array(frames["probabilities"], dtype=np.float32, copy=True)),
            value=torch.from_numpy(values_from_frames(frames)),
        )

    return LeelaBatch(
        planes=torch.from_numpy(planes_from_frames(frames)),
        policy=torch.from_numpy(np.array(frames["probabilities"], dtype=np.float32, copy=True)),
        value=torch.from_numpy(values_from_frames(frames)),
    )


def planes_from_frames(frames: np.ndarray) -> np.ndarray:
    """Build [batch, 112, 8, 8] planes matching lczero-training's tensor stage."""

    batch_size = len(frames)
    planes = np.empty(
        (batch_size, INPUT_PLANE_COUNT, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32,
    )
    planes[:, :HISTORY_PLANE_COUNT] = binary_planes_from_frames(frames)
    planes[:, HISTORY_PLANE_COUNT:] = plane_scalars_from_frames(frames)[:, :, None, None]
    return planes


def binary_planes_from_frames(frames: np.ndarray) -> np.ndarray:
    """Build compact [batch, 104, 8, 8] uint8 history planes."""
    batch_size = len(frames)
    packed = np.ascontiguousarray(frames["planes"])
    bytes_view = packed.view(np.uint8).reshape(batch_size, HISTORY_PLANE_COUNT, 8)
    history = np.unpackbits(bytes_view, bitorder="big", axis=2)
    return history.reshape(
        batch_size,
        HISTORY_PLANE_COUNT,
        BOARD_SIZE,
        BOARD_SIZE,
    )


def plane_scalars_from_frames(frames: np.ndarray) -> np.ndarray:
    """Build scalar values for planes 104..111."""

    batch_size = len(frames)
    scalars = np.empty((batch_size, INPUT_PLANE_COUNT - HISTORY_PLANE_COUNT), dtype=np.float32)
    scalars[:, 0] = frames["castling_us_ooo"]
    scalars[:, 1] = frames["castling_us_oo"]
    scalars[:, 2] = frames["castling_them_ooo"]
    scalars[:, 3] = frames["castling_them_oo"]
    scalars[:, 4] = frames["side_to_move_or_enpassant"]
    scalars[:, 5] = frames["rule50_count"].astype(np.float32) / 99.0
    scalars[:, 6] = 0.0
    scalars[:, 7] = 1.0
    return scalars


def values_from_frames(frames: np.ndarray) -> np.ndarray:
    """Build [batch, 6, 3] values matching lczero-training's tensor contract."""

    batch_size = len(frames)
    values = np.empty((batch_size, VALUE_TYPE_COUNT, VALUE_FIELDS), dtype=np.float32)
    values[:, 0, 0] = frames["result_q"]
    values[:, 0, 1] = frames["result_d"]
    values[:, 0, 2] = frames["plies_left"]
    values[:, 1, 0] = frames["best_q"]
    values[:, 1, 1] = frames["best_d"]
    values[:, 1, 2] = frames["best_m"]
    values[:, 2, 0] = frames["played_q"]
    values[:, 2, 1] = frames["played_d"]
    values[:, 2, 2] = frames["played_m"]
    values[:, 3, 0] = frames["orig_q"]
    values[:, 3, 1] = frames["orig_d"]
    values[:, 3, 2] = frames["orig_m"]
    values[:, 4, 0] = frames["root_q"]
    values[:, 4, 1] = frames["root_d"]
    values[:, 4, 2] = frames["root_m"]
    values[:, 5, 0] = 0.0
    values[:, 5, 1] = 0.0
    values[:, 5, 2] = np.nan
    return values


def _take_batch(
    pending: list[np.ndarray],
    pending_count: int,
    batch_size: int,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    selected: list[np.ndarray] = []
    selected_count = 0

    while selected_count < batch_size:
        frames = pending.pop(0)
        needed = batch_size - selected_count
        if len(frames) <= needed:
            selected.append(frames)
            selected_count += len(frames)
        else:
            selected.append(frames[:needed])
            pending.insert(0, frames[needed:])
            selected_count += needed

    pending_count -= batch_size
    batch_frames = np.concatenate(selected) if len(selected) > 1 else selected[0]
    return batch_frames, pending, pending_count


def _worker_paths(paths: Sequence[Path]) -> list[Path]:
    worker = get_worker_info()
    if worker is None:
        return list(paths)
    return list(paths)[worker.id :: worker.num_workers]


def _is_gzip(payload: bytes) -> bool:
    return len(payload) >= 2 and payload[0] == 0x1F and payload[1] == 0x8B


def _looks_like_glob(value: str) -> bool:
    return any(char in value for char in "*?[")
