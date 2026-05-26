"""LCZero v6 training data loader."""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.utils.data import IterableDataset, get_worker_info

from chess_engine_4.data.native import iter_native_packed_batches

DEFAULT_DATA_ENV_VAR = "CHESS_ENGINE_4_DATA_PATH"

POLICY_SIZE = 1858
COMPACT_POLICY_SIZE = 218
INPUT_PLANE_COUNT = 112
HISTORY_PLANE_COUNT = 104
BOARD_SIZE = 8
VALUE_TYPE_COUNT = 6
VALUE_FIELDS = 3
V6_RECORD_SIZE = 8356


class LeelaTarDataset(IterableDataset[tuple[torch.Tensor, ...]]):
    """Yield packed training batches from v6 tar training data.

    Output contract:
    - packed_planes: uint8 tensor shaped [batch, 104, 8]
    - plane_scalars: float32 tensor shaped [batch, 8]
    - policy_indices: int16 tensor shaped [batch, 218], padded with -1
    - policy_probs: float16 tensor shaped [batch, 218], padded with 0
    - value: float32 tensor shaped [batch, 6, 3] with rows:
      result, best, played, orig, root, short-term

    Decoding is native-only. Build the Rust extension with maturin before local
    training, or use the Modal training image which builds it automatically.
    """

    def __init__(
        self,
        paths: Sequence[Path | str] | Path | str | None = None,
        *,
        batch_size: int,
        env_var: str = DEFAULT_DATA_ENV_VAR,
        max_records: int | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.paths = resolve_data_paths(paths, env_var=env_var)
        self.batch_size = batch_size
        self.max_records = max_records
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        yield from iter_native_packed_batches(
            _worker_paths(self.paths),
            batch_size=self.batch_size,
            max_records=self.max_records,
            drop_last=self.drop_last,
        )


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


def _worker_paths(paths: Sequence[Path]) -> list[Path]:
    worker = get_worker_info()
    if worker is None:
        return list(paths)
    return list(paths)[worker.id :: worker.num_workers]


def _looks_like_glob(value: str) -> bool:
    return any(char in value for char in "*?[")
