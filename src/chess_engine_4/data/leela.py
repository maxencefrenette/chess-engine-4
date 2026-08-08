"""LCZero v6 training data loader."""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from dotenv import load_dotenv

from chess_engine_4.data.native import iter_native_packed_batches, iter_native_parquet_batches

DEFAULT_DATA_ENV_VAR = "CHESS_ENGINE_4_DATA_PATH"

POLICY_SIZE = 1858
COMPACT_POLICY_SIZE = 218
INPUT_PLANE_COUNT = 112
HISTORY_PLANE_COUNT = 104
RULE50_PLANE_INDEX = HISTORY_PLANE_COUNT + 5
BOARD_SIZE = 8
VALUE_TYPE_COUNT = 6
VALUE_FIELDS = 3
V6_RECORD_SIZE = 8356


class LeelaTarDataset:
    """Yield packed training batches from v6 tar training data.

    Output contract:
    - packed_planes: uint8 tensor shaped [batch, 104, 8]
    - plane_scalars: float32 tensor shaped [batch, 8], including raw rule-50 plies
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
        prefetch_per_thread: int = 2,
        threads: int = 2,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if prefetch_per_thread <= 0:
            raise ValueError("prefetch_per_thread must be positive.")
        if threads <= 0:
            raise ValueError("threads must be positive.")

        self.paths = resolve_data_paths(paths, env_var=env_var)
        self.batch_size = batch_size
        self.prefetch_per_thread = prefetch_per_thread
        self.threads = threads

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        return iter_native_packed_batches(
            self.paths,
            batch_size=self.batch_size,
            prefetch_per_thread=self.prefetch_per_thread,
            threads=self.threads,
        )


class LeelaParquetDataset(LeelaTarDataset):
    """Yield the standard packed training batches from converted Parquet data."""

    def __init__(
        self,
        paths: Sequence[Path | str] | Path | str | None = None,
        *,
        batch_size: int,
        env_var: str = DEFAULT_DATA_ENV_VAR,
        prefetch_per_thread: int = 2,
        threads: int = 2,
        retention_rate: float = 1.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if prefetch_per_thread <= 0:
            raise ValueError("prefetch_per_thread must be positive.")
        if threads <= 0:
            raise ValueError("threads must be positive.")
        if retention_rate not in {0.25, 0.5, 1.0}:
            raise ValueError("retention_rate must be one of: 0.25, 0.5, 1.0.")
        self.paths = resolve_data_paths(paths, env_var=env_var, suffix=".parquet")
        self.batch_size = batch_size
        self.prefetch_per_thread = prefetch_per_thread
        self.threads = threads
        self.retention_rate = retention_rate

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        retention_numerator = {0.25: 1, 0.5: 2, 1.0: 4}[self.retention_rate]
        return iter_native_parquet_batches(
            self.paths,
            batch_size=self.batch_size,
            prefetch_per_thread=self.prefetch_per_thread,
            threads=self.threads,
            retention_numerator=retention_numerator,
            retention_denominator=4,
        )


def resolve_data_paths(
    paths: Sequence[Path | str] | Path | str | None,
    *,
    env_var: str = DEFAULT_DATA_ENV_VAR,
    suffix: str = ".tar",
) -> list[Path]:
    """Resolve explicit paths or the configured environment variable into data files."""

    load_dotenv(dotenv_path=Path.cwd() / ".env")

    raw_paths: list[str]
    if paths is None:
        value = os.environ.get(env_var)
        if not value:
            raise ValueError(f"Set {env_var} to a Leela data {suffix} file, directory, or glob.")
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
                resolved.extend(sorted(match.glob(f"*{suffix}")))
            elif match.exists():
                resolved.append(match)

    if not resolved:
        raise FileNotFoundError(f"No Leela {suffix} files found from: {raw_paths}")
    return resolved


def _looks_like_glob(value: str) -> bool:
    return any(char in value for char in "*?[")
