"""LCZero v6 training data loader."""

from __future__ import annotations

import glob
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from dotenv import load_dotenv

from chess_engine_4.data.native import iter_native_parquet_batches

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


class LeelaParquetDataset:
    """Yield the standard packed training batches from converted Parquet data."""

    def __init__(
        self,
        paths: Sequence[Path | str] | Path | str | None = None,
        *,
        batch_size: int,
        env_var: str = DEFAULT_DATA_ENV_VAR,
        prefetch_per_thread: int = 2,
        threads: int = 2,
        sampling_rate: float = 1.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if prefetch_per_thread <= 0:
            raise ValueError("prefetch_per_thread must be positive.")
        if threads <= 0:
            raise ValueError("threads must be positive.")
        if not 0.0 < sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be greater than 0 and at most 1.")
        self.paths = resolve_data_paths(paths, env_var=env_var, suffix=".parquet")
        self.batch_size = batch_size
        self.prefetch_per_thread = prefetch_per_thread
        self.threads = threads
        self.sampling_rate = sampling_rate

    def __iter__(
        self,
    ) -> Iterator[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        return iter_native_parquet_batches(
            self.paths,
            batch_size=self.batch_size,
            prefetch_per_thread=self.prefetch_per_thread,
            threads=self.threads,
            sampling_rate=self.sampling_rate,
        )


def resolve_data_paths(
    paths: Sequence[Path | str] | Path | str | None,
    *,
    env_var: str = DEFAULT_DATA_ENV_VAR,
    suffix: str = ".parquet",
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
