"""Optional Rust-backed LCZero batch loader."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import torch


def _load_native_module():
    try:
        import chess_engine_4_native
    except ImportError as error:
        raise RuntimeError(
            "The Rust dataloader extension is required. Build it with "
            "`uv run maturin develop --manifest-path crates/leela_loader/Cargo.toml --release`."
        ) from error
    return chess_engine_4_native


def iter_native_packed_batches(
    paths: Sequence[Path],
    *,
    batch_size: int,
    prefetch_per_thread: int,
    threads: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    chess_engine_4_native = _load_native_module()

    native_iter = chess_engine_4_native.iter_prefetched_packed_batches(
        [str(path) for path in paths],
        batch_size,
        prefetch_per_thread,
        threads,
    )

    def as_torch_batch(batch):
        packed_planes, plane_scalars, policy_indices, policy_probs, value = batch
        return (
            torch.from_numpy(packed_planes),
            torch.from_numpy(plane_scalars),
            torch.from_numpy(policy_indices),
            torch.from_numpy(policy_probs),
            torch.from_numpy(value),
        )

    return map(as_torch_batch, native_iter)


def iter_native_parquet_batches(
    paths: Sequence[Path],
    *,
    batch_size: int,
    prefetch_per_thread: int,
    threads: int,
    sampling_rate: float = 1.0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    chess_engine_4_native = _load_native_module()
    native_iter = chess_engine_4_native.iter_prefetched_parquet_batches(
        [str(path) for path in paths],
        batch_size,
        prefetch_per_thread,
        threads,
        sampling_rate,
    )

    def as_torch_batch(batch):
        return tuple(torch.from_numpy(array) for array in batch)

    return map(as_torch_batch, native_iter)


def convert_native_lc0_tar_to_parquet(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    native = _load_native_module()
    return native.convert_lc0_tar_to_parquet(str(input_path), str(output_path))


def native_parquet_row_counts(paths: Sequence[Path]) -> list[tuple[str, int]]:
    native = _load_native_module()
    return native.parquet_row_counts([str(path) for path in paths])


def inspect_native_lc0_tars(
    paths: Sequence[Path],
) -> tuple[list[tuple[str, int, int]], int]:
    native = _load_native_module()
    return native.inspect_lc0_tars([str(path) for path in paths])
