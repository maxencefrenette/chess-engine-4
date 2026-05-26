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
    for packed_planes, plane_scalars, policy_indices, policy_probs, value in native_iter:
        yield (
            torch.from_numpy(packed_planes),
            torch.from_numpy(plane_scalars),
            torch.from_numpy(policy_indices),
            torch.from_numpy(policy_probs),
            torch.from_numpy(value),
        )
