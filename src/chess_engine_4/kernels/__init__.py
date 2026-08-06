"""Blackwell CUDA kernel bindings."""

from chess_engine_4.kernels.dense import (
    dense_block_forward,
    dense_block_trainable,
)

__all__ = ["dense_block_forward", "dense_block_trainable"]
