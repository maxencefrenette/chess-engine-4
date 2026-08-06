from __future__ import annotations

import pytest
import torch

from chess_engine_4.kernels.dense import (
    SUPPORTED_DENSE_WIDTHS,
    dense_block_forward,
    quantize_mxfp8_transpose,
)


def test_dense_kernel_supports_power_of_two_widths() -> None:
    assert {32, 64, 128, 256, 512, 1024, 2048} == SUPPORTED_DENSE_WIDTHS


def test_dense_kernel_rejects_cpu_input() -> None:
    x = torch.zeros(128, 128, dtype=torch.bfloat16)
    norm = torch.ones(128, dtype=torch.bfloat16)
    gate_up = torch.zeros(1024, 128, dtype=torch.bfloat16)
    down = torch.zeros(128, 512, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="x must be a CUDA tensor"):
        dense_block_forward(x, norm, gate_up, down, precision="bf16")


def test_transpose_quantizer_rejects_cpu_input() -> None:
    with pytest.raises(ValueError, match="tensor must be a CUDA tensor"):
        quantize_mxfp8_transpose(torch.zeros(128, 256, dtype=torch.bfloat16))
