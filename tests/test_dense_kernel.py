from __future__ import annotations

import pytest
import torch

from chess_engine_4.kernels.capabilities import SUPPORTED_DENSE_WIDTHS, dense_op_prefix
from chess_engine_4.kernels.dense import (
    _dense_op,
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


def test_dense_kernel_dispatches_sm120_bindings() -> None:
    assert dense_op_prefix((12, 0)) == "sm120_"


def test_dense_kernel_rejects_unsupported_gpu_architecture() -> None:
    with pytest.raises(
        ValueError,
        match="support SM80, SM90, SM100, and SM120, got SM110",
    ):
        dense_op_prefix((11, 0))


def test_dense_kernel_dispatches_sm90_prefix(monkeypatch) -> None:
    sentinel = object()
    module = type("Extension", (), {"sm90_dense_rmsnorm_forward": sentinel})()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr("chess_engine_4.kernels.dense.extension", lambda: module)

    assert _dense_op(torch.empty(0), "rmsnorm_forward") is sentinel


def test_transpose_quantizer_rejects_cpu_input() -> None:
    with pytest.raises(ValueError, match="tensor must be a CUDA tensor"):
        quantize_mxfp8_transpose(torch.zeros(128, 256, dtype=torch.bfloat16))
