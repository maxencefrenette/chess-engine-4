"""Specialized dense-block CUDA operators."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

D_MODEL = 128
HIDDEN_DIM = 512
GATE_UP_DIM = 2 * HIDDEN_DIM
TK_GEMM_OUTPUT_ALIGNMENT = 256
MXFP8_TILE_SIZE = 128


@dataclass(frozen=True, slots=True)
class Mxfp8Tensor:
    values: torch.Tensor
    scales: torch.Tensor


def _extension() -> Any:
    try:
        return importlib.import_module("_chess_engine_4_kernels")
    except ImportError as exc:
        raise RuntimeError(
            "The CUDA kernel extension is not built. Run `uv run build-kernels` on a "
            f"Blackwell CUDA host. Import error: {exc}"
        ) from exc


def _check_matrix(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must have dtype torch.bfloat16")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if any(size % MXFP8_TILE_SIZE for size in tensor.shape):
        raise ValueError(f"{name} dimensions must be multiples of {MXFP8_TILE_SIZE}")


def quantize_mxfp8(tensor: torch.Tensor) -> Mxfp8Tensor:
    _check_matrix("tensor", tensor)
    rows, columns = tensor.shape
    values = torch.empty_like(tensor, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        rows // MXFP8_TILE_SIZE,
        columns // MXFP8_TILE_SIZE,
        32,
        16,
        dtype=torch.uint8,
        device=tensor.device,
    )
    _extension().mxfp8_quantize(tensor, values, scales)
    return Mxfp8Tensor(values=values, scales=scales)


def mxfp8_gemm(left: Mxfp8Tensor, right: Mxfp8Tensor) -> torch.Tensor:
    rows, reduction = left.values.shape
    output_columns, right_reduction = right.values.shape
    if reduction != right_reduction:
        raise ValueError("MXFP8 GEMM reduction dimensions do not match")
    if output_columns % TK_GEMM_OUTPUT_ALIGNMENT:
        raise ValueError(
            f"ThunderKittens baseline GEMM requires output columns divisible by "
            f"{TK_GEMM_OUTPUT_ALIGNMENT}"
        )
    output = torch.empty(
        rows,
        output_columns,
        dtype=torch.bfloat16,
        device=left.values.device,
    )
    _extension().mxfp8_gemm(
        left.values,
        left.scales,
        right.values,
        right.scales,
        output,
    )
    return output


def dense_d128_mxfp8_forward(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run a d128 SwiGLU residual block through ThunderKittens MXFP8 kernels.

    This correctness baseline composes separately launched TK kernels. The down
    projection is padded to 256 outputs because TK's stock MXFP8 GEMM binding is
    specialized for 256-column output tiles.
    """

    _check_matrix("x", x)
    if x.shape[1] != D_MODEL:
        raise ValueError(f"x must have shape [batch, {D_MODEL}]")
    if norm_weight.shape != (D_MODEL,) or norm_weight.dtype != torch.bfloat16:
        raise ValueError(f"norm_weight must be BF16 with shape [{D_MODEL}]")
    if gate_up_weight.shape != (GATE_UP_DIM, D_MODEL):
        raise ValueError(f"gate_up_weight must have shape [{GATE_UP_DIM}, {D_MODEL}]")
    if down_weight.shape != (D_MODEL, HIDDEN_DIM):
        raise ValueError(f"down_weight must have shape [{D_MODEL}, {HIDDEN_DIM}]")

    residual = x
    normalized = F.rms_norm(x, (D_MODEL,), norm_weight, eps).contiguous()
    gate_up = mxfp8_gemm(quantize_mxfp8(normalized), quantize_mxfp8(gate_up_weight))
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = (F.silu(gate) * up).contiguous()

    padded_down_weight = F.pad(
        down_weight,
        (0, 0, 0, TK_GEMM_OUTPUT_ALIGNMENT - D_MODEL),
    ).contiguous()
    projected = mxfp8_gemm(
        quantize_mxfp8(hidden),
        quantize_mxfp8(padded_down_weight),
    )
    return residual + projected[:, :D_MODEL]
