"""Specialized dense-block CUDA operators."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

SUPPORTED_DENSE_WIDTHS = frozenset({128, 256, 512, 1024, 2048})
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
    _extension().dense_quantize_mxfp8(tensor, values, scales)
    return Mxfp8Tensor(values=values, scales=scales)


def quantize_mxfp8_transpose(tensor: torch.Tensor) -> Mxfp8Tensor:
    """Transpose and quantize BF16 without materializing the BF16 transpose."""

    _check_matrix("tensor", tensor)
    rows, columns = tensor.shape
    values = torch.empty(
        columns,
        rows,
        dtype=torch.float8_e4m3fn,
        device=tensor.device,
    )
    scales = torch.empty(
        columns // MXFP8_TILE_SIZE,
        rows // MXFP8_TILE_SIZE,
        32,
        16,
        dtype=torch.uint8,
        device=tensor.device,
    )
    _extension().dense_quantize_mxfp8_transpose(tensor, values, scales)
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
    _extension().dense_mxfp8_gemm_wide(
        left.values,
        left.scales,
        right.values,
        right.scales,
        output,
    )
    return output


def _mxfp8_gemm_columns(
    left: torch.Tensor | Mxfp8Tensor,
    right: torch.Tensor | Mxfp8Tensor,
    *,
    output_columns: int,
) -> torch.Tensor:
    left_mxfp8 = left if isinstance(left, Mxfp8Tensor) else quantize_mxfp8(left)
    right_mxfp8 = right if isinstance(right, Mxfp8Tensor) else quantize_mxfp8(right)
    if output_columns != 128:
        return mxfp8_gemm(left_mxfp8, right_mxfp8)
    output = torch.empty(
        left_mxfp8.values.shape[0],
        output_columns,
        dtype=torch.bfloat16,
        device=left_mxfp8.values.device,
    )
    _extension().dense_mxfp8_gemm_narrow(
        left_mxfp8.values,
        left_mxfp8.scales,
        right_mxfp8.values,
        right_mxfp8.scales,
        output,
    )
    return output


def dense_mxfp8_forward(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run a dense SwiGLU residual block through ThunderKittens MXFP8 kernels."""

    output, _ = _dense_mxfp8_forward_components(
        x,
        norm_weight,
        gate_up_weight,
        down_weight,
        eps=eps,
    )
    return output


def dense_mxfp8_trainable(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run the MXFP8 forward with an explicit MXFP8/BF16 backward."""

    return _DenseMxfp8Function.apply(
        x,
        norm_weight,
        gate_up_weight,
        down_weight,
        eps,
    )


class _DenseMxfp8Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        output, intermediates = _dense_mxfp8_forward_components(
            x,
            norm_weight,
            gate_up_weight,
            down_weight,
            eps=eps,
        )
        normalized, gate_up, hidden = intermediates
        ctx.save_for_backward(
            x,
            norm_weight,
            gate_up_weight,
            down_weight,
            normalized,
            gate_up,
            hidden,
        )
        ctx.eps = eps
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        (
            x,
            norm_weight,
            gate_up_weight,
            down_weight,
            normalized,
            gate_up,
            hidden,
        ) = ctx.saved_tensors
        with torch.cuda.device(x.device):
            return _dense_mxfp8_backward(
                x,
                norm_weight,
                gate_up_weight,
                down_weight,
                normalized,
                gate_up,
                hidden,
                grad_output.contiguous(),
                eps=ctx.eps,
            )


def _dense_mxfp8_backward(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    normalized: torch.Tensor,
    gate_up: torch.Tensor,
    hidden: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
    d_model = x.shape[1]
    grad_hidden = mxfp8_gemm(
        quantize_mxfp8(grad_output),
        quantize_mxfp8_transpose(down_weight),
    )
    grad_gate_up = torch.empty_like(gate_up)
    _extension().dense_swiglu_backward(
        grad_hidden,
        gate_up,
        grad_gate_up,
    )

    grad_down_weight = _mxfp8_weight_gradient(
        grad_output,
        hidden,
        output_rows=d_model,
    )
    grad_gate_up_weight = _mxfp8_gemm_columns(
        quantize_mxfp8_transpose(grad_gate_up),
        quantize_mxfp8_transpose(normalized),
        output_columns=d_model,
    )
    grad_normalized = _mxfp8_gemm_columns(
        quantize_mxfp8(grad_gate_up),
        quantize_mxfp8_transpose(gate_up_weight),
        output_columns=d_model,
    )

    grad_x = torch.empty_like(x)
    grad_norm_weight_workspace = torch.zeros(
        d_model,
        dtype=torch.float32,
        device=x.device,
    )
    grad_norm_weight = torch.empty_like(norm_weight)
    _extension().dense_rmsnorm_backward(
        x,
        norm_weight,
        grad_normalized,
        grad_output,
        grad_x,
        grad_norm_weight_workspace,
        grad_norm_weight,
        eps,
    )
    return grad_x, grad_norm_weight, grad_gate_up_weight, grad_down_weight, None


def _dense_mxfp8_forward_components(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    _check_matrix("x", x)
    d_model = x.shape[1]
    hidden_dim = 4 * d_model
    gate_up_dim = 2 * hidden_dim
    if d_model not in SUPPORTED_DENSE_WIDTHS:
        choices = ", ".join(str(width) for width in sorted(SUPPORTED_DENSE_WIDTHS))
        raise ValueError(f"unsupported d_model={d_model}; choose from {choices}")
    if norm_weight.shape != (d_model,) or norm_weight.dtype != torch.bfloat16:
        raise ValueError(f"norm_weight must be BF16 with shape [{d_model}]")
    if gate_up_weight.shape != (gate_up_dim, d_model):
        raise ValueError(f"gate_up_weight must have shape [{gate_up_dim}, {d_model}]")
    if down_weight.shape != (d_model, hidden_dim):
        raise ValueError(f"down_weight must have shape [{d_model}, {hidden_dim}]")

    normalized = torch.empty_like(x)
    _extension().dense_rmsnorm_forward(x, norm_weight, normalized, eps)
    gate_up = mxfp8_gemm(quantize_mxfp8(normalized), quantize_mxfp8(gate_up_weight))
    hidden = torch.empty(
        x.shape[0],
        hidden_dim,
        dtype=x.dtype,
        device=x.device,
    )
    _extension().dense_swiglu_forward(gate_up, hidden)
    projected = _mxfp8_gemm_columns(
        hidden,
        down_weight,
        output_columns=d_model,
    )
    _extension().dense_residual_add(x, projected)
    return projected, (normalized, gate_up, hidden)


def _mxfp8_weight_gradient(
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    *,
    output_rows: int,
) -> torch.Tensor:
    if output_rows != 128:
        return mxfp8_gemm(
            quantize_mxfp8_transpose(grad_output),
            quantize_mxfp8_transpose(hidden),
        )
    left = grad_output.T.contiguous()
    right = hidden.T.contiguous()
    padded_left = F.pad(
        left,
        (0, 0, 0, TK_GEMM_OUTPUT_ALIGNMENT - output_rows),
    ).contiguous()
    output = mxfp8_gemm(
        quantize_mxfp8(padded_left),
        quantize_mxfp8(right),
    )
    return output[:output_rows]


def _dense_bf16_forward(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    normalized = F.rms_norm(x, (x.shape[1],), norm_weight, eps)
    gate, up = F.linear(normalized, gate_up_weight).chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    return x + F.linear(hidden, down_weight)
