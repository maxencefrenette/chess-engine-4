"""Specialized dense-block CUDA operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from chess_engine_4.kernels.capabilities import (
    dense_op_prefix,
    require_dense_kernel,
    require_dense_precision,
)
from chess_engine_4.kernels.extension import extension
from chess_engine_4.model.config import Precision

TK_GEMM_OUTPUT_ALIGNMENT = 256
MXFP8_TILE_SIZE = 128


def _dense_op(tensor: torch.Tensor, name: str) -> Any:
    prefix = dense_op_prefix(torch.cuda.get_device_capability(tensor.device))
    return getattr(extension(), f"{prefix}dense_{name}")


@dataclass(frozen=True, slots=True)
class Mxfp8Tensor:
    values: torch.Tensor
    scales: torch.Tensor


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
    require_dense_precision(torch.cuda.get_device_capability(tensor.device), "mxfp8")
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
    extension().dense_quantize_mxfp8(tensor, values, scales)
    return Mxfp8Tensor(values=values, scales=scales)


def quantize_mxfp8_transpose(tensor: torch.Tensor) -> Mxfp8Tensor:
    """Transpose and quantize BF16 without materializing the BF16 transpose."""

    _check_matrix("tensor", tensor)
    require_dense_precision(torch.cuda.get_device_capability(tensor.device), "mxfp8")
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
    extension().dense_quantize_mxfp8_transpose(tensor, values, scales)
    return Mxfp8Tensor(values=values, scales=scales)


def mxfp8_gemm(left: Mxfp8Tensor, right: Mxfp8Tensor) -> torch.Tensor:
    require_dense_precision(
        torch.cuda.get_device_capability(left.values.device),
        "mxfp8",
    )
    rows, reduction = left.values.shape
    output_columns, right_reduction = right.values.shape
    if reduction != right_reduction:
        raise ValueError("MXFP8 GEMM reduction dimensions do not match")
    if output_columns % TK_GEMM_OUTPUT_ALIGNMENT:
        raise ValueError(
            f"ThunderKittens GEMM requires output columns divisible by {TK_GEMM_OUTPUT_ALIGNMENT}"
        )
    output = torch.empty(
        rows,
        output_columns,
        dtype=torch.bfloat16,
        device=left.values.device,
    )
    extension().dense_mxfp8_gemm(
        left.values,
        left.scales,
        right.values,
        right.scales,
        output,
    )
    return output


def _bf16_gemm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    _check_bf16_matrix("left", left)
    _check_bf16_matrix("right", right)
    if left.shape[1] != right.shape[1]:
        raise ValueError("BF16 GEMM reduction dimensions do not match")
    require_dense_precision(torch.cuda.get_device_capability(left.device), "bf16")
    output = torch.empty(
        left.shape[0],
        right.shape[0],
        dtype=torch.bfloat16,
        device=left.device,
    )
    _dense_op(left, "bf16_gemm")(left, right, output)
    return output


def _check_bf16_matrix(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must have dtype torch.bfloat16")
    if tensor.ndim != 2 or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous matrix")


def dense_block_forward(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    precision: Precision,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run a dense SwiGLU residual block through the custom TK kernels."""

    output, _ = _dense_forward_components(
        x,
        norm_weight,
        gate_up_weight,
        down_weight,
        precision=precision,
        eps=eps,
    )
    return output


def dense_block_trainable(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    precision: Precision,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run the custom forward and explicit low-precision backward."""

    output = _DenseBlockFunction.apply(
        x,
        norm_weight,
        gate_up_weight,
        down_weight,
        eps,
        precision,
    )
    return cast(torch.Tensor, output)


class _DenseBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        eps: float,
        precision: Precision,
    ) -> torch.Tensor:
        output, intermediates = _dense_forward_components(
            x,
            norm_weight,
            gate_up_weight,
            down_weight,
            precision=precision,
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
        ctx.precision = precision
        return output

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        (grad_output,) = grad_outputs
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
            return _dense_backward(
                x,
                norm_weight,
                gate_up_weight,
                down_weight,
                normalized,
                gate_up,
                hidden,
                grad_output.contiguous(),
                precision=ctx.precision,
                eps=ctx.eps,
            )


def _dense_backward(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    normalized: torch.Tensor,
    gate_up: torch.Tensor,
    hidden: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    precision: Precision,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
    d_model = x.shape[1]
    if precision == "bf16":
        gradients = _dense_bf16_backward(
            x,
            norm_weight,
            gate_up_weight,
            down_weight,
            normalized,
            gate_up,
            hidden,
            grad_output,
            eps=eps,
        )
        return (*gradients[:-1], None, None)
    if precision != "mxfp8":
        raise ValueError(f"custom dense kernel does not support precision={precision!r}")
    grad_hidden = mxfp8_gemm(
        quantize_mxfp8(grad_output),
        quantize_mxfp8_transpose(down_weight),
    )
    grad_gate_up = torch.empty_like(gate_up)
    _dense_op(grad_hidden, "swiglu_backward")(
        grad_hidden,
        gate_up,
        grad_gate_up,
    )

    grad_down_weight = mxfp8_gemm(
        quantize_mxfp8_transpose(grad_output),
        quantize_mxfp8_transpose(hidden),
    )
    grad_gate_up_weight = mxfp8_gemm(
        quantize_mxfp8_transpose(grad_gate_up),
        quantize_mxfp8_transpose(normalized),
    )
    grad_normalized = mxfp8_gemm(
        quantize_mxfp8(grad_gate_up),
        quantize_mxfp8_transpose(gate_up_weight),
    )

    grad_x = torch.empty_like(x)
    grad_norm_weight_workspace = torch.zeros(
        d_model,
        dtype=torch.float32,
        device=x.device,
    )
    grad_norm_weight = torch.empty_like(norm_weight)
    _dense_op(x, "rmsnorm_backward")(
        x,
        norm_weight,
        grad_normalized,
        grad_output,
        grad_x,
        grad_norm_weight_workspace,
        grad_norm_weight,
        eps,
    )
    return grad_x, grad_norm_weight, grad_gate_up_weight, grad_down_weight, None, None


def _dense_bf16_backward(
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
    grad_hidden = _bf16_gemm(grad_output, down_weight.T.contiguous())
    grad_gate_up = torch.empty_like(gate_up)
    _dense_op(grad_hidden, "swiglu_backward")(grad_hidden, gate_up, grad_gate_up)
    grad_down_weight = torch.mm(grad_output.T, hidden)
    grad_gate_up_weight = torch.mm(grad_gate_up.T, normalized)
    grad_normalized = torch.mm(grad_gate_up, gate_up_weight)

    grad_x = torch.empty_like(x)
    grad_norm_weight_workspace = torch.zeros(d_model, dtype=torch.float32, device=x.device)
    grad_norm_weight = torch.empty_like(norm_weight)
    _dense_op(x, "rmsnorm_backward")(
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


def _dense_forward_components(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    precision: Precision,
    eps: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    _check_bf16_matrix("x", x)
    d_model = x.shape[1]
    hidden_dim = 4 * d_model
    gate_up_dim = 2 * hidden_dim
    require_dense_kernel(
        capability=torch.cuda.get_device_capability(x.device),
        precision=precision,
        d_model=d_model,
        hidden_dim=hidden_dim,
        rows=x.shape[0],
    )
    if norm_weight.shape != (d_model,) or norm_weight.dtype != torch.bfloat16:
        raise ValueError(f"norm_weight must be BF16 with shape [{d_model}]")
    if gate_up_weight.shape != (gate_up_dim, d_model):
        raise ValueError(f"gate_up_weight must have shape [{gate_up_dim}, {d_model}]")
    if down_weight.shape != (d_model, hidden_dim):
        raise ValueError(f"down_weight must have shape [{d_model}, {hidden_dim}]")

    normalized = torch.empty_like(x)
    _dense_op(x, "rmsnorm_forward")(x, norm_weight, normalized, eps)
    if precision == "bf16":
        gate_up = _bf16_gemm(normalized, gate_up_weight)
        hidden = torch.empty(x.shape[0], hidden_dim, dtype=x.dtype, device=x.device)
        _dense_op(gate_up, "swiglu_forward")(gate_up, hidden)
        projected = _bf16_gemm(hidden, down_weight)
        _dense_op(x, "residual_add")(x, projected)
        return projected, (normalized, gate_up, hidden)
    if precision != "mxfp8":
        raise ValueError(f"custom dense kernel does not support precision={precision!r}")
    gate_up = mxfp8_gemm(quantize_mxfp8(normalized), quantize_mxfp8(gate_up_weight))
    hidden = torch.empty(
        x.shape[0],
        hidden_dim,
        dtype=x.dtype,
        device=x.device,
    )
    _dense_op(gate_up, "swiglu_forward")(gate_up, hidden)
    projected = mxfp8_gemm(
        quantize_mxfp8(hidden),
        quantize_mxfp8(down_weight),
    )
    _dense_op(x, "residual_add")(x, projected)
    return projected, (normalized, gate_up, hidden)
