"""Specialized MoE CUDA operators."""

from __future__ import annotations

from typing import Any

import torch

from chess_engine_4.kernels.capabilities import (
    moe_op_prefix,
    require_moe_kernel,
)
from chess_engine_4.kernels.extension import extension

EXPERT_COUNT = 64


def _moe_op(tensor: torch.Tensor, d_model: int, suffix: str):
    capability = torch.cuda.get_device_capability(tensor.device)
    prefix = moe_op_prefix(capability)
    return getattr(extension(), f"{prefix}moe_d{d_model}_{suffix}")


def moe_forward(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    route_probs: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a supported sorted, padded 64-expert BF16 MLP."""

    rows = x.shape[0]
    d_model = _supported_width(x)
    hidden = torch.empty(rows, 2 * d_model, dtype=torch.bfloat16, device=x.device)
    output = torch.empty(rows, d_model, dtype=torch.bfloat16, device=x.device)
    _moe_op(x, d_model, "forward")(
        x,
        gate_up_weight,
        down_weight,
        route_probs,
        expert_offsets,
        hidden,
        output,
    )
    return output


def moe_trainable(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    route_probs: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a supported expert MLP with its specialized BF16 backward."""

    return _MoeFunction.apply(
        x,
        gate_up_weight,
        down_weight,
        route_probs,
        expert_offsets,
    )


class _MoeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        route_probs: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        rows = x.shape[0]
        d_model = _supported_width(x)
        hidden = torch.empty(rows, 2 * d_model, dtype=torch.bfloat16, device=x.device)
        raw_output = torch.empty(rows, d_model, dtype=torch.bfloat16, device=x.device)
        output = torch.empty_like(raw_output)
        _moe_op(x, d_model, "training_forward")(
            x,
            gate_up_weight,
            down_weight,
            route_probs,
            expert_offsets,
            hidden,
            raw_output,
            output,
        )
        ctx.save_for_backward(
            x,
            gate_up_weight,
            down_weight,
            route_probs,
            expert_offsets,
            hidden,
            raw_output,
        )
        ctx.d_model = d_model
        return output

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        (grad_output,) = grad_outputs
        (
            x,
            gate_up_weight,
            down_weight,
            route_probs,
            expert_offsets,
            hidden,
            raw_output,
        ) = ctx.saved_tensors
        grad_input = torch.empty_like(x)
        grad_gate_up_weight = torch.empty_like(gate_up_weight)
        grad_down_weight = torch.empty_like(down_weight)
        grad_route_probs = torch.empty_like(route_probs)
        grad_unscaled_output = torch.empty_like(grad_output)
        grad_hidden = torch.empty_like(hidden)
        grad_gate_up = torch.empty(
            x.shape[0],
            4 * ctx.d_model,
            dtype=torch.bfloat16,
            device=x.device,
        )
        _moe_op(x, ctx.d_model, "backward")(
            x,
            gate_up_weight,
            down_weight,
            route_probs,
            expert_offsets,
            hidden,
            raw_output,
            grad_output.contiguous(),
            grad_input,
            grad_gate_up_weight,
            grad_down_weight,
            grad_route_probs,
            grad_unscaled_output,
            grad_hidden,
            grad_gate_up,
        )
        return (
            grad_input,
            grad_gate_up_weight,
            grad_down_weight,
            grad_route_probs,
            None,
        )


def _supported_width(x: torch.Tensor) -> int:
    d_model = x.shape[1]
    require_moe_kernel(
        capability=torch.cuda.get_device_capability(x.device),
        precision="bf16",
        d_model=d_model,
        hidden_dim=2 * d_model,
        rows=x.shape[0],
    )
    return d_model
