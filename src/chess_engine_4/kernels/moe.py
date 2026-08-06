"""Specialized MoE CUDA operators."""

from __future__ import annotations

import torch

from chess_engine_4.kernels.extension import extension

EXPERT_COUNT = 64
SUPPORTED_MOE_WIDTHS = frozenset({128, 256, 512})
TOKEN_ALIGNMENT = 16


def moe_forward(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    route_probs: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a supported sorted, padded 64-expert BF16 MLP on SM120."""

    rows = x.shape[0]
    d_model = _supported_width(x)
    hidden = torch.empty(rows, 2 * d_model, dtype=torch.bfloat16, device=x.device)
    output = torch.empty(rows, d_model, dtype=torch.bfloat16, device=x.device)
    getattr(extension(), f"moe_d{d_model}_forward")(
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
        ctx: object,
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
        getattr(extension(), f"moe_d{d_model}_training_forward")(
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
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
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
        getattr(extension(), f"moe_d{ctx.d_model}_backward")(
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
    if d_model not in SUPPORTED_MOE_WIDTHS:
        raise ValueError(
            f"custom MoE kernels require d_model in {sorted(SUPPORTED_MOE_WIDTHS)}, "
            f"got {d_model}"
        )
    return d_model
