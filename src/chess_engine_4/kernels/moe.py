"""Specialized MoE CUDA operators."""

from __future__ import annotations

import torch

from chess_engine_4.kernels.extension import extension

EXPERT_COUNT = 64
D_MODEL = 128
HIDDEN_DIM = 256
GATE_UP_DIM = 2 * HIDDEN_DIM
TOKEN_ALIGNMENT = 16


def moe_d128_forward(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    route_probs: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a sorted, padded 64-expert d128 BF16 expert MLP on SM120."""

    rows = x.shape[0]
    hidden = torch.empty(rows, HIDDEN_DIM, dtype=torch.bfloat16, device=x.device)
    output = torch.empty(rows, D_MODEL, dtype=torch.bfloat16, device=x.device)
    extension().moe_d128_forward(
        x,
        gate_up_weight,
        down_weight,
        route_probs,
        expert_offsets,
        hidden,
        output,
    )
    return output


def moe_d128_trainable(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    route_probs: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the d128 expert MLP with its specialized BF16 backward."""

    return _MoeD128Function.apply(
        x,
        gate_up_weight,
        down_weight,
        route_probs,
        expert_offsets,
    )


class _MoeD128Function(torch.autograd.Function):
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
        hidden = torch.empty(rows, HIDDEN_DIM, dtype=torch.bfloat16, device=x.device)
        raw_output = torch.empty(rows, D_MODEL, dtype=torch.bfloat16, device=x.device)
        output = torch.empty_like(raw_output)
        extension().moe_d128_training_forward(
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
            GATE_UP_DIM,
            dtype=torch.bfloat16,
            device=x.device,
        )
        extension().moe_d128_backward(
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
