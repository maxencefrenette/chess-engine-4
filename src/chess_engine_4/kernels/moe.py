"""Specialized MoE CUDA operators."""

from __future__ import annotations

import torch

from chess_engine_4.kernels.extension import extension

EXPERT_COUNT = 64
D_MODEL = 128
HIDDEN_DIM = 256
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
