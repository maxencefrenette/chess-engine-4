"""Shared model output contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ChessNetOutput:
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor
    router_aux_loss: torch.Tensor | None = None
    router_dead_experts: torch.Tensor | None = None
