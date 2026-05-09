"""LCZero-style training losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_engine_4.model.mlp import MlpChessNetOutput

ROOT_VALUE_INDEX = 4
MOVES_LEFT_SCALE = 20.0
MOVES_LEFT_HUBER_DELTA = 10.0 / MOVES_LEFT_SCALE


@dataclass(frozen=True, slots=True)
class LossWeights:
    policy: float = 1.0
    value: float = 1.0
    moves_left: float = 1.0


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    moves_left: torch.Tensor


def policy_cross_entropy(policy_logits: torch.Tensor, policy_target: torch.Tensor) -> torch.Tensor:
    """Soft-label policy cross entropy with LCZero illegal-move masking."""

    legal = policy_target >= 0
    masked_logits = policy_logits.masked_fill(~legal, torch.finfo(policy_logits.dtype).min)
    targets = policy_target.relu()
    log_probs = nn.functional.log_softmax(masked_logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def wdl_target_from_q_d(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Convert LCZero Q/D labels to WDL probabilities."""

    win = (1.0 + q - d) / 2.0
    loss = (1.0 - q - d) / 2.0
    return torch.stack([win, d, loss], dim=-1).clamp_min(0.0)


def value_cross_entropy(wdl_logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    root = values[:, ROOT_VALUE_INDEX]
    targets = wdl_target_from_q_d(root[:, 0], root[:, 1])
    log_probs = nn.functional.log_softmax(wdl_logits, dim=-1)
    return -(targets.detach() * log_probs).sum(dim=-1).mean()


def moves_left_loss(moves_left: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    target = values[:, ROOT_VALUE_INDEX, 2]
    return nn.functional.huber_loss(
        moves_left / MOVES_LEFT_SCALE,
        target / MOVES_LEFT_SCALE,
        delta=MOVES_LEFT_HUBER_DELTA,
    )


def lczero_loss(
    output: MlpChessNetOutput,
    policy_target: torch.Tensor,
    values: torch.Tensor,
    *,
    weights: LossWeights | None = None,
) -> LossBreakdown:
    if weights is None:
        weights = LossWeights()
    policy = policy_cross_entropy(output.policy_logits, policy_target)
    value = value_cross_entropy(output.wdl_logits, values)
    mlh = moves_left_loss(output.moves_left, values)
    total = weights.policy * policy + weights.value * value + weights.moves_left * mlh
    return LossBreakdown(total=total, policy=policy, value=value, moves_left=mlh)
