"""Reusable chess network heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from chess_engine_4.data.leela import POLICY_SIZE
from chess_engine_4.model.policy_map import ATTENTION_POLICY_MAP


@dataclass(frozen=True, slots=True)
class DensePolicyHeadConfig:
    kind: str = "dense"


@dataclass(frozen=True, slots=True)
class AttentionPolicyHeadConfig:
    kind: str = "attention"
    embedding_size: int | None = None
    d_model: int | None = None


class DensePolicyHead(nn.Module):
    def __init__(self, d_model: int, policy_size: int = POLICY_SIZE) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, policy_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AttentionPolicyHead(nn.Module):
    """LC0-style attention policy head."""

    def __init__(
        self,
        input_dim: int,
        *,
        config: AttentionPolicyHeadConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = AttentionPolicyHeadConfig()
        embedding_size = config.embedding_size or input_dim
        policy_d_model = config.d_model or embedding_size
        self.tokens = nn.Linear(input_dim, embedding_size)
        self.q = nn.Linear(embedding_size, policy_d_model)
        self.k = nn.Linear(embedding_size, policy_d_model)
        self.promotion = nn.Linear(policy_d_model, 4, bias=False)
        self.scale = math.sqrt(policy_d_model)
        self.register_buffer(
            "policy_map",
            torch.tensor(ATTENTION_POLICY_MAP, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(self.tokens(x))
        q = self.q(x)
        k = self.k(x)
        qk = torch.matmul(q, k.transpose(-1, -2))

        promotion_offsets = self.promotion(k[:, -8:]).transpose(1, 2) * self.scale
        promotion_offsets = promotion_offsets[:, :3] + promotion_offsets[:, 3:4]
        base_promotion_logits = qk[:, -16:-8, -8:]
        promotion_logits = (
            base_promotion_logits.unsqueeze(-1) + promotion_offsets.transpose(1, 2).unsqueeze(1)
        )

        attention_logits = qk.flatten(start_dim=1) / self.scale
        promotion_logits = promotion_logits.reshape(x.shape[0], 8 * 24) / self.scale
        logits = torch.cat([attention_logits, promotion_logits], dim=-1)
        return logits.index_select(dim=-1, index=self.policy_map)


class PooledValueHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens.mean(dim=1))


class PooledMovesLeftHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens.mean(dim=1)).squeeze(-1)
