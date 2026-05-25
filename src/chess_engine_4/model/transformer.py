"""64-token transformer chess network."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from chess_engine_4.data.leela import INPUT_PLANE_COUNT, POLICY_SIZE
from chess_engine_4.model.embeddings import SquareInputEmbedding
from chess_engine_4.model.heads import (
    AttentionPolicyHead,
    AttentionPolicyHeadConfig,
    PooledMovesLeftHead,
    PooledValueHead,
)
from chess_engine_4.model.output import ChessNetOutput


@dataclass(frozen=True, slots=True)
class Transformer64ChessNetConfig:
    kind: str = "transformer64"
    input_planes: int = INPUT_PLANE_COUNT
    board_size: int = 8
    policy_size: int = POLICY_SIZE
    d_model: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    rms_norm_eps: float = 1e-6
    learned_square_embeddings: bool = True
    policy: AttentionPolicyHeadConfig = field(default_factory=AttentionPolicyHeadConfig)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model, eps=rms_norm_eps, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.mlp_norm = nn.RMSNorm(d_model, eps=rms_norm_eps, elementwise_affine=False)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = residual + attn_output

        residual = x
        x = self.mlp_norm(x)
        x = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return residual + self.down_proj(x)


class Transformer64ChessNet(nn.Module):
    """Vanilla attention model with one token per board square."""

    def __init__(self, config: Transformer64ChessNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = Transformer64ChessNetConfig()
        if config.board_size != 8:
            raise ValueError("Transformer64ChessNet expects board_size=8.")
        if config.policy_size != POLICY_SIZE:
            raise ValueError("Transformer64ChessNet expects LC0 policy_size=1858.")
        if config.policy.kind != "attention":
            raise ValueError("Transformer64ChessNet only supports policy.kind='attention'.")
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.config = config
        hidden_dim = int(config.d_model * config.mlp_ratio)

        self.input = SquareInputEmbedding(
            input_planes=config.input_planes,
            board_size=config.board_size,
            d_model=config.d_model,
            learned_square_embeddings=config.learned_square_embeddings,
        )
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    hidden_dim=hidden_dim,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            elementwise_affine=False,
        )
        self.policy_head = AttentionPolicyHead(
            config.d_model,
            config=config.policy,
        )
        self.wdl_head = PooledValueHead(config.d_model)
        self.moves_left_head = PooledMovesLeftHead(config.d_model)

    def forward(self, planes: torch.Tensor) -> ChessNetOutput:
        x = self.input(planes)
        x = self.blocks(x)
        x = self.norm(x)
        return ChessNetOutput(
            policy_logits=self.policy_head(x),
            wdl_logits=self.wdl_head(x),
            moves_left=self.moves_left_head(x),
        )


def transformer64_parameter_count(
    *,
    input_planes: int = INPUT_PLANE_COUNT,
    board_size: int = 8,
    d_model: int,
    depth: int,
    mlp_ratio: float = 4.0,
) -> int:
    if board_size != 8:
        raise ValueError("Transformer64 parameter counting expects board_size=8.")
    hidden_dim = int(d_model * mlp_ratio)
    input_params = input_planes * d_model + d_model + board_size * board_size * d_model
    block_params = depth * (4 * d_model * d_model + 3 * d_model * hidden_dim)
    policy_params = d_model * d_model + d_model + 2 * (d_model * d_model + d_model) + 4 * d_model
    wdl_params = d_model * 3 + 3
    moves_left_params = d_model + 1
    return (
        input_params
        + block_params
        + policy_params
        + wdl_params
        + moves_left_params
    )
