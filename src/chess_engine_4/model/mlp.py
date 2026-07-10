"""MLP-only chess network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_engine_4.data.leela import INPUT_PLANE_COUNT, POLICY_SIZE
from chess_engine_4.model.output import ChessNetOutput


@dataclass(frozen=True, slots=True)
class MlpChessNetConfig:
    kind: str = "mlp"
    input_planes: int = INPUT_PLANE_COUNT
    board_size: int = 8
    policy_size: int = POLICY_SIZE
    d_model: int = 1024
    depth: int = 8
    mlp_ratio: float = 4.0
    rms_norm_eps: float = 1e-6


class MlpBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, rms_norm_eps: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=rms_norm_eps, elementwise_affine=False)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return residual + self.down_proj(x)


class MlpChessNet(nn.Module):
    """Single-token MLP model over flattened LCZero input planes."""

    def __init__(self, config: MlpChessNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = MlpChessNetConfig()
        self.config = config
        input_dim = config.input_planes * config.board_size * config.board_size
        hidden_dim = int(config.d_model * config.mlp_ratio)

        self.input = nn.Linear(input_dim, config.d_model)
        self.blocks = nn.Sequential(
            *[
                MlpBlock(
                    d_model=config.d_model,
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
        self.policy_head = nn.Linear(config.d_model, config.policy_size)
        self.wdl_head = nn.Linear(config.d_model, 3)
        self.moves_left_head = nn.Linear(config.d_model, 1)

    def forward(self, planes: torch.Tensor) -> ChessNetOutput:
        x = planes.flatten(start_dim=1)
        x = self.input(x)
        x = self.blocks(x)
        x = self.norm(x)
        return ChessNetOutput(
            policy_logits=self.policy_head(x),
            wdl_logits=self.wdl_head(x),
            moves_left=self.moves_left_head(x).squeeze(-1),
        )


def mlp_parameter_count(
    *,
    input_planes: int = INPUT_PLANE_COUNT,
    board_size: int = 8,
    policy_size: int = POLICY_SIZE,
    d_model: int,
    depth: int,
    mlp_ratio: float = 4.0,
) -> int:
    input_dim = input_planes * board_size * board_size
    hidden_dim = int(d_model * mlp_ratio)
    block_params = depth * (3 * d_model * hidden_dim)
    input_params = input_dim * d_model + d_model
    policy_params = d_model * policy_size + policy_size
    wdl_params = d_model * 3 + 3
    moves_left_params = d_model + 1
    return (
        input_params
        + block_params
        + policy_params
        + wdl_params
        + moves_left_params
    )
