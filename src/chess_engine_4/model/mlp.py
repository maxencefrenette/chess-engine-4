"""MLP-only chess network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_engine_4.data.leela import INPUT_PLANE_COUNT, POLICY_SIZE
from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.transformer_engine import te


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
        transformer_engine = te()
        self.mlp = transformer_engine.LayerNormMLP(
            d_model,
            hidden_dim,
            eps=rms_norm_eps,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class MlpChessNet(nn.Module):
    """Single-token MLP model over flattened LCZero input planes."""

    def __init__(self, config: MlpChessNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = MlpChessNetConfig()
        self.config = config
        input_dim = config.input_planes * config.board_size * config.board_size
        hidden_dim = int(config.d_model * config.mlp_ratio)
        transformer_engine = te()

        self.input = transformer_engine.Linear(
            input_dim,
            config.d_model,
        )
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
        self.norm = transformer_engine.RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
        )
        self.policy_head = transformer_engine.Linear(
            config.d_model,
            config.policy_size,
        )
        self.wdl_head = transformer_engine.Linear(
            config.d_model,
            3,
        )
        self.moves_left_head = transformer_engine.Linear(
            config.d_model,
            1,
        )

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
    norm_params = (depth + 1) * d_model
    input_params = input_dim * d_model + d_model
    policy_params = d_model * policy_size + policy_size
    wdl_params = d_model * 3 + 3
    moves_left_params = d_model + 1
    return (
        input_params
        + block_params
        + norm_params
        + policy_params
        + wdl_params
        + moves_left_params
    )
