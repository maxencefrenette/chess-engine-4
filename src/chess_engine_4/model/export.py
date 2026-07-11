"""Portable dense model used for meta-device profiling and CPU tests."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.registry import ModelConfig


class _DenseBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.gate_up = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(self.norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class PortableChessNet(nn.Module):
    """LC0-plane dense model composed only of portable PyTorch operations."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.input_planes * config.board_size * config.board_size
        self.input = nn.Linear(input_dim, config.d_model)

        hidden_dim = int(config.d_model * config.expansion_ratio)
        self.blocks = nn.ModuleList(
            [
                _DenseBlock(config.d_model, hidden_dim, config.rms_norm_eps)
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.policy_head = nn.Linear(config.d_model, config.policy_size)
        self.wdl_head = nn.Linear(config.d_model, 3)
        self.moves_left_head = nn.Linear(config.d_model, 1)

    def forward(self, planes: torch.Tensor) -> ChessNetOutput:
        x = self.input(planes.flatten(start_dim=1))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return ChessNetOutput(
            policy_logits=self.policy_head(x),
            wdl_logits=self.wdl_head(x),
            moves_left=self.moves_left_head(x).squeeze(-1),
        )
