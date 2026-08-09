"""Portable dense model used for meta-device profiling and CPU tests."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from chess_engine_4.data.leela import BOARD_SIZE, POLICY_SIZE
from chess_engine_4.model.dense import (
    GATED_ACTIVATIONS,
    PLANES_PER_HISTORY_POSITION,
    DenseChessNetConfig,
    model_input_plane_count,
    normalize_lc0_planes,
    select_lc0_history,
)
from chess_engine_4.model.moe import (
    ACTIVE_EXPERT_COUNT,
    DENSE_EXPANSION_RATIO,
    EXPERT_COUNT,
    ROUTER_OUTPUT_SIZE,
    Moe64A2ChessNetConfig,
)
from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.registry import ModelConfig


class _DenseBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, eps: float, activation: str) -> None:
        super().__init__()
        self.activation = activation
        self.norm = nn.RMSNorm(d_model, eps=eps)
        projection_size = 2 * hidden_dim if activation in GATED_ACTIVATIONS else hidden_dim
        self.gate_up = nn.Linear(d_model, projection_size, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.gate_up(self.norm(x))
        if self.activation == "swiglu":
            gate, up = projected.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
        elif self.activation == "geglu":
            gate, up = projected.chunk(2, dim=-1)
            hidden = F.gelu(gate) * up
        elif self.activation == "gelu":
            hidden = F.gelu(projected)
        elif self.activation == "silu":
            hidden = F.silu(projected)
        elif self.activation == "srelu":
            hidden = F.relu(projected).square()
        else:
            raise ValueError(f"unsupported activation: {self.activation}")
        return x + self.down(hidden)


class PortableChessNet(nn.Module):
    """LC0-plane dense model composed only of portable PyTorch operations."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(config, DenseChessNetConfig):
            raise TypeError("PortableChessNet only supports dense models.")
        self.config = config
        input_dim = model_input_plane_count(config.history_length) * BOARD_SIZE**2
        self.input = nn.Linear(input_dim, config.d_model)

        hidden_dim = int(config.d_model * config.expansion_ratio)
        self.blocks = nn.ModuleList(
            [
                _DenseBlock(
                    config.d_model,
                    hidden_dim,
                    config.rms_norm_eps,
                    config.activation,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.policy_head = nn.Linear(config.d_model, POLICY_SIZE)
        self.wdl_head = nn.Linear(config.d_model, 3)
        self.moves_left_head = nn.Linear(config.d_model, 1)

    def forward(self, planes: torch.Tensor) -> ChessNetOutput:
        x = select_lc0_history(planes, self.config.history_length)
        rule50_plane_index = self.config.history_length * PLANES_PER_HISTORY_POSITION + 5
        x = self.input(
            normalize_lc0_planes(x, rule50_plane_index=rule50_plane_index).flatten(start_dim=1)
        )
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return ChessNetOutput(
            policy_logits=self.policy_head(x),
            wdl_logits=self.wdl_head(x),
            moves_left=self.moves_left_head(x).squeeze(-1),
        )


class _PortableMoeFlopsBlock(nn.Module):
    """Portable surrogate that executes the same number of active expert projections."""

    def __init__(self, config: Moe64A2ChessNetConfig) -> None:
        super().__init__()
        hidden_dim = int(config.d_model * config.expansion_ratio)
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.router = nn.Linear(config.d_model, ROUTER_OUTPUT_SIZE, bias=False)
        self.experts = nn.ModuleList(
            [
                _DenseBlock(
                    config.d_model,
                    hidden_dim,
                    config.rms_norm_eps,
                    config.activation,
                )
                for _ in range(ACTIVE_EXPERT_COUNT)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        route_probs = self.router(normalized)[:, :EXPERT_COUNT].softmax(dim=-1)
        route_probs = route_probs[:, :ACTIVE_EXPERT_COUNT]
        route_probs = route_probs / route_probs.sum(dim=-1, keepdim=True)
        expert_outputs = torch.stack(
            [
                expert.down(_activate(expert.gate_up(normalized), expert.activation))
                for expert in self.experts
            ],
            dim=1,
        )
        return x + (expert_outputs * route_probs.unsqueeze(-1)).sum(dim=1)


def _activate(projected: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "swiglu":
        gate, up = projected.chunk(2, dim=-1)
        return F.silu(gate) * up
    raise ValueError(f"unsupported MoE activation: {activation}")


class PortableMoeFlopsNet(nn.Module):
    """Meta-device MoE surrogate for active training-FLOP measurement only."""

    def __init__(self, config: Moe64A2ChessNetConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = model_input_plane_count(config.history_length) * BOARD_SIZE**2
        self.input = nn.Linear(input_dim, config.d_model)
        self.blocks = nn.ModuleList(
            [
                _PortableMoeFlopsBlock(config)
                if layer_index % 2 == 0
                else _DenseBlock(
                    config.d_model,
                    DENSE_EXPANSION_RATIO * config.d_model,
                    config.rms_norm_eps,
                    config.activation,
                )
                for layer_index in range(config.depth)
            ]
        )
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.policy_head = nn.Linear(config.d_model, POLICY_SIZE)
        self.wdl_head = nn.Linear(config.d_model, 3)
        self.moves_left_head = nn.Linear(config.d_model, 1)

    def forward(self, planes: torch.Tensor) -> ChessNetOutput:
        x = select_lc0_history(planes, self.config.history_length)
        rule50_plane_index = self.config.history_length * PLANES_PER_HISTORY_POSITION + 5
        x = self.input(
            normalize_lc0_planes(x, rule50_plane_index=rule50_plane_index).flatten(start_dim=1)
        )
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return ChessNetOutput(
            policy_logits=self.policy_head(x),
            wdl_logits=self.wdl_head(x),
            moves_left=self.moves_left_head(x).squeeze(-1),
        )
