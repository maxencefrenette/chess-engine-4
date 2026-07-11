"""Portable PyTorch inference models reconstructed from TE checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from chess_engine_4.model.mlp import MlpChessNetConfig
from chess_engine_4.model.mlp_moe import MlpMoeChessNetConfig
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


class _MoeBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.num_experts_per_token = num_experts_per_token
        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.gate_up_weight = nn.Parameter(torch.empty(num_experts, 2 * hidden_dim, d_model))
        self.down_weight = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        nn.init.kaiming_uniform_(self.gate_up_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        router_probs = torch.softmax(self.router(x), dim=-1)
        route_probs, route_experts = torch.topk(
            router_probs,
            k=self.num_experts_per_token,
            dim=-1,
        )
        route_probs = route_probs / route_probs.sum(dim=-1, keepdim=True)

        gate_up_weight = self.gate_up_weight[route_experts]
        gate_up = torch.matmul(gate_up_weight, x[:, None, :, None]).squeeze(-1)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        down_weight = self.down_weight[route_experts]
        expert_output = torch.matmul(down_weight, hidden.unsqueeze(-1)).squeeze(-1)
        return residual + (expert_output * route_probs.unsqueeze(-1)).sum(dim=1)


class PortableChessNet(nn.Module):
    """LC0-plane inference model composed only of portable PyTorch operations."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.input_planes * config.board_size * config.board_size
        self.input = nn.Linear(input_dim, config.d_model)

        if isinstance(config, MlpChessNetConfig):
            hidden_dim = int(config.d_model * config.mlp_ratio)
            self.blocks = nn.ModuleList(
                [
                    _DenseBlock(config.d_model, hidden_dim, config.rms_norm_eps)
                    for _ in range(config.depth)
                ]
            )
        elif isinstance(config, MlpMoeChessNetConfig):
            hidden_dim = int(config.d_model * config.expert_mlp_ratio)
            self.blocks = nn.ModuleList(
                [
                    _MoeBlock(
                        config.d_model,
                        hidden_dim,
                        config.num_experts,
                        config.num_experts_per_token,
                        config.rms_norm_eps,
                    )
                    for _ in range(config.depth)
                ]
            )
        else:
            raise TypeError(f"unsupported model config type: {type(config).__name__}")

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


def portable_model_from_te_state_dict(
    config: ModelConfig,
    state_dict: Mapping[str, torch.Tensor],
) -> PortableChessNet:
    """Reconstruct a TE checkpoint as a portable inference-only model."""

    model = PortableChessNet(config)
    portable_state = {
        "input.weight": state_dict["input.weight"],
        "input.bias": state_dict["input.bias"],
        "norm.weight": state_dict.get("norm.weight", torch.ones(config.d_model)),
        "policy_head.weight": state_dict["policy_head.weight"][: config.policy_size],
        "policy_head.bias": state_dict["policy_head.bias"][: config.policy_size],
        "wdl_head.weight": state_dict["wdl_head.weight"][:3],
        "wdl_head.bias": state_dict["wdl_head.bias"][:3],
        "moves_left_head.weight": state_dict["moves_left_head.weight"][:1],
        "moves_left_head.bias": state_dict["moves_left_head.bias"][:1],
    }

    if isinstance(config, MlpChessNetConfig):
        for layer in range(config.depth):
            source = f"blocks.{layer}.mlp"
            target = f"blocks.{layer}"
            if f"{source}.fc1_weight" in state_dict:
                portable_state[f"{target}.norm.weight"] = state_dict[
                    f"{source}.layer_norm_weight"
                ]
                portable_state[f"{target}.gate_up.weight"] = state_dict[
                    f"{source}.fc1_weight"
                ]
                portable_state[f"{target}.down.weight"] = state_dict[f"{source}.fc2_weight"]
            else:
                legacy = f"blocks.{layer}"
                portable_state[f"{target}.norm.weight"] = torch.ones(config.d_model)
                portable_state[f"{target}.gate_up.weight"] = torch.cat(
                    [
                        state_dict[f"{legacy}.gate_proj.weight"],
                        state_dict[f"{legacy}.up_proj.weight"],
                    ]
                )
                portable_state[f"{target}.down.weight"] = state_dict[
                    f"{legacy}.down_proj.weight"
                ]
    elif isinstance(config, MlpMoeChessNetConfig):
        for layer in range(config.depth):
            source = f"blocks.{layer}"
            target = f"blocks.{layer}"
            portable_state[f"{target}.norm.weight"] = state_dict.get(
                f"{source}.norm.weight",
                torch.ones(config.d_model),
            )
            portable_state[f"{target}.router.weight"] = state_dict[
                f"{source}.router.weight"
            ][: config.num_experts]
            if f"{source}.expert_fc1.weight0" in state_dict:
                portable_state[f"{target}.gate_up_weight"] = torch.stack(
                    [
                        state_dict[f"{source}.expert_fc1.weight{expert}"]
                        for expert in range(config.num_experts)
                    ]
                )
                portable_state[f"{target}.down_weight"] = torch.stack(
                    [
                        state_dict[f"{source}.expert_fc2.weight{expert}"]
                        for expert in range(config.num_experts)
                    ]
                )
            else:
                gate = state_dict[f"{source}.gate_proj"].transpose(1, 2)
                up = state_dict[f"{source}.up_proj"].transpose(1, 2)
                portable_state[f"{target}.gate_up_weight"] = torch.cat((gate, up), dim=1)
                portable_state[f"{target}.down_weight"] = state_dict[
                    f"{source}.down_proj"
                ].transpose(1, 2)

    model.load_state_dict(portable_state)
    return model
