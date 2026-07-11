"""MLP-MoE chess network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_engine_4.data.leela import INPUT_PLANE_COUNT, POLICY_SIZE
from chess_engine_4.model.mlp import mxfp8_aligned_size
from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.transformer_engine import te


@dataclass(frozen=True, slots=True)
class MlpMoeChessNetConfig:
    kind: str = "mlp_moe"
    input_planes: int = INPUT_PLANE_COUNT
    board_size: int = 8
    policy_size: int = POLICY_SIZE
    d_model: int = 1024
    depth: int = 8
    num_experts: int = 16
    num_experts_per_token: int = 2
    expert_mlp_ratio: float = 4.0
    rms_norm_eps: float = 1e-6


class MoeBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if not 0 < num_experts_per_token <= num_experts:
            raise ValueError("num_experts_per_token must be in [1, num_experts].")
        if d_model % 32 or hidden_dim % 32:
            raise ValueError("MXFP8 MoE requires d_model and hidden_dim divisible by 32.")

        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        transformer_engine = te()
        self.norm = transformer_engine.RMSNorm(
            d_model,
            eps=rms_norm_eps,
            params_dtype=torch.bfloat16,
        )
        self.router = transformer_engine.Linear(
            d_model,
            mxfp8_aligned_size(num_experts),
            bias=False,
            params_dtype=torch.bfloat16,
        )
        self.expert_fc1 = transformer_engine.GroupedLinear(
            num_experts,
            d_model,
            2 * hidden_dim,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        self.expert_fc2 = transformer_engine.GroupedLinear(
            num_experts,
            hidden_dim,
            d_model,
            bias=False,
            params_dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        router_logits = self.router(x)[:, : self.num_experts]
        router_probs = torch.softmax(router_logits, dim=-1)
        route_probs, route_experts = torch.topk(
            router_probs,
            k=self.num_experts_per_token,
            dim=-1,
        )
        route_probs = route_probs / route_probs.sum(dim=-1, keepdim=True)

        routing_probs = torch.zeros_like(router_probs).scatter(1, route_experts, route_probs)
        routing_map = routing_probs.ne(0).to(torch.int32)
        routed = self._run_experts(x, routing_map, routing_probs)
        expert_fraction = routing_map.float().mean(dim=0).to(router_probs.dtype)
        balanced_fraction = expert_fraction / self.num_experts_per_token
        load_balancing_loss = self.num_experts * torch.sum(
            router_probs.mean(dim=0) * balanced_fraction
        )
        dead_experts = (expert_fraction == 0).sum().to(router_probs.dtype)
        return residual + routed, load_balancing_loss, dead_experts

    def _run_experts(
        self,
        x: torch.Tensor,
        routing_map: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> torch.Tensor:
        transformer_engine = te()
        tokens_per_expert = routing_map.sum(dim=0, dtype=torch.int64)
        permuted_x, permuted_probs, row_id_map = transformer_engine.moe_permute_with_probs(
            x,
            routing_probs,
            routing_map,
            x.shape[0] * self.num_experts_per_token,
        )
        counts = tokens_per_expert.tolist()
        padded_counts = [((count + 31) // 32) * 32 for count in counts]
        x_chunks = torch.split(permuted_x, counts)
        prob_chunks = torch.split(permuted_probs, counts)
        padded_x = torch.cat(
            [
                torch.nn.functional.pad(chunk, (0, 0, 0, padded - count))
                for chunk, count, padded in zip(x_chunks, counts, padded_counts, strict=True)
            ]
        )
        padded_probs = torch.cat(
            [
                torch.nn.functional.pad(chunk, (0, padded - count))
                for chunk, count, padded in zip(prob_chunks, counts, padded_counts, strict=True)
            ]
        )
        gate, up = self.expert_fc1(padded_x, padded_counts).chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate) * up
        hidden = hidden * padded_probs.unsqueeze(-1)
        padded_output = self.expert_fc2(hidden, padded_counts)
        permuted_output = torch.cat(
            [
                chunk[:count]
                for chunk, count in zip(
                    torch.split(padded_output, padded_counts),
                    counts,
                    strict=True,
                )
            ]
        )
        return transformer_engine.moe_unpermute(
            permuted_output,
            row_id_map,
            restore_shape=x.shape,
        )


class MlpMoeChessNet(nn.Module):
    """Single-token MLP model with MoE SwiGLU blocks."""

    def __init__(self, config: MlpMoeChessNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = MlpMoeChessNetConfig()
        self.config = config
        input_dim = config.input_planes * config.board_size * config.board_size
        hidden_dim = int(config.d_model * config.expert_mlp_ratio)
        transformer_engine = te()

        self.input = transformer_engine.Linear(
            input_dim,
            config.d_model,
            params_dtype=torch.bfloat16,
        )
        self.blocks = nn.ModuleList(
            [
                MoeBlock(
                    d_model=config.d_model,
                    hidden_dim=hidden_dim,
                    num_experts=config.num_experts,
                    num_experts_per_token=config.num_experts_per_token,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = transformer_engine.RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            params_dtype=torch.bfloat16,
        )
        self.policy_head = transformer_engine.Linear(
            config.d_model,
            mxfp8_aligned_size(config.policy_size),
            params_dtype=torch.bfloat16,
        )
        self.wdl_head = transformer_engine.Linear(
            config.d_model,
            32,
            params_dtype=torch.bfloat16,
        )
        self.moves_left_head = transformer_engine.Linear(
            config.d_model,
            32,
            params_dtype=torch.bfloat16,
        )

    def forward(self, planes: torch.Tensor) -> ChessNetOutput:
        x = self.input(planes.flatten(start_dim=1))
        aux_losses = []
        dead_experts = []
        for block in self.blocks:
            x, aux_loss, block_dead_experts = block(x)
            aux_losses.append(aux_loss)
            dead_experts.append(block_dead_experts)
        x = self.norm(x)
        router_dead_experts_by_layer = torch.stack(dead_experts)
        return ChessNetOutput(
            policy_logits=self.policy_head(x)[:, : self.config.policy_size],
            wdl_logits=self.wdl_head(x)[:, :3],
            moves_left=self.moves_left_head(x)[:, 0],
            aux_loss=torch.stack(aux_losses).mean(),
            router_dead_experts=router_dead_experts_by_layer.mean(),
            router_dead_experts_max=router_dead_experts_by_layer.max(),
        )


def mlp_moe_parameter_count(
    *,
    input_planes: int = INPUT_PLANE_COUNT,
    board_size: int = 8,
    policy_size: int = POLICY_SIZE,
    d_model: int,
    depth: int,
    num_experts: int = 8,
    expert_mlp_ratio: float = 4.0,
) -> int:
    input_dim = input_planes * board_size * board_size
    hidden_dim = int(d_model * expert_mlp_ratio)
    input_params = input_dim * d_model + d_model
    expert_params = depth * num_experts * (3 * d_model * hidden_dim)
    router_params = depth * d_model * mxfp8_aligned_size(num_experts)
    norm_params = (depth + 1) * d_model
    aligned_policy_size = mxfp8_aligned_size(policy_size)
    policy_params = d_model * aligned_policy_size + aligned_policy_size
    wdl_params = d_model * 32 + 32
    moves_left_params = d_model * 32 + 32
    return (
        input_params
        + expert_params
        + router_params
        + norm_params
        + policy_params
        + wdl_params
        + moves_left_params
    )
