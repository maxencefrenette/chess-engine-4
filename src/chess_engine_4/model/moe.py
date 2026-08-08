"""Alternating dense and 64-expert, 2-active mixture-of-experts chess network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F

from chess_engine_4.data.leela import INPUT_PLANE_COUNT, POLICY_SIZE
from chess_engine_4.model.config import InputPipeline, KernelBackend, Precision
from chess_engine_4.model.dense import (
    HISTORY_LENGTH,
    PLANES_PER_HISTORY_POSITION,
    DenseBlock,
    model_input_plane_count,
    mxfp8_aligned_size,
    normalize_lc0_planes,
    select_lc0_history,
)
from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.transformer_engine import te, te_router

EXPERT_COUNT = 64
ACTIVE_EXPERT_COUNT = 2
ROUTER_OUTPUT_SIZE = mxfp8_aligned_size(EXPERT_COUNT)
DENSE_EXPANSION_RATIO = 4
MXFP8_FEATURE_ALIGNMENT = 32
MXFP8_TOKEN_ALIGNMENT = 128
# Dynamic fused dispatch amortizes its host synchronization at large widths.
FUSED_DISPATCH_MIN_WIDTH = 1024


def _route_slots(
    flat_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    route_order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[route_order]
    expert_offsets = tokens_per_expert.cumsum(dim=0) - tokens_per_expert
    sorted_slots = (
        torch.arange(flat_experts.numel(), device=flat_experts.device)
        - expert_offsets[sorted_experts]
    )
    return torch.empty_like(sorted_slots).scatter_(0, route_order, sorted_slots)


@dataclass(frozen=True, slots=True)
class Moe64A2ChessNetConfig:
    kind: str = "moe64a2"
    input_planes: int = INPUT_PLANE_COUNT
    board_size: int = 8
    policy_size: int = POLICY_SIZE
    history_length: int = HISTORY_LENGTH
    d_model: int = 1024
    depth: int = 8
    expansion_ratio: float = 2.0
    activation: str = "swiglu"
    rms_norm_eps: float = 1e-6
    precision: Precision = "mxfp8"
    kernel_backend: KernelBackend = "te"
    input_pipeline: InputPipeline = "pinned"

    num_experts: ClassVar[int] = EXPERT_COUNT
    num_active_experts: ClassVar[int] = ACTIVE_EXPERT_COUNT

    def __post_init__(self) -> None:
        if not 1 <= self.history_length <= HISTORY_LENGTH:
            raise ValueError(f"history_length must be in [1, {HISTORY_LENGTH}]")
        if self.d_model % MXFP8_FEATURE_ALIGNMENT != 0:
            raise ValueError(f"d_model must be divisible by {MXFP8_FEATURE_ALIGNMENT}")
        if self.expansion_ratio <= 0:
            raise ValueError("expansion_ratio must be positive")
        if self.depth <= 0 or self.depth % 2 != 0:
            raise ValueError("moe64a2 depth must be a positive even number")
        if self.activation != "swiglu":
            raise ValueError("moe64a2 only supports activation='swiglu'")

    @property
    def cuda_graph_compatible(self) -> bool:
        return self.d_model < FUSED_DISPATCH_MIN_WIDTH


class MoeBlock(nn.Module):
    """Pre-norm dropless top-2 MoE block using TE's fused token dispatcher."""

    def __init__(self, config: Moe64A2ChessNetConfig) -> None:
        super().__init__()
        transformer_engine = te()
        hidden_dim = int(config.d_model * config.expansion_ratio)
        if hidden_dim % MXFP8_FEATURE_ALIGNMENT != 0:
            raise ValueError(
                f"expert hidden dimension must be divisible by {MXFP8_FEATURE_ALIGNMENT}"
            )

        self.d_model = config.d_model
        self.hidden_dim = hidden_dim
        self.cuda_graph_compatible = config.cuda_graph_compatible
        self._custom_kernels_enabled = False
        self.norm = transformer_engine.RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            params_dtype=torch.bfloat16,
        )
        self.router = transformer_engine.Linear(
            config.d_model,
            # TE requires this projection's output dimension to be MXFP8-aligned.
            ROUTER_OUTPUT_SIZE,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        self.experts = transformer_engine.ops.Sequential(
            transformer_engine.ops.GroupedLinear(
                EXPERT_COUNT,
                config.d_model,
                2 * hidden_dim,
                bias=False,
                dtype=torch.bfloat16,
            ),
            transformer_engine.ops.ScaledSwiGLU(),
            transformer_engine.ops.GroupedLinear(
                EXPERT_COUNT,
                hidden_dim,
                config.d_model,
                bias=False,
                dtype=torch.bfloat16,
            ),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        router_probs, routing_map = te_router().fused_topk_with_score_function(
            self.router(x)[:, :EXPERT_COUNT].float(),
            topk=ACTIVE_EXPERT_COUNT,
            use_pre_softmax=True,
            num_groups=None,
            group_topk=None,
            scaling_factor=None,
            score_function="softmax",
            expert_bias=None,
        )
        routed, tokens_per_expert = self._run_experts(x, router_probs, routing_map)
        router_aux_loss = te_router().fused_moe_aux_loss(
            router_probs,
            tokens_per_expert,
            total_num_tokens=x.shape[0],
            num_experts=EXPERT_COUNT,
            topk=ACTIVE_EXPERT_COUNT,
            coeff=1.0,
        )
        dead_experts = tokens_per_expert.eq(0).sum()
        return residual + routed, router_aux_loss, dead_experts

    def _run_experts(
        self,
        x: torch.Tensor,
        router_probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cuda_graph_compatible:
            return self._run_experts_static(x, router_probs, routing_map)
        return self._run_experts_fused(x, router_probs, routing_map)

    def _run_experts_fused(
        self,
        x: torch.Tensor,
        router_probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens_per_expert = routing_map.sum(dim=0, dtype=torch.int64)
        (
            permuted_x,
            permuted_probs,
            row_id_map,
            pad_offsets,
            padded_tokens_per_expert,
        ) = te().moe_permute_and_pad_with_probs(
            x,
            router_probs,
            routing_map,
            tokens_per_expert,
            256,
        )
        expert_output = self.experts(
            permuted_x,
            padded_tokens_per_expert,
            permuted_probs,
            padded_tokens_per_expert,
        )
        routed = te().moe_unpermute(
            expert_output,
            row_id_map,
            restore_shape=x.shape,
            pad_offsets=pad_offsets,
        )
        return routed, tokens_per_expert

    def _run_experts_static(
        self,
        x: torch.Tensor,
        router_probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        route_experts = routing_map.to(torch.uint8).topk(ACTIVE_EXPERT_COUNT, dim=1).indices
        route_probs = router_probs.gather(1, route_experts.long())
        flat_experts = route_experts.reshape(-1).long()
        tokens_per_expert = routing_map.sum(dim=0, dtype=torch.int64)
        route_slots = _route_slots(flat_experts, tokens_per_expert)

        aligned_splits = (
            (tokens_per_expert + MXFP8_TOKEN_ALIGNMENT - 1) // MXFP8_TOKEN_ALIGNMENT
        ) * MXFP8_TOKEN_ALIGNMENT
        max_padded_tokens = batch_size * ACTIVE_EXPERT_COUNT + EXPERT_COUNT * (
            MXFP8_TOKEN_ALIGNMENT - 1
        )
        max_padded_tokens = (
            (max_padded_tokens + MXFP8_TOKEN_ALIGNMENT - 1) // MXFP8_TOKEN_ALIGNMENT
        ) * MXFP8_TOKEN_ALIGNMENT
        padding_slack = max_padded_tokens - aligned_splits.sum()
        expert_splits = aligned_splits + F.pad(padding_slack[None], (EXPERT_COUNT - 1, 0))
        expert_offsets = aligned_splits.cumsum(dim=0) - aligned_splits
        padded_positions = expert_offsets[flat_experts] + route_slots

        padded_x = x.new_zeros(max_padded_tokens, self.d_model)
        for route_index in range(ACTIVE_EXPERT_COUNT):
            padded_x.index_copy_(
                0,
                padded_positions[route_index::ACTIVE_EXPERT_COUNT],
                x,
            )
        padded_probs = x.new_zeros(max_padded_tokens)
        route_probs = route_probs.to(x.dtype)
        for route_index in range(ACTIVE_EXPERT_COUNT):
            padded_probs.index_copy_(
                0,
                padded_positions[route_index::ACTIVE_EXPERT_COUNT],
                route_probs[:, route_index],
            )
        if self._custom_kernels_enabled:
            # Trailing graph-capacity rows are storage, not expert work.
            expert_offsets = F.pad(
                aligned_splits.cumsum(dim=0, dtype=torch.int32),
                (1, 0),
            )
            expert_output = self.experts(padded_x, padded_probs, expert_offsets)
        else:
            expert_output = self.experts(
                padded_x,
                expert_splits,
                padded_probs,
                expert_splits,
            )
        routed = expert_output[padded_positions].reshape(
            batch_size,
            ACTIVE_EXPERT_COUNT,
            self.d_model,
        )
        return routed.sum(dim=1), tokens_per_expert

    def enable_custom_kernels(self) -> None:
        from chess_engine_4.kernels.capabilities import require_moe_model_shape

        if self._custom_kernels_enabled:
            return
        require_moe_model_shape(
            d_model=self.d_model,
            hidden_dim=self.hidden_dim,
            activation="swiglu",
            num_experts=EXPERT_COUNT,
            num_active_experts=ACTIVE_EXPERT_COUNT,
        )
        gate_up = self.experts[0]
        down = self.experts[2]
        self.experts = _CustomExperts(
            torch.stack(
                [getattr(gate_up, f"weight{expert}").detach() for expert in range(EXPERT_COUNT)]
            ),
            torch.stack(
                [getattr(down, f"weight{expert}").detach() for expert in range(EXPERT_COUNT)]
            ),
        )
        self._custom_kernels_enabled = True


class _CustomExperts(nn.Module):
    def __init__(self, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> None:
        super().__init__()
        self.gate_up_weight = nn.Parameter(gate_up_weight)
        self.down_weight = nn.Parameter(down_weight)

    def forward(
        self,
        x: torch.Tensor,
        route_probs: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        from chess_engine_4.kernels import moe_trainable

        return moe_trainable(
            x,
            self.gate_up_weight,
            self.down_weight,
            route_probs,
            expert_offsets,
        )


class Moe64A2ChessNet(nn.Module):
    """Single-token chess model alternating dense and 64-expert layers."""

    def __init__(self, config: Moe64A2ChessNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = Moe64A2ChessNetConfig()
        self.config = config
        self.cuda_graph_compatible = config.cuda_graph_compatible
        input_dim = model_input_plane_count(config.history_length) * config.board_size**2
        transformer_engine = te()

        self.input = transformer_engine.Linear(
            input_dim,
            config.d_model,
            params_dtype=torch.bfloat16,
        )
        self.blocks = nn.ModuleList(
            [
                MoeBlock(config)
                if layer_index % 2 == 0
                else DenseBlock(
                    d_model=config.d_model,
                    hidden_dim=DENSE_EXPANSION_RATIO * config.d_model,
                    rms_norm_eps=config.rms_norm_eps,
                    activation=config.activation,
                    precision=config.precision,
                )
                for layer_index in range(config.depth)
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
        x = select_lc0_history(planes, self.config.history_length)
        rule50_plane_index = self.config.history_length * PLANES_PER_HISTORY_POSITION + 5
        x = normalize_lc0_planes(x, rule50_plane_index=rule50_plane_index).flatten(start_dim=1)
        x = self.input(x)
        router_aux_losses = []
        dead_experts = []
        for block in self.blocks:
            if isinstance(block, MoeBlock):
                x, router_aux_loss, block_dead_experts = block(x)
                router_aux_losses.append(router_aux_loss)
                dead_experts.append(block_dead_experts)
            else:
                x = block(x)
        x = self.norm(x)
        return ChessNetOutput(
            policy_logits=self.policy_head(x)[:, : self.config.policy_size],
            wdl_logits=self.wdl_head(x)[:, :3],
            moves_left=self.moves_left_head(x)[:, 0],
            router_aux_loss=torch.stack(router_aux_losses).mean(),
            router_dead_experts=torch.stack(dead_experts).max(),
        )

    def enable_custom_kernels(self) -> None:
        for block in self.blocks:
            if isinstance(block, MoeBlock):
                block.enable_custom_kernels()


def moe64a2_parameter_count(
    *,
    input_planes: int = INPUT_PLANE_COUNT,
    history_length: int = HISTORY_LENGTH,
    board_size: int = 8,
    policy_size: int = POLICY_SIZE,
    d_model: int,
    depth: int,
    expansion_ratio: float = 2.0,
) -> int:
    if depth <= 0 or depth % 2 != 0:
        raise ValueError("moe64a2 depth must be a positive even number")
    auxiliary_planes = input_planes - HISTORY_LENGTH * PLANES_PER_HISTORY_POSITION
    selected_planes = history_length * PLANES_PER_HISTORY_POSITION + auxiliary_planes
    input_dim = selected_planes * board_size * board_size
    hidden_dim = int(d_model * expansion_ratio)
    input_params = input_dim * d_model + d_model
    moe_depth = depth // 2
    dense_depth = depth // 2
    expert_params = moe_depth * EXPERT_COUNT * 3 * d_model * hidden_dim
    router_params = moe_depth * d_model * ROUTER_OUTPUT_SIZE
    dense_params = dense_depth * 3 * d_model * (DENSE_EXPANSION_RATIO * d_model)
    norm_params = (depth + 1) * d_model
    aligned_policy_size = mxfp8_aligned_size(policy_size)
    policy_params = d_model * aligned_policy_size + aligned_policy_size
    wdl_params = d_model * 32 + 32
    moves_left_params = d_model * 32 + 32
    return (
        input_params
        + expert_params
        + router_params
        + dense_params
        + norm_params
        + policy_params
        + wdl_params
        + moves_left_params
    )
