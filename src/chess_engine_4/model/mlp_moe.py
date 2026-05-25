"""MLP-MoE chess network."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from chess_engine_4.data.leela import INPUT_PLANE_COUNT, POLICY_SIZE
from chess_engine_4.model.heads import DensePolicyHeadConfig
from chess_engine_4.model.output import ChessNetOutput


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
    policy: DensePolicyHeadConfig = field(default_factory=DensePolicyHeadConfig)


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

        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.norm = nn.RMSNorm(d_model, eps=rms_norm_eps, elementwise_affine=False)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.gate_proj = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        self.up_proj = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert in range(self.num_experts):
            nn.init.kaiming_uniform_(self.gate_proj[expert], a=5**0.5)
            nn.init.kaiming_uniform_(self.up_proj[expert], a=5**0.5)
            nn.init.kaiming_uniform_(self.down_proj[expert], a=5**0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1)
        route_probs, route_experts = torch.topk(
            router_probs,
            k=self.num_experts_per_token,
            dim=-1,
        )
        route_probs = route_probs / route_probs.sum(dim=-1, keepdim=True)

        routed = self._run_experts(x, route_experts, route_probs)
        load_balancing_loss, dead_experts = self._router_metrics(router_probs, route_experts)
        return residual + routed, load_balancing_loss, dead_experts

    @torch.compiler.disable
    def _run_experts(
        self,
        x: torch.Tensor,
        route_experts: torch.Tensor,
        route_probs: torch.Tensor,
    ) -> torch.Tensor:
        if x.device.type == "meta":
            return self._run_experts_meta(x, route_probs)

        batch_size, d_model = x.shape
        flat_experts = route_experts.reshape(-1)
        flat_probs = route_probs.reshape(-1)
        flat_tokens = torch.arange(batch_size, device=x.device).repeat_interleave(
            self.num_experts_per_token
        )
        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts[order]
        sorted_tokens = flat_tokens[order]
        sorted_probs = flat_probs[order]
        sorted_x = x[sorted_tokens]

        active_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        offsets = torch.cumsum(counts, dim=0).to(torch.int32)

        gate = F.grouped_mm(
            sorted_x,
            self.gate_proj.to(sorted_x.dtype)[active_experts],
            offs=offsets,
        )
        up = F.grouped_mm(sorted_x, self.up_proj.to(sorted_x.dtype)[active_experts], offs=offsets)
        hidden = F.silu(gate) * up
        down = F.grouped_mm(hidden, self.down_proj.to(hidden.dtype)[active_experts], offs=offsets)
        down = down * sorted_probs[:, None]

        output = torch.zeros(batch_size, d_model, device=x.device, dtype=down.dtype)
        output.index_add_(0, sorted_tokens, down)
        return output

    def _run_experts_meta(self, x: torch.Tensor, route_probs: torch.Tensor) -> torch.Tensor:
        """Profile MoE FLOPs on meta without data-dependent routing ops.

        The real route uses unique_consecutive to build grouped_mm offsets, but that op has no
        meta kernel. This path keeps the grouped_mm shapes visible to the profiler while avoiding
        route-dependent tensor values that do not exist on the meta device.
        """
        batch_size, d_model = x.shape
        sorted_x = x.to(torch.bfloat16).repeat_interleave(self.num_experts_per_token, dim=0)
        offsets = torch.empty(1, device=x.device, dtype=torch.int32)

        gate = F.grouped_mm(
            sorted_x,
            self.gate_proj[:1].to(sorted_x.dtype),
            offs=offsets,
        )
        up = F.grouped_mm(sorted_x, self.up_proj[:1].to(sorted_x.dtype), offs=offsets)
        hidden = F.silu(gate) * up
        down = F.grouped_mm(hidden, self.down_proj[:1].to(hidden.dtype), offs=offsets)
        down = down.reshape(batch_size, self.num_experts_per_token, d_model)
        return (down * route_probs.to(down.dtype).unsqueeze(-1)).sum(dim=1)

    def _router_metrics(
        self,
        router_probs: torch.Tensor,
        route_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expert_prob = router_probs.mean(dim=0)
        expert_tokens = F.one_hot(route_experts, num_classes=self.num_experts).float()
        expert_fraction = expert_tokens.mean(dim=(0, 1)).to(expert_prob.dtype)
        load_balancing_loss = self.num_experts * torch.sum(expert_prob * expert_fraction)
        dead_experts = (expert_fraction == 0).sum().to(expert_prob.dtype)
        return load_balancing_loss, dead_experts


class MlpMoeChessNet(nn.Module):
    """Single-token MLP model with MoE SwiGLU blocks."""

    flops_profile_dtype = torch.bfloat16

    def __init__(self, config: MlpMoeChessNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = MlpMoeChessNetConfig()
        if config.policy.kind != "dense":
            raise ValueError("MlpMoeChessNet only supports policy.kind='dense'.")
        self.config = config
        input_dim = config.input_planes * config.board_size * config.board_size
        hidden_dim = int(config.d_model * config.expert_mlp_ratio)

        self.input = nn.Linear(input_dim, config.d_model)
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
        self.norm = nn.RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            elementwise_affine=False,
        )
        self.policy_head = nn.Linear(config.d_model, config.policy_size)
        self.wdl_head = nn.Linear(config.d_model, 3)
        self.moves_left_head = nn.Linear(config.d_model, 1)

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
            policy_logits=self.policy_head(x),
            wdl_logits=self.wdl_head(x),
            moves_left=self.moves_left_head(x).squeeze(-1),
            aux_loss=torch.stack(aux_losses).mean(),
            router_dead_experts=router_dead_experts_by_layer.mean(),
            router_dead_experts_max=router_dead_experts_by_layer.max(),
        )

    def extra_training_flops_per_sample(self) -> int:
        return mlp_moe_grouped_mm_training_flops_per_sample(
            d_model=self.config.d_model,
            depth=self.config.depth,
            num_experts_per_token=self.config.num_experts_per_token,
            expert_mlp_ratio=self.config.expert_mlp_ratio,
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
    router_params = depth * d_model * num_experts
    policy_params = d_model * policy_size + policy_size
    wdl_params = d_model * 3 + 3
    moves_left_params = d_model + 1
    return (
        input_params
        + expert_params
        + router_params
        + policy_params
        + wdl_params
        + moves_left_params
    )


def mlp_moe_grouped_mm_training_flops_per_sample(
    *,
    d_model: int,
    depth: int,
    num_experts_per_token: int = 2,
    expert_mlp_ratio: float = 4.0,
) -> int:
    hidden_dim = int(d_model * expert_mlp_ratio)
    forward_flops_per_block = num_experts_per_token * 6 * d_model * hidden_dim
    return 3 * depth * forward_flops_per_block
