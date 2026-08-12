"""Training-only packed input adapter and CUDA graph setup."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from chess_engine_4.data.leela import BOARD_SIZE, HISTORY_PLANE_COUNT, INPUT_PLANE_COUNT
from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.transformer_engine import autocast_context, quantization_recipe, te

type PackedPlaneInput = tuple[torch.Tensor, torch.Tensor]
type GraphOutput = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class PlaneInputExpander(nn.Module):
    """Expand packed training planes into the LC0 plane tensor consumed by core models."""

    def __init__(self) -> None:
        super().__init__()
        self._packed_byte_bit_masks: torch.Tensor
        self.register_buffer(
            "_packed_byte_bit_masks",
            torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8),
            persistent=False,
        )

    def forward(self, packed_planes: torch.Tensor, plane_scalars: torch.Tensor) -> torch.Tensor:
        bits = torch.bitwise_and(
            packed_planes.unsqueeze(-1), self._packed_byte_bit_masks
        ).ne(0)
        history = bits.reshape(
            packed_planes.shape[0],
            HISTORY_PLANE_COUNT,
            BOARD_SIZE,
            BOARD_SIZE,
        ).to(dtype=plane_scalars.dtype)
        scalars = plane_scalars[:, :, None, None].expand(-1, -1, BOARD_SIZE, BOARD_SIZE)
        return torch.cat((history, scalars), dim=1)


class _GraphablePackedModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.input_expander = PlaneInputExpander()
        self.model = model

    def forward(self, packed_planes: torch.Tensor, plane_scalars: torch.Tensor) -> GraphOutput:
        output: ChessNetOutput = self.model(self.input_expander(packed_planes, plane_scalars))
        router_dead_experts = output.router_dead_experts
        router_tokens_per_expert = output.router_tokens_per_expert
        if router_dead_experts is None:
            router_dead_experts = torch.zeros(
                (), dtype=torch.int64, device=output.policy_logits.device
            )
        if router_tokens_per_expert is None:
            router_tokens_per_expert = torch.zeros(
                (0, 0), dtype=torch.int64, device=output.policy_logits.device
            )
        return (
            output.policy_logits,
            output.wdl_logits,
            output.moves_left,
            router_dead_experts,
            router_tokens_per_expert,
        )


class _GraphedTrainingModel(nn.Module):
    def __init__(self, graph: Callable[..., GraphOutput], *, has_router: bool) -> None:
        super().__init__()
        self.graph = graph
        self.has_router = has_router

    def forward(self, planes: PackedPlaneInput) -> ChessNetOutput:
        (
            policy_logits,
            wdl_logits,
            moves_left,
            router_dead_experts,
            router_tokens_per_expert,
        ) = self.graph(*planes)
        return ChessNetOutput(
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            moves_left=moves_left,
            router_dead_experts=router_dead_experts if self.has_router else None,
            router_tokens_per_expert=(router_tokens_per_expert if self.has_router else None),
        )


def build_training_model(model: nn.Module, *, batch_size: int, precision: str) -> nn.Module:
    """Build the packed-input training model, using a CUDA graph when supported."""

    sample_planes = _sample_packed_planes(batch_size)
    recipe = quantization_recipe(precision)
    with autocast_context(precision):
        graphable = _GraphablePackedModel(model).cuda().train()
        has_router = getattr(model.config, "kind", None) == "moe64a2"
        if not getattr(model, "cuda_graph_compatible", True):
            return _GraphedTrainingModel(graphable, has_router=has_router)
        graph = te().make_graphed_callables(
            graphable,
            sample_planes,
            allow_unused_input=True,
            enabled=recipe is not None,
            recipe=recipe,
        )
        return _GraphedTrainingModel(
            graph,
            has_router=has_router,
        )


def _sample_packed_planes(batch_size: int) -> PackedPlaneInput:
    return (
        torch.zeros(
            batch_size,
            HISTORY_PLANE_COUNT,
            BOARD_SIZE,
            device="cuda",
            dtype=torch.uint8,
        ),
        torch.zeros(
            batch_size,
            INPUT_PLANE_COUNT - HISTORY_PLANE_COUNT,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    )
