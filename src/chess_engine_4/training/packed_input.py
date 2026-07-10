"""Training-only packed input adapter and CUDA graph setup."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch
from torch import nn

from chess_engine_4.data.leela import BOARD_SIZE, HISTORY_PLANE_COUNT, INPUT_PLANE_COUNT
from chess_engine_4.model.mlp_moe import MlpMoeChessNet
from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.transformer_engine import te

type PackedPlaneInput = tuple[torch.Tensor, torch.Tensor]
type GraphOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class PlaneInputExpander(nn.Module):
    """Expand packed training planes into the LC0 plane tensor consumed by core models."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "_packed_byte_bit_masks",
            torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8),
            persistent=False,
        )

    def forward(self, packed_planes: torch.Tensor, plane_scalars: torch.Tensor) -> torch.Tensor:
        bits = packed_planes.unsqueeze(-1).bitwise_and(self._packed_byte_bit_masks).ne(0)
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
        return output.policy_logits, output.wdl_logits, output.moves_left


class _GraphedDenseTrainingModel(nn.Module):
    def __init__(self, graph: Callable[..., GraphOutput]) -> None:
        super().__init__()
        self.graph = graph

    def forward(self, planes: PackedPlaneInput) -> ChessNetOutput:
        policy_logits, wdl_logits, moves_left = self.graph(*planes)
        return ChessNetOutput(
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            moves_left=moves_left,
        )


class PackedInputTrainingModel(nn.Module):
    """Wrap a core model with compiled packed-plane expansion."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.input_expander = torch.compile(PlaneInputExpander(), mode="reduce-overhead")
        self.model = model

    def forward(self, planes: PackedPlaneInput) -> ChessNetOutput:
        return self.model(self.input_expander(*planes))


def build_training_model(model: nn.Module, *, batch_size: int) -> nn.Module:
    """Build the fastest supported packed-input adapter for a core model."""

    if isinstance(model, MlpMoeChessNet):
        return PackedInputTrainingModel(model).cuda()

    sample_planes = _sample_packed_planes(batch_size)
    with (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings(
            "ignore",
            message="The AccumulateGrad node's stream does not match.*",
            category=UserWarning,
        )
        graphable = _GraphablePackedModel(model).cuda().train()
        graph = te().make_graphed_callables(
            graphable,
            sample_planes,
            allow_unused_input=True,
        )
        return _GraphedDenseTrainingModel(graph)


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
