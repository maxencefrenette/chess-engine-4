"""Training-only packed input adapter."""

from __future__ import annotations

import torch
from torch import nn

from chess_engine_4.data.leela import (
    BOARD_SIZE,
    HISTORY_PLANE_COUNT,
)
from chess_engine_4.model.output import ChessNetOutput

type PackedPlaneInput = tuple[torch.Tensor, torch.Tensor]


class PlaneInputExpander(nn.Module):
    """Expand packed training planes into the LC0 plane tensor consumed by core models."""

    def __init__(self) -> None:
        super().__init__()
        # Masks for unpacking each bit in one rank byte. This is a buffer so it follows
        # the training wrapper to CUDA/MPS/CPU without becoming a checkpoint parameter.
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
        scalars = plane_scalars[:, :, None, None].expand(
            -1,
            -1,
            BOARD_SIZE,
            BOARD_SIZE,
        )
        return torch.cat((history, scalars), dim=1)


class PackedInputTrainingModel(nn.Module):
    """Wrap a core LC0-plane model for packed-plane training batches."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.input_expander = torch.compile(PlaneInputExpander(), mode="reduce-overhead")
        self.model = model

    def forward(self, planes: PackedPlaneInput) -> ChessNetOutput:
        packed_planes, plane_scalars = planes
        return self.model(self.input_expander(packed_planes, plane_scalars))
