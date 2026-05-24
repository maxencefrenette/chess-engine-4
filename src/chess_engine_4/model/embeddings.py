"""Reusable input embeddings."""

from __future__ import annotations

import torch
from torch import nn


class SquareInputEmbedding(nn.Module):
    """Project LC0 planes into one token per board square."""

    def __init__(
        self,
        *,
        input_planes: int,
        board_size: int,
        d_model: int,
        learned_square_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.input = nn.Linear(input_planes, d_model)
        self.square_embedding = (
            nn.Parameter(torch.zeros(board_size * board_size, d_model))
            if learned_square_embeddings
            else None
        )

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        tokens = planes.flatten(start_dim=2).transpose(1, 2)
        tokens = self.input(tokens)
        if self.square_embedding is not None:
            tokens = tokens + self.square_embedding
        return tokens
