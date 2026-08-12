"""Dense chess network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from chess_engine_4.data.leela import (
    BOARD_SIZE,
    HISTORY_PLANE_COUNT,
    INPUT_PLANE_COUNT,
    POLICY_SIZE,
    RULE50_PLANE_INDEX,
)
from chess_engine_4.model.config import InputPipeline, KernelBackend, Precision
from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.transformer_engine import te

SUPPORTED_ACTIVATIONS = frozenset({"geglu", "gelu", "silu", "srelu", "swiglu"})
GATED_ACTIVATIONS = frozenset({"geglu", "swiglu"})
HISTORY_LENGTH = 8
PLANES_PER_HISTORY_POSITION = HISTORY_PLANE_COUNT // HISTORY_LENGTH


def mxfp8_aligned_size(size: int) -> int:
    return (size + 31) // 32 * 32


def normalize_lc0_planes(
    planes: torch.Tensor,
    *,
    rule50_plane_index: int = RULE50_PLANE_INDEX,
) -> torch.Tensor:
    """Normalize lc0's raw rule-50 ply plane to the model's learned input scale."""

    planes[:, rule50_plane_index].div_(99.0)
    return planes


def select_lc0_history(planes: torch.Tensor, history_length: int) -> torch.Tensor:
    """Retain recent history and all auxiliary planes from lc0's 112-plane input."""

    history_planes = history_length * PLANES_PER_HISTORY_POSITION
    if history_planes == HISTORY_PLANE_COUNT:
        return planes
    return torch.cat((planes[:, :history_planes], planes[:, HISTORY_PLANE_COUNT:]), dim=1)


def model_input_plane_count(history_length: int) -> int:
    auxiliary_planes = INPUT_PLANE_COUNT - HISTORY_PLANE_COUNT
    return history_length * PLANES_PER_HISTORY_POSITION + auxiliary_planes


@dataclass(frozen=True, slots=True)
class DenseChessNetConfig:
    kind: str = "dense"
    history_length: int = HISTORY_LENGTH
    d_model: int = 1024
    depth: int = 8
    expansion_ratio: float = 4.0
    # Checkpoints created before activation was configurable implicitly used SwiGLU.
    activation: str = "swiglu"
    rms_norm_eps: float = 1e-6
    precision: Precision = "mxfp8"
    kernel_backend: KernelBackend = "te"
    input_pipeline: InputPipeline = "pinned"

    def __post_init__(self) -> None:
        if self.d_model < 64:
            raise ValueError("d_model must be at least 64")
        if not 1 <= self.history_length <= HISTORY_LENGTH:
            raise ValueError(f"history_length must be in [1, {HISTORY_LENGTH}]")
        if self.expansion_ratio <= 0:
            raise ValueError("expansion_ratio must be positive")
        if self.activation not in SUPPORTED_ACTIVATIONS:
            choices = ", ".join(sorted(SUPPORTED_ACTIVATIONS))
            raise ValueError(f"unsupported activation {self.activation!r}; choose from {choices}")


class DenseBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        rms_norm_eps: float,
        activation: str,
        precision: Precision,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.rms_norm_eps = rms_norm_eps
        self.activation = activation
        self.precision = precision
        self._custom_kernels_enabled = False
        transformer_engine = te()
        self.layer = transformer_engine.LayerNormMLP(
            d_model,
            hidden_dim,
            eps=rms_norm_eps,
            bias=False,
            normalization="RMSNorm",
            activation=activation,
            params_dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._custom_kernels_enabled:
            from chess_engine_4.kernels import dense_block_trainable

            return dense_block_trainable(
                x,
                self.layer.layer_norm_weight,
                self.layer.fc1_weight,
                self.layer.fc2_weight,
                precision=self.precision,
                eps=self.rms_norm_eps,
            )
        output = self.layer(x)
        return x + cast(torch.Tensor, output)

    def enable_custom_kernels(self) -> None:
        from chess_engine_4.kernels.capabilities import require_dense_model_shape

        require_dense_model_shape(
            d_model=self.d_model,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
        )
        self._custom_kernels_enabled = True


class DenseChessNet(nn.Module):
    """Single-token dense model over flattened LCZero input planes."""

    def __init__(self, config: DenseChessNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = DenseChessNetConfig()
        self.config = config
        input_dim = model_input_plane_count(config.history_length) * BOARD_SIZE**2
        hidden_dim = int(config.d_model * config.expansion_ratio)
        transformer_engine = te()

        self.input = transformer_engine.Linear(
            input_dim,
            config.d_model,
            params_dtype=torch.bfloat16,
        )
        self.blocks = nn.Sequential(
            *[
                DenseBlock(
                    d_model=config.d_model,
                    hidden_dim=hidden_dim,
                    rms_norm_eps=config.rms_norm_eps,
                    activation=config.activation,
                    precision=config.precision,
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
            mxfp8_aligned_size(POLICY_SIZE),
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
        x = self.blocks(x)
        x = self.norm(x)
        return ChessNetOutput(
            policy_logits=self.policy_head(x)[:, :POLICY_SIZE],
            wdl_logits=self.wdl_head(x)[:, :3],
            moves_left=self.moves_left_head(x)[:, 0],
        )

    def enable_custom_kernels(self) -> None:
        for block in self.blocks:
            cast(DenseBlock, block).enable_custom_kernels()


def dense_parameter_count(
    *,
    history_length: int = HISTORY_LENGTH,
    d_model: int,
    depth: int,
    expansion_ratio: float = 4.0,
    activation: str = "geglu",
) -> int:
    input_dim = model_input_plane_count(history_length) * BOARD_SIZE**2
    hidden_dim = int(d_model * expansion_ratio)
    projection_count = 3 if activation in GATED_ACTIVATIONS else 2
    block_params = depth * (projection_count * d_model * hidden_dim)
    norm_params = (depth + 1) * d_model
    input_params = input_dim * d_model + d_model
    aligned_policy_size = mxfp8_aligned_size(POLICY_SIZE)
    policy_params = d_model * aligned_policy_size + aligned_policy_size
    wdl_params = d_model * 32 + 32
    moves_left_params = d_model * 32 + 32
    return (
        input_params + block_params + norm_params + policy_params + wdl_params + moves_left_params
    )
