"""Dense chess network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_engine_4.data.leela import (
    HISTORY_PLANE_COUNT,
    INPUT_PLANE_COUNT,
    POLICY_SIZE,
    RULE50_PLANE_INDEX,
)
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
    input_planes: int = INPUT_PLANE_COUNT
    board_size: int = 8
    policy_size: int = POLICY_SIZE
    history_length: int = HISTORY_LENGTH
    d_model: int = 1024
    depth: int = 8
    expansion_ratio: float = 4.0
    # Checkpoints created before activation was configurable implicitly used SwiGLU.
    activation: str = "swiglu"
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.rms_norm_eps = rms_norm_eps
        self.activation = activation
        self._use_custom_kernel = False
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
        if self._use_custom_kernel:
            from chess_engine_4.kernels import dense_mxfp8_trainable

            return dense_mxfp8_trainable(
                x,
                self.layer.layer_norm_weight,
                self.layer.fc1_weight,
                self.layer.fc2_weight,
                eps=self.rms_norm_eps,
            )
        return x + self.layer(x)

    def enable_experimental_dense_kernel(self) -> None:
        from chess_engine_4.kernels.dense import SUPPORTED_DENSE_WIDTHS

        if self.d_model not in SUPPORTED_DENSE_WIDTHS or self.hidden_dim != 4 * self.d_model:
            raise ValueError(
                "the experimental dense kernel requires d_model in "
                f"{sorted(SUPPORTED_DENSE_WIDTHS)} and expansion_ratio=4"
            )
        if self.activation != "swiglu":
            raise ValueError("the experimental dense kernel requires activation='swiglu'")
        self._use_custom_kernel = True


class DenseChessNet(nn.Module):
    """Single-token dense model over flattened LCZero input planes."""

    def __init__(self, config: DenseChessNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = DenseChessNetConfig()
        self.config = config
        input_dim = model_input_plane_count(config.history_length) * config.board_size**2
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
        x = select_lc0_history(planes, self.config.history_length)
        rule50_plane_index = self.config.history_length * PLANES_PER_HISTORY_POSITION + 5
        x = normalize_lc0_planes(x, rule50_plane_index=rule50_plane_index).flatten(start_dim=1)
        x = self.input(x)
        x = self.blocks(x)
        x = self.norm(x)
        return ChessNetOutput(
            policy_logits=self.policy_head(x)[:, : self.config.policy_size],
            wdl_logits=self.wdl_head(x)[:, :3],
            moves_left=self.moves_left_head(x)[:, 0],
        )

    def enable_experimental_dense_kernel(self) -> None:
        for block in self.blocks:
            block.enable_experimental_dense_kernel()


def dense_parameter_count(
    *,
    input_planes: int = INPUT_PLANE_COUNT,
    history_length: int = HISTORY_LENGTH,
    board_size: int = 8,
    policy_size: int = POLICY_SIZE,
    d_model: int,
    depth: int,
    expansion_ratio: float = 4.0,
    activation: str = "geglu",
) -> int:
    auxiliary_planes = input_planes - HISTORY_PLANE_COUNT
    selected_planes = history_length * PLANES_PER_HISTORY_POSITION + auxiliary_planes
    input_dim = selected_planes * board_size * board_size
    hidden_dim = int(d_model * expansion_ratio)
    projection_count = 3 if activation in GATED_ACTIVATIONS else 2
    block_params = depth * (projection_count * d_model * hidden_dim)
    norm_params = (depth + 1) * d_model
    input_params = input_dim * d_model + d_model
    aligned_policy_size = mxfp8_aligned_size(policy_size)
    policy_params = d_model * aligned_policy_size + aligned_policy_size
    wdl_params = d_model * 32 + 32
    moves_left_params = d_model * 32 + 32
    return (
        input_params + block_params + norm_params + policy_params + wdl_params + moves_left_params
    )
