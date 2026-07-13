"""Dense chess network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_engine_4.data.leela import INPUT_PLANE_COUNT, POLICY_SIZE, RULE50_PLANE_INDEX
from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.transformer_engine import te

SUPPORTED_ACTIVATIONS = frozenset({"geglu", "gelu", "silu", "srelu", "swiglu"})
GATED_ACTIVATIONS = frozenset({"geglu", "swiglu"})


def mxfp8_aligned_size(size: int) -> int:
    return (size + 31) // 32 * 32


def normalize_lc0_planes(planes: torch.Tensor) -> torch.Tensor:
    """Normalize lc0's raw rule-50 ply plane to the model's learned input scale."""

    planes[:, RULE50_PLANE_INDEX].div_(99.0)
    return planes


@dataclass(frozen=True, slots=True)
class DenseChessNetConfig:
    kind: str = "dense"
    input_planes: int = INPUT_PLANE_COUNT
    board_size: int = 8
    policy_size: int = POLICY_SIZE
    d_model: int = 1024
    depth: int = 8
    expansion_ratio: float = 4.0
    # Checkpoints created before activation was configurable implicitly used SwiGLU.
    activation: str = "swiglu"
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
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
        return x + self.layer(x)


class DenseChessNet(nn.Module):
    """Single-token dense model over flattened LCZero input planes."""

    def __init__(self, config: DenseChessNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = DenseChessNetConfig()
        self.config = config
        input_dim = config.input_planes * config.board_size * config.board_size
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
        x = normalize_lc0_planes(planes).flatten(start_dim=1)
        x = self.input(x)
        x = self.blocks(x)
        x = self.norm(x)
        return ChessNetOutput(
            policy_logits=self.policy_head(x)[:, : self.config.policy_size],
            wdl_logits=self.wdl_head(x)[:, :3],
            moves_left=self.moves_left_head(x)[:, 0],
        )


def dense_parameter_count(
    *,
    input_planes: int = INPUT_PLANE_COUNT,
    board_size: int = 8,
    policy_size: int = POLICY_SIZE,
    d_model: int,
    depth: int,
    expansion_ratio: float = 4.0,
    activation: str = "geglu",
) -> int:
    input_dim = input_planes * board_size * board_size
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
