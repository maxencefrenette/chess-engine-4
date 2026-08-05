"""Model registry and config parsing."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from chess_engine_4.model.dense import DenseChessNet, DenseChessNetConfig, dense_parameter_count
from chess_engine_4.model.moe import (
    Moe64A2ChessNet,
    Moe64A2ChessNetConfig,
    moe64a2_parameter_count,
)

type ModelConfig = DenseChessNetConfig | Moe64A2ChessNetConfig


def model_config_from_dict(values: dict[str, Any]) -> ModelConfig:
    kind = values.get("kind", "dense")
    if kind == "dense":
        return _build_model_section(DenseChessNetConfig, values, section_name="[model]")
    if kind == "moe64a2":
        return _build_model_section(Moe64A2ChessNetConfig, values, section_name="[model]")
    raise ValueError(f"unknown model kind: {kind}")


def build_model(config: ModelConfig) -> DenseChessNet | Moe64A2ChessNet:
    if isinstance(config, DenseChessNetConfig):
        return DenseChessNet(config)
    if isinstance(config, Moe64A2ChessNetConfig):
        return Moe64A2ChessNet(config)
    raise TypeError(f"unsupported model config type: {type(config).__name__}")


def model_parameter_count(config: ModelConfig) -> int:
    if isinstance(config, DenseChessNetConfig):
        return dense_parameter_count(
            input_planes=config.input_planes,
            history_length=config.history_length,
            board_size=config.board_size,
            policy_size=config.policy_size,
            d_model=config.d_model,
            depth=config.depth,
            expansion_ratio=config.expansion_ratio,
            activation=config.activation,
        )
    if isinstance(config, Moe64A2ChessNetConfig):
        return moe64a2_parameter_count(
            input_planes=config.input_planes,
            history_length=config.history_length,
            board_size=config.board_size,
            policy_size=config.policy_size,
            d_model=config.d_model,
            depth=config.depth,
            expansion_ratio=config.expansion_ratio,
        )
    raise TypeError(f"unsupported model config type: {type(config).__name__}")


def _build_model_section[ConfigT](
    section_type: type[ConfigT],
    values: dict[str, Any],
    *,
    section_name: str,
) -> ConfigT:
    field_names = {field.name for field in fields(section_type)}
    unknown = sorted(set(values) - field_names)
    if unknown:
        unknown_names = ", ".join(unknown)
        raise ValueError(f"{section_name} has unknown key(s): {unknown_names}.")
    return section_type(**values)
