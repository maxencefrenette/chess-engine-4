"""Model registry and config parsing."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from torch import nn

from chess_engine_4.model.mlp import MlpChessNet, MlpChessNetConfig
from chess_engine_4.model.mlp_moe import MlpMoeChessNet, MlpMoeChessNetConfig

type ModelConfig = MlpChessNetConfig | MlpMoeChessNetConfig


def model_config_from_dict(values: dict[str, Any]) -> ModelConfig:
    kind = values.get("kind", "mlp")
    if kind == "mlp":
        return _build_model_section(MlpChessNetConfig, values, section_name="[model]")
    if kind == "mlp_moe":
        return _build_model_section(MlpMoeChessNetConfig, values, section_name="[model]")
    raise ValueError(f"unknown model kind: {kind}")


def build_model(config: ModelConfig) -> nn.Module:
    if isinstance(config, MlpChessNetConfig):
        return MlpChessNet(config)
    if isinstance(config, MlpMoeChessNetConfig):
        return MlpMoeChessNet(config)
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
