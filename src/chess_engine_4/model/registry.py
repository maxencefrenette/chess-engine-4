"""Model registry and config parsing."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from chess_engine_4.model.dense import DenseChessNet, DenseChessNetConfig

type ModelConfig = DenseChessNetConfig


def model_config_from_dict(values: dict[str, Any]) -> ModelConfig:
    kind = values.get("kind", "dense")
    if kind == "dense":
        return _build_model_section(DenseChessNetConfig, values, section_name="[model]")
    raise ValueError(f"unknown model kind: {kind}")


def build_model(config: ModelConfig) -> DenseChessNet:
    if isinstance(config, DenseChessNetConfig):
        return DenseChessNet(config)
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
