"""Model registry and config parsing."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from torch import nn

from chess_engine_4.model.heads import AttentionPolicyHeadConfig, DensePolicyHeadConfig
from chess_engine_4.model.mlp import MlpChessNet, MlpChessNetConfig
from chess_engine_4.model.mlp_moe import MlpMoeChessNet, MlpMoeChessNetConfig
from chess_engine_4.model.transformer import Transformer64ChessNet, Transformer64ChessNetConfig

type ModelConfig = MlpChessNetConfig | MlpMoeChessNetConfig | Transformer64ChessNetConfig


def model_config_from_dict(values: dict[str, Any]) -> ModelConfig:
    kind = values.get("kind", "mlp")
    if kind == "mlp":
        values = {**values, "policy": _policy_config_from_dict(values.get("policy"), "dense")}
        return _build_model_section(MlpChessNetConfig, values, section_name="[model]")
    if kind == "mlp_moe":
        values = {**values, "policy": _policy_config_from_dict(values.get("policy"), "dense")}
        return _build_model_section(MlpMoeChessNetConfig, values, section_name="[model]")
    if kind == "transformer64":
        values = {**values, "policy": _policy_config_from_dict(values.get("policy"), "attention")}
        return _build_model_section(Transformer64ChessNetConfig, values, section_name="[model]")
    raise ValueError(f"unknown model kind: {kind}")


def build_model(config: ModelConfig) -> nn.Module:
    if isinstance(config, MlpChessNetConfig):
        return MlpChessNet(config)
    if isinstance(config, MlpMoeChessNetConfig):
        return MlpMoeChessNet(config)
    if isinstance(config, Transformer64ChessNetConfig):
        return Transformer64ChessNet(config)
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


def _policy_config_from_dict(values: object, default_kind: str) -> object:
    if values is None:
        values = {"kind": default_kind}
    if not isinstance(values, dict):
        raise ValueError("[model.policy] must be a table.")
    kind = values.get("kind", default_kind)
    if kind == "dense":
        return _build_model_section(
            DensePolicyHeadConfig,
            values,
            section_name="[model.policy]",
        )
    if kind == "attention":
        return _build_model_section(
            AttentionPolicyHeadConfig,
            values,
            section_name="[model.policy]",
        )
    raise ValueError(f"unknown policy head kind: {kind}")
