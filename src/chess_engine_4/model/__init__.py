"""Model definitions."""

from chess_engine_4.model.mlp import MlpChessNet, MlpChessNetConfig, mlp_parameter_count
from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.registry import ModelConfig, build_model, model_config_from_dict
from chess_engine_4.model.transformer import (
    Transformer64ChessNet,
    Transformer64ChessNetConfig,
    transformer64_parameter_count,
)

__all__ = [
    "ChessNetOutput",
    "MlpChessNet",
    "MlpChessNetConfig",
    "ModelConfig",
    "Transformer64ChessNet",
    "Transformer64ChessNetConfig",
    "build_model",
    "mlp_parameter_count",
    "model_config_from_dict",
    "transformer64_parameter_count",
]
