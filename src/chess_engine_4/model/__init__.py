"""Model definitions."""

from chess_engine_4.model.mlp import MlpChessNet, MlpChessNetConfig
from chess_engine_4.model.output import ChessNetOutput
from chess_engine_4.model.registry import ModelConfig, build_model, model_config_from_dict
from chess_engine_4.model.transformer import Transformer64ChessNet, Transformer64ChessNetConfig

__all__ = [
    "ChessNetOutput",
    "MlpChessNet",
    "MlpChessNetConfig",
    "ModelConfig",
    "Transformer64ChessNet",
    "Transformer64ChessNetConfig",
    "build_model",
    "model_config_from_dict",
]
