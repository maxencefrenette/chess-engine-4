from dataclasses import asdict

import torch

from chess_engine_4.model import DenseChessNetConfig
from chess_engine_4.training.checkpoint2leela import _model_from_checkpoint


def test_loads_dense_checkpoint() -> None:
    model_config = DenseChessNetConfig(d_model=8, depth=1, expansion_ratio=2.0)
    loaded = _model_from_checkpoint(
        {
            "config": {"model": asdict(model_config)},
            "model_state_dict": _dense_te_state_dict(model_config),
        }
    )

    output = loaded(torch.zeros(2, 112, 8, 8))
    assert output.policy_logits.shape == (2, 1858)


def _dense_te_state_dict(config: DenseChessNetConfig) -> dict[str, torch.Tensor]:
    input_dim = config.input_planes * config.board_size * config.board_size
    hidden_dim = int(config.d_model * config.expansion_ratio)
    state = {
        "input.weight": torch.randn(config.d_model, input_dim),
        "input.bias": torch.randn(config.d_model),
        "norm.weight": torch.ones(config.d_model),
        "policy_head.weight": torch.randn(config.policy_size, config.d_model),
        "policy_head.bias": torch.randn(config.policy_size),
        "wdl_head.weight": torch.randn(3, config.d_model),
        "wdl_head.bias": torch.randn(3),
        "moves_left_head.weight": torch.randn(1, config.d_model),
        "moves_left_head.bias": torch.randn(1),
    }
    for layer in range(config.depth):
        prefix = f"blocks.{layer}.layer"
        state[f"{prefix}.layer_norm_weight"] = torch.ones(config.d_model)
        state[f"{prefix}.fc1_weight"] = torch.randn(2 * hidden_dim, config.d_model)
        state[f"{prefix}.fc2_weight"] = torch.randn(config.d_model, hidden_dim)
    return state
