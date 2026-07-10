from dataclasses import asdict

import torch

from chess_engine_4.model import MlpChessNetConfig
from chess_engine_4.training.checkpoint2leela import _model_from_checkpoint


def test_loads_te_dense_checkpoint_with_legacy_policy_config() -> None:
    model_config = MlpChessNetConfig(d_model=8, depth=1, mlp_ratio=2.0)
    config = asdict(model_config)
    config["policy"] = {"kind": "dense"}

    loaded = _model_from_checkpoint(
        {
            "config": {"model": config},
            "model_state_dict": _dense_te_state_dict(model_config),
        }
    )

    output = loaded(torch.zeros(2, 112, 8, 8))
    assert output.policy_logits.shape == (2, 1858)


def test_loads_pre_te_dense_checkpoint() -> None:
    model_config = MlpChessNetConfig(d_model=8, depth=1, mlp_ratio=2.0)
    state = _dense_te_state_dict(model_config)
    state["blocks.0.gate_proj.weight"] = state.pop("blocks.0.mlp.fc1_weight")[:16]
    state["blocks.0.up_proj.weight"] = torch.randn(16, 8)
    state["blocks.0.down_proj.weight"] = state.pop("blocks.0.mlp.fc2_weight")
    state.pop("blocks.0.mlp.layer_norm_weight")
    state.pop("norm.weight")

    loaded = _model_from_checkpoint(
        {
            "config": {"model": asdict(model_config)},
            "model_state_dict": state,
        }
    )

    assert loaded(torch.zeros(2, 112, 8, 8)).policy_logits.shape == (2, 1858)


def _dense_te_state_dict(config: MlpChessNetConfig) -> dict[str, torch.Tensor]:
    input_dim = config.input_planes * config.board_size * config.board_size
    hidden_dim = int(config.d_model * config.mlp_ratio)
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
        prefix = f"blocks.{layer}.mlp"
        state[f"{prefix}.layer_norm_weight"] = torch.ones(config.d_model)
        state[f"{prefix}.fc1_weight"] = torch.randn(2 * hidden_dim, config.d_model)
        state[f"{prefix}.fc2_weight"] = torch.randn(config.d_model, hidden_dim)
    return state
