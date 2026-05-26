from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine_4.model import MlpChessNet, MlpMoeChessNet, Transformer64ChessNet
from chess_engine_4.training.config import load_training_config, with_overrides


def test_load_training_config_reads_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "train.toml"
    config_path.write_text(
        """
[run]
name = "audit"
compute_budget = 1e12
step_penalty_k = 1.1

[infra]
gpu_type = "l4"
dataloader_threads = 4
dataloader_prefetch_per_thread = 3

[data]
batch_size = 8

[model]
kind = "mlp"
d_model = 64
depth = 2

[optimizer]
lr = 0.001

[loss]
moves_left = 0.5
""".strip()
    )

    config = load_training_config(config_path)

    assert config.run.name == "audit"
    assert config.run.compute_budget == 1e12
    assert config.run.step_penalty_k == 1.1
    assert config.infra.gpu_type == "l4"
    assert config.infra.dataloader_threads == 4
    assert config.infra.dataloader_prefetch_per_thread == 3
    assert config.data.batch_size == 8
    assert config.model.d_model == 64
    assert config.model.depth == 2
    assert config.model.kind == "mlp"
    assert config.optimizer.lr == 0.001
    assert config.loss.moves_left == 0.5


def test_load_training_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "train.toml"
    config_path.write_text(
        """
[model]
width = 64
""".strip()
    )

    with pytest.raises(ValueError, match="unknown key"):
        load_training_config(config_path)


def test_with_overrides_keeps_config_as_source_of_truth() -> None:
    config = load_training_config("configs/mlp/1e18.toml")

    overridden = with_overrides(
        config,
        compute_budget=1e11,
        step_penalty_k=1.1,
        batch_size=4,
        d_model=64,
        depth=2,
        lr=0.001,
    )

    assert overridden.run.compute_budget == 1e11
    assert overridden.run.step_penalty_k == 1.1
    assert overridden.data.batch_size == 4
    assert overridden.model.d_model == 64
    assert overridden.model.depth == 2
    assert overridden.optimizer.lr == 0.001
    assert overridden.optimizer.weight_decay == config.optimizer.weight_decay
    assert overridden.loss == config.loss


def test_load_training_config_supports_transformer64() -> None:
    config = load_training_config("configs/transformer64/1e14.toml")
    model = Transformer64ChessNet(config.model)

    assert config.model.kind == "transformer64"
    assert config.model.num_heads == 4
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_load_training_config_supports_mlp_moe() -> None:
    config = load_training_config("configs/mlp_moe16a2/1e18.toml")
    model = MlpMoeChessNet(config.model)

    assert config.model.kind == "mlp_moe"
    assert config.model.num_experts == 16
    assert config.model.num_experts_per_token == 2
    assert config.loss.router_aux == 0.03
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_with_overrides_supports_transformer_heads() -> None:
    config = load_training_config("configs/transformer64/1e14.toml")

    overridden = with_overrides(config, d_model=64, depth=3, num_heads=8)

    assert overridden.model.d_model == 64
    assert overridden.model.depth == 3
    assert overridden.model.num_heads == 8


def test_with_overrides_supports_router_aux() -> None:
    config = load_training_config("configs/mlp_moe16a2/1e18.toml")

    overridden = with_overrides(config, router_aux=0.001)

    assert overridden.loss.router_aux == 0.001


def test_train_options_supports_max_steps() -> None:
    from chess_engine_4.training.cli import TrainOptions

    assert TrainOptions(max_steps=200).max_steps == 200


def test_1e18_config_builds_expected_model_size() -> None:
    config = load_training_config("configs/mlp/1e18.toml")
    model = MlpChessNet(config.model)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trunk_parameter_count = sum(
        parameter.numel() for block in model.blocks for parameter in block.parameters()
    )

    assert trunk_parameter_count == 147_456
    assert parameter_count == 727_302
