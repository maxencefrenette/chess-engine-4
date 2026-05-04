from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine_4.model import MlpChessNet
from chess_engine_4.training.config import load_training_config, with_overrides


def test_load_training_config_reads_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "train.toml"
    config_path.write_text(
        """
[run]
name = "audit"
steps = 25

[data]
batch_size = 8
max_records = 128

[model]
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
    assert config.run.steps == 25
    assert config.data.batch_size == 8
    assert config.data.max_records == 128
    assert config.model.d_model == 64
    assert config.model.depth == 2
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
    config = load_training_config("configs/d192x3.toml")

    overridden = with_overrides(config, steps=3, batch_size=4, device="cpu")

    assert overridden.run.steps == 3
    assert overridden.data.batch_size == 4
    assert overridden.run.device == "cpu"
    assert overridden.model == config.model
    assert overridden.optimizer == config.optimizer
    assert overridden.loss == config.loss


def test_d192x3_config_builds_about_one_million_trunk_parameters() -> None:
    config = load_training_config("configs/d192x3.toml")
    model = MlpChessNet(config.model)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trunk_parameter_count = sum(
        parameter.numel() for block in model.blocks for parameter in block.parameters()
    )

    assert 1_000_000 <= trunk_parameter_count <= 1_500_000
    assert 2_750_000 <= parameter_count <= 3_250_000
