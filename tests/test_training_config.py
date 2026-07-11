from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine_4.model import mlp_moe_parameter_count, mlp_parameter_count
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
max_grad_norm = 2.0
lr_warmup_steps = 50
lr_cooldown_frac = 0.1

[loss]
moves_left = 0.5
""".strip()
    )

    config = load_training_config(config_path)

    assert config.run.name == "audit"
    assert config.run.compute_budget == 1e12
    assert config.run.step_penalty_k == 1.1
    assert config.infra.dataloader_threads == 4
    assert config.infra.dataloader_prefetch_per_thread == 3
    assert config.data.batch_size == 8
    assert config.model.d_model == 64
    assert config.model.depth == 2
    assert config.model.kind == "mlp"
    assert config.optimizer.lr == 0.001
    assert config.optimizer.max_grad_norm == 2.0
    assert config.optimizer.lr_warmup_steps == 50
    assert config.optimizer.lr_cooldown_frac == 0.1
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
        batch_size=4,
        d_model=64,
        depth=2,
        lr=0.001,
        max_grad_norm=3.0,
        lr_warmup_steps=25,
        lr_cooldown_frac=0.2,
    )

    assert overridden.run.compute_budget == 1e11
    assert overridden.run.step_penalty_k == config.run.step_penalty_k
    assert overridden.data.batch_size == 4
    assert overridden.model.d_model == 64
    assert overridden.model.depth == 2
    assert overridden.optimizer.lr == 0.001
    assert overridden.optimizer.max_grad_norm == 3.0
    assert overridden.optimizer.lr_warmup_steps == 25
    assert overridden.optimizer.lr_cooldown_frac == 0.2
    assert overridden.optimizer.weight_decay == config.optimizer.weight_decay
    assert overridden.loss == config.loss


def test_load_training_config_supports_mlp_moe() -> None:
    config = load_training_config("configs/mlp_moe16a2/1e18.toml")

    assert config.model.kind == "mlp_moe"
    assert config.model.num_experts == 16
    assert config.model.num_experts_per_token == 2
    assert config.loss.router_aux == 0.03
    assert mlp_moe_parameter_count(
        d_model=config.model.d_model,
        depth=config.model.depth,
        num_experts=config.model.num_experts,
        expert_mlp_ratio=config.model.expert_mlp_ratio,
    ) > 0


def test_with_overrides_supports_router_aux() -> None:
    config = load_training_config("configs/mlp_moe16a2/1e18.toml")

    overridden = with_overrides(config, router_aux=0.001)

    assert overridden.loss.router_aux == 0.001


def test_train_options_supports_max_steps() -> None:
    from chess_engine_4.training.cli import TrainOptions

    options = TrainOptions(max_steps=200)

    assert options.max_steps == 200


def test_training_profile_config_validates_steps() -> None:
    from chess_engine_4.training.profiling import TrainingProfileConfig

    profile = TrainingProfileConfig(warmup_steps=50, profile_steps=200)

    assert profile.total_steps == 250
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainingProfileConfig(warmup_steps=-1)
    with pytest.raises(ValueError, match="profile_steps"):
        TrainingProfileConfig(profile_steps=0)


def test_lr_cooldown_schedule() -> None:
    from chess_engine_4.training.cli import _scheduled_lr

    assert _scheduled_lr(
        base_lr=1.0,
        warmup_steps=0,
        cooldown_frac=0.1,
        step=90,
        total_steps=100,
    ) == 1.0
    assert _scheduled_lr(
        base_lr=1.0,
        warmup_steps=0,
        cooldown_frac=0.1,
        step=95,
        total_steps=100,
    ) == 0.5
    assert _scheduled_lr(
        base_lr=1.0,
        warmup_steps=0,
        cooldown_frac=0.1,
        step=100,
        total_steps=100,
    ) == pytest.approx(0.0)


def test_lr_warmup_schedule() -> None:
    from chess_engine_4.training.cli import _scheduled_lr

    assert _scheduled_lr(
        base_lr=1.0,
        warmup_steps=50,
        cooldown_frac=0.1,
        step=1,
        total_steps=100,
    ) == 0.02
    assert _scheduled_lr(
        base_lr=1.0,
        warmup_steps=50,
        cooldown_frac=0.1,
        step=50,
        total_steps=100,
    ) == 1.0
    assert _scheduled_lr(
        base_lr=1.0,
        warmup_steps=50,
        cooldown_frac=0.1,
        step=51,
        total_steps=100,
    ) == 1.0


def test_1e18_config_builds_expected_model_size() -> None:
    config = load_training_config("configs/mlp/1e18.toml")

    assert mlp_parameter_count(
        d_model=config.model.d_model,
        depth=config.model.depth,
        mlp_ratio=config.model.mlp_ratio,
    ) == 733_408


def test_precision_recipe_rejects_unknown_value(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-precision.toml"
    config_path.write_text('[precision]\nrecipe = "fp8"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown quantization recipe: fp8"):
        load_training_config(config_path)
