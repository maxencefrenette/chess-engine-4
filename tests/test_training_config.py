from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine_4.model import dense_parameter_count
from chess_engine_4.training.config import load_training_config, with_overrides


def test_load_training_config_requires_factory(tmp_path: Path) -> None:
    config_path = tmp_path / "train.py"
    config_path.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected callable config"):
        load_training_config(config_path, d_model=64)


def test_with_overrides_keeps_config_as_source_of_truth() -> None:
    config = load_training_config("configs/dense.py", d_model=64)

    overridden = with_overrides(
        config,
        steps=321,
        batch_size=4,
        depth=2,
        expansion_ratio=2.0,
        activation="silu",
        lr=0.001,
        max_grad_norm=3.0,
        lr_warmup_steps=25,
        lr_cooldown_frac=0.2,
    )

    assert overridden.run.steps == 321
    assert overridden.run.batch_size == 4
    assert overridden.model.d_model == 64
    assert overridden.model.depth == 2
    assert overridden.model.expansion_ratio == 2.0
    assert overridden.model.activation == "silu"
    assert overridden.optimizer.lr == 0.001
    assert overridden.optimizer.max_grad_norm == 3.0
    assert overridden.optimizer.lr_warmup_steps == 25
    assert overridden.optimizer.lr_cooldown_frac == 0.2
    assert overridden.optimizer.weight_decay == config.optimizer.weight_decay
    assert overridden.loss == config.loss


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

    assert (
        _scheduled_lr(
            base_lr=1.0,
            warmup_steps=0,
            cooldown_frac=0.1,
            step=90,
            total_steps=100,
        )
        == 1.0
    )
    assert (
        _scheduled_lr(
            base_lr=1.0,
            warmup_steps=0,
            cooldown_frac=0.1,
            step=95,
            total_steps=100,
        )
        == 0.5
    )
    assert _scheduled_lr(
        base_lr=1.0,
        warmup_steps=0,
        cooldown_frac=0.1,
        step=100,
        total_steps=100,
    ) == pytest.approx(0.0)


def test_lr_warmup_schedule() -> None:
    from chess_engine_4.training.cli import _scheduled_lr

    assert (
        _scheduled_lr(
            base_lr=1.0,
            warmup_steps=50,
            cooldown_frac=0.1,
            step=1,
            total_steps=100,
        )
        == 0.02
    )
    assert (
        _scheduled_lr(
            base_lr=1.0,
            warmup_steps=50,
            cooldown_frac=0.1,
            step=50,
            total_steps=100,
        )
        == 1.0
    )
    assert (
        _scheduled_lr(
            base_lr=1.0,
            warmup_steps=50,
            cooldown_frac=0.1,
            step=51,
            total_steps=100,
        )
        == 1.0
    )


def test_d64_config_builds_expected_model_size() -> None:
    config = load_training_config("configs/dense.py", d_model=64)

    assert (
        dense_parameter_count(
            d_model=config.model.d_model,
            depth=config.model.depth,
            expansion_ratio=config.model.expansion_ratio,
            activation=config.model.activation,
        )
        == 733_408
    )


@pytest.mark.parametrize(
    ("d_model", "depth", "batch_size"),
    [
        (32, 2, 2_048),
        (64, 3, 4_096),
        (128, 4, 8_192),
        (256, 5, 16_384),
        (512, 5, 32_768),
        (1_024, 6, 65_536),
        (1_536, 7, 98_304),
    ],
)
def test_dense_family_recipe(
    d_model: int,
    depth: int,
    batch_size: int,
) -> None:
    config = load_training_config("configs/dense.py", d_model=d_model)
    parameter_count = dense_parameter_count(
        d_model=config.model.d_model,
        depth=config.model.depth,
        expansion_ratio=config.model.expansion_ratio,
        activation=config.model.activation,
    )

    assert config.run.name == f"d{d_model}"
    assert config.model.depth == depth
    assert config.run.batch_size == batch_size
    assert config.run.steps == round(50 * parameter_count / batch_size)
    assert config.run.steps * batch_size / parameter_count == pytest.approx(50, rel=1e-3)


def test_dense_family_requires_aligned_width() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        load_training_config("configs/dense.py", d_model=100)


def test_dense_family_scales_training_horizon() -> None:
    baseline = load_training_config("configs/dense.py", d_model=128)
    undertrained = load_training_config(
        "configs/dense.py",
        d_model=128,
        training_ratio=0.25,
    )

    assert undertrained.run.name == "d128-r0.25"
    assert undertrained.run.training_ratio == 0.25
    assert undertrained.run.steps * undertrained.run.batch_size / 1_956_512 == pytest.approx(
        12.5,
        rel=1e-3,
    )
    assert baseline.optimizer.lr == 0.001
    assert undertrained.optimizer.lr == 0.0015


def test_precision_recipe_rejects_unknown_value(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_precision.py"
    config_path.write_text(
        """from chess_engine_4.training.config import PrecisionConfig, TrainingConfig

def config(*, d_model: int, training_ratio: float = 1.0) -> TrainingConfig:
    return TrainingConfig(precision=PrecisionConfig(recipe="fp8"))
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown quantization recipe: fp8"):
        load_training_config(config_path, d_model=64)
