from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from chess_engine_4.modal_train import print_launch_summary
from chess_engine_4.model import dense_parameter_count
from chess_engine_4.training.config import (
    load_training_config,
    training_config_from_dict,
    validate_training_hardware,
    with_overrides,
)


def test_load_training_config_requires_factory(tmp_path: Path) -> None:
    config_path = tmp_path / "train.py"
    config_path.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected callable config"):
        load_training_config(config_path, d_model=64)


def test_with_overrides_keeps_config_as_source_of_truth() -> None:
    config = load_training_config("configs/dense.py", d_model=64)

    overridden = with_overrides(
        config,
        seed=2,
        steps=321,
        batch_size=4,
        depth=2,
        expansion_ratio=2.0,
        history_length=4,
        activation="silu",
        quantization_recipe="nvfp4",
        lr=0.001,
        max_grad_norm=3.0,
        lr_warmup_steps=25,
        lr_cooldown_frac=0.2,
        gpu="RTX-PRO-6000",
    )

    assert overridden.run.seed == 2
    assert overridden.run.steps == 321
    assert overridden.run.batch_size == 4
    assert overridden.model.d_model == 64
    assert overridden.model.depth == 2
    assert overridden.model.expansion_ratio == 2.0
    assert overridden.model.history_length == 4
    assert overridden.model.activation == "silu"
    assert overridden.model.precision == "nvfp4"
    assert overridden.optimizer.lr == 0.001
    assert overridden.optimizer.max_grad_norm == 3.0
    assert overridden.optimizer.lr_warmup_steps == 25
    assert overridden.optimizer.lr_cooldown_frac == 0.2
    assert overridden.infra.gpu == "RTX-PRO-6000"
    assert overridden.optimizer.weight_decay == config.optimizer.weight_decay
    assert overridden.loss == config.loss


def test_training_config_round_trips_through_dict() -> None:
    config = load_training_config("configs/dense.py", d_model=64)

    assert training_config_from_dict(asdict(config)) == config


def test_launch_summary_records_retention(capsys) -> None:
    config = load_training_config("configs/dense.py", d_model=64)

    print_launch_summary(
        config,
        data_retention_rate=0.25,
    )

    summary = capsys.readouterr().out
    assert "data_retention_rate=0.25" in summary


def test_rtx_pro_6000_rejects_unsupported_low_precision_recipe() -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=1024),
        gpu="RTX-PRO-6000",
    )

    with pytest.raises(ValueError, match="requires precision='bf16'"):
        validate_training_hardware(config)


@pytest.mark.parametrize("d_model", [128, 256, 512])
def test_custom_moe_kernel_requires_supported_bf16_width_on_rtx_pro_6000(
    d_model: int,
) -> None:
    config = load_training_config("configs/moe64a2.py", d_model=d_model)
    if d_model == 512:
        config = with_overrides(
            config,
            gpu="RTX-PRO-6000",
            quantization_recipe="bf16",
            kernel_backend="custom",
        )

    validate_training_hardware(config)

    with pytest.raises(ValueError, match="supported BF16 moe64a2"):
        validate_training_hardware(with_overrides(config, gpu="B200"))


@pytest.mark.parametrize("config_path", ["configs/dense.py", "configs/moe64a2.py"])
def test_a100_accepts_custom_bf16_kernels(config_path: str) -> None:
    config = load_training_config(config_path, d_model=128)
    config = with_overrides(
        config,
        gpu="A100",
        quantization_recipe="bf16",
        kernel_backend="custom",
    )

    validate_training_hardware(config)


def test_a100_rejects_low_precision_recipe() -> None:
    config = load_training_config("configs/dense.py", d_model=128)
    config = with_overrides(config, gpu="A100", quantization_recipe="mxfp8")

    with pytest.raises(ValueError, match="A100 training requires precision='bf16'"):
        validate_training_hardware(config)


def test_moe64a2_family_recipe_round_trips() -> None:
    config = load_training_config("configs/moe64a2.py", d_model=128)

    assert config.run.name == "moe64a2-d128-r0.05"
    assert config.run.training_ratio == 0.05
    assert config.run.batch_size == 16_384
    assert config.optimizer.lr == 2.8e-3
    assert config.model.kind == "moe64a2"
    assert config.model.num_experts == 64
    assert config.model.num_active_experts == 2
    assert config.model.expansion_ratio == 2.0
    assert config.loss.router_aux == 0.01
    assert config.model.kernel_backend == "custom"
    assert config.model.precision == "bf16"
    assert config.infra.gpu == "RTX-PRO-6000"
    assert training_config_from_dict(asdict(config)) == config


@pytest.mark.parametrize("d_model", [32, 64, 2048])
def test_moe64a2_recipe_rejects_unsupported_widths(
    d_model: int,
) -> None:
    with pytest.raises(ValueError, match="d_model must be one of"):
        load_training_config("configs/moe64a2.py", d_model=d_model)


def test_training_profile_config_validates_steps() -> None:
    from chess_engine_4.training.profiling import TrainingProfileConfig

    profile = TrainingProfileConfig(warmup_steps=50, profile_steps=200)

    assert profile.total_steps == 250
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainingProfileConfig(warmup_steps=-1)
    with pytest.raises(ValueError, match="profile_steps"):
        TrainingProfileConfig(profile_steps=0)


def test_loss_ema_consumes_every_pending_step() -> None:
    from chess_engine_4.training.cli import _update_ema_metrics

    metrics: dict[str, float | int] = {
        "loss/task": 6.0,
        "metrics/policy_top1": 0.5,
    }
    ema_metrics: dict[str, float] = {}

    _update_ema_metrics(
        metrics,
        ema_metrics,
        loss_tasks=[10.0, 8.0, 6.0],
    )

    assert metrics["loss/task[ema=0.99]"] == pytest.approx(9.9402)
    assert metrics["metrics/policy_top1[ema=0.9]"] == pytest.approx(0.5)

    metrics["metrics/policy_top1"] = 0.7
    _update_ema_metrics(metrics, ema_metrics, loss_tasks=[6.0])

    assert metrics["metrics/policy_top1[ema=0.9]"] == pytest.approx(0.52)


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
        == 979_488
    )


@pytest.mark.parametrize(
    ("d_model", "depth", "batch_size"),
    [
        (32, 8, 1_024),
        (64, 8, 2_048),
        (128, 8, 4_096),
        (256, 8, 8_192),
        (512, 8, 16_384),
        (1_024, 8, 32_768),
        (1_536, 8, 49_152),
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

    assert config.run.name == f"d{d_model}-r0.2"
    assert config.run.training_ratio == 0.2
    assert config.model.depth == depth
    assert config.run.batch_size == batch_size
    assert config.run.steps == round(10 * parameter_count / batch_size)
    assert config.run.steps * batch_size / parameter_count == pytest.approx(10, rel=1e-3)


@pytest.mark.parametrize(
    ("d_model", "gpu"),
    [
        (32, "RTX-PRO-6000"),
        (64, "RTX-PRO-6000"),
        (128, "RTX-PRO-6000"),
        (256, "RTX-PRO-6000"),
        (512, "B200"),
        (1024, "B200"),
        (2048, "B200"),
    ],
)
def test_dense_family_selects_cost_efficient_gpu(d_model: int, gpu: str) -> None:
    config = load_training_config("configs/dense.py", d_model=d_model)

    assert config.infra.gpu == gpu


def test_dense_family_requires_aligned_width() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        load_training_config("configs/dense.py", d_model=100)


def test_dense_family_scales_training_horizon() -> None:
    baseline = load_training_config("configs/dense.py", d_model=128, training_ratio=1.0)
    undertrained = load_training_config(
        "configs/dense.py",
        d_model=128,
        training_ratio=0.25,
    )

    assert undertrained.run.name == "d128-r0.25"
    assert undertrained.run.training_ratio == 0.25
    assert undertrained.run.steps * undertrained.run.batch_size / 2_743_456 == pytest.approx(
        12.5,
        rel=1e-3,
    )
    assert baseline.optimizer.lr == 0.00055
    assert undertrained.optimizer.lr == 0.0013


def test_dense_family_recomputes_recipe_for_history_length() -> None:
    full = load_training_config("configs/dense.py", d_model=128, history_length=8)
    shortened = load_training_config("configs/dense.py", d_model=128, history_length=2)

    assert shortened.model.history_length == 2
    assert shortened.run.steps < full.run.steps
    assert shortened.optimizer.lr > full.optimizer.lr
