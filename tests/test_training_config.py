from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from chess_engine_4.model import dense_parameter_count
from chess_engine_4.training.config import (
    load_training_config,
    resolve_training_kernel,
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


def test_legacy_fixed_lc0_geometry_is_accepted_and_removed() -> None:
    values = asdict(load_training_config("configs/dense.py", d_model=64))
    values["model"].update(input_planes=112, board_size=8, policy_size=1858)

    config = training_config_from_dict(values)

    assert "input_planes" not in asdict(config.model)
    assert "board_size" not in asdict(config.model)
    assert "policy_size" not in asdict(config.model)


@pytest.mark.parametrize(
    ("name", "value"),
    [("input_planes", 111), ("board_size", 10), ("policy_size", 1860)],
)
def test_legacy_non_lc0_geometry_is_rejected(name: str, value: int) -> None:
    values = asdict(load_training_config("configs/dense.py", d_model=64))
    values["model"][name] = value

    with pytest.raises(ValueError, match=rf"{name} must be"):
        training_config_from_dict(values)


def test_rtx_pro_6000_rejects_unsupported_low_precision_recipe() -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=1024),
        gpu="RTX-PRO-6000",
    )

    with pytest.raises(ValueError, match="mxfp8.*not supported on SM120"):
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

    with pytest.raises(ValueError, match="support SM80 and SM120, got SM100"):
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

    with pytest.raises(ValueError, match="mxfp8.*not supported on SM80"):
        validate_training_hardware(config)


def test_rtx_pro_6000_accepts_te_nvfp4() -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=128),
        gpu="RTX-PRO-6000",
        quantization_recipe="nvfp4",
        kernel_backend="te",
    )

    assert resolve_training_kernel(config).variant == "te-nvfp4"


def test_custom_dense_rejects_sm120_before_launch() -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=128),
        gpu="RTX-PRO-6000",
        quantization_recipe="bf16",
        kernel_backend="custom",
    )

    with pytest.raises(ValueError, match="support SM80 and SM100, got SM120"):
        validate_training_hardware(config)


def test_custom_dense_rejects_incompatible_expansion_ratio() -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=256),
        gpu="B200",
        expansion_ratio=2.0,
        quantization_recipe="mxfp8",
        kernel_backend="custom",
    )

    with pytest.raises(ValueError, match="expansion_ratio=4"):
        validate_training_hardware(config)


def test_custom_dense_rejects_uncompiled_width_before_launch() -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=1536),
        kernel_backend="custom",
    )

    with pytest.raises(ValueError, match="require d_model in"):
        validate_training_hardware(config)


def test_custom_dense_rejects_nvfp4_before_launch() -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=256),
        gpu="B200",
        quantization_recipe="nvfp4",
        kernel_backend="custom",
    )

    with pytest.raises(ValueError, match="require precision='bf16' or 'mxfp8'"):
        validate_training_hardware(config)


def test_custom_moe_rejects_incompatible_expansion_ratio() -> None:
    config = with_overrides(
        load_training_config("configs/moe64a2.py", d_model=128),
        expansion_ratio=4.0,
    )

    with pytest.raises(ValueError, match="expansion_ratio=2"):
        validate_training_hardware(config)


@pytest.mark.parametrize(
    ("d_model", "supported"),
    [(128, False), (256, True), (512, True), (1024, True), (2048, True)],
)
def test_custom_mxfp8_dense_accepts_mechanically_compatible_widths(
    d_model: int,
    supported: bool,
) -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=d_model),
        gpu="B200",
        quantization_recipe="mxfp8",
        kernel_backend="custom",
    )

    if supported:
        assert resolve_training_kernel(config).variant == "dense-sm100-mxfp8"
    else:
        with pytest.raises(ValueError, match="d_model divisible by 256"):
            validate_training_hardware(config)


@pytest.mark.parametrize(
    ("gpu", "batch_size", "supported", "alignment"),
    [
        ("A100", 16, True, 16),
        ("A100", 17, False, 16),
        ("B200", 128, True, 128),
        ("B200", 127, False, 128),
    ],
)
def test_custom_dense_validates_architecture_row_alignment(
    gpu: str,
    batch_size: int,
    supported: bool,
    alignment: int,
) -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=256),
        gpu=gpu,
        batch_size=batch_size,
        quantization_recipe="bf16",
        kernel_backend="custom",
    )

    if supported:
        validate_training_hardware(config)
    else:
        with pytest.raises(ValueError, match=f"rows divisible by {alignment}"):
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
