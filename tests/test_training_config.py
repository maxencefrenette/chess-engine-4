from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pytest

from chess_engine_4.modal_train import (
    add_training_config_arguments,
    print_launch_summary,
    resolve_training_config,
)
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


def test_training_config_ignores_legacy_router_auxiliary_loss() -> None:
    values = asdict(load_training_config("configs/moe64a2.py", d_model=256))
    values["loss"]["router_aux"] = 0.01

    config = training_config_from_dict(values)

    assert "router_aux" not in asdict(config.loss)


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


def test_launch_summary_records_sampling_rate(capsys) -> None:
    config = load_training_config("configs/dense.py", d_model=64)

    print_launch_summary(config)

    summary = capsys.readouterr().out
    assert "sampling_rate=1" in summary


def test_moe_launch_summary_records_quantile_routing(capsys) -> None:
    config = load_training_config("configs/moe64a2.py", d_model=128)

    print_launch_summary(config)

    assert "router_load_balancing=quantile" in capsys.readouterr().out


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

    b200_config = with_overrides(config, gpu="B200")
    validate_training_hardware(b200_config)
    assert resolve_training_kernel(b200_config).variant == "moe-sm100-bf16"


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


def test_custom_dense_accepts_sm120_bf16() -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=128),
        gpu="RTX-PRO-6000",
        quantization_recipe="bf16",
        kernel_backend="custom",
    )

    validate_training_hardware(config)
    assert resolve_training_kernel(config).variant == "dense-sm120-bf16"


@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4"])
def test_custom_dense_rejects_sm120_low_precision(precision: str) -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=128),
        gpu="RTX-PRO-6000",
        quantization_recipe=precision,
        kernel_backend="custom",
    )

    with pytest.raises(ValueError, match="SM120 require precision='bf16'"):
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


def test_custom_dense_accepts_d1280_before_launch() -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=1280),
        kernel_backend="custom",
    )

    assert resolve_training_kernel(config).variant == "dense-sm100-mxfp8"


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
    [
        (128, False),
        (256, True),
        (512, True),
        (768, True),
        (1024, True),
        (1280, True),
    ],
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


@pytest.mark.parametrize("gpu", ["H100", "H200"])
@pytest.mark.parametrize("config_path", ["configs/dense.py", "configs/moe64a2.py"])
def test_hopper_accepts_explicit_custom_bf16_training(
    gpu: str,
    config_path: str,
) -> None:
    config = with_overrides(
        load_training_config(config_path, d_model=128),
        gpu=gpu,
        quantization_recipe="bf16",
        kernel_backend="custom",
    )

    assert resolve_training_kernel(config).variant in {
        "dense-sm90-bf16",
        "moe-sm90-bf16",
    }
    assert training_config_from_dict(asdict(config)) == config


@pytest.mark.parametrize("gpu", ["H100", "H200"])
@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4"])
def test_hopper_rejects_blackwell_only_precision_before_launch(
    gpu: str,
    precision: str,
) -> None:
    config = with_overrides(
        load_training_config("configs/dense.py", d_model=256),
        gpu=gpu,
        quantization_recipe=precision,
        kernel_backend="custom",
    )

    with pytest.raises(ValueError, match="SM90 require precision='bf16'"):
        validate_training_hardware(config)


@pytest.mark.parametrize(
    ("config_path", "gpu", "extra_args", "expected_backend"),
    [
        ("configs/dense.py", "H100", [], "te"),
        ("configs/moe64a2.py", "H200", [], "custom"),
        ("configs/moe64a2.py", "H100", ["--kernel-backend", "te"], "te"),
    ],
)
def test_gpu_cli_override_does_not_change_kernel_backend(
    config_path: str,
    gpu: str,
    extra_args: list[str],
    expected_backend: str,
) -> None:
    parser = argparse.ArgumentParser()
    add_training_config_arguments(parser, include_steps=True)
    args = parser.parse_args(
        ["--config", config_path, "--d-model", "128", "--gpu", gpu, *extra_args]
    )

    config = resolve_training_config(args)

    assert config.infra.gpu == gpu
    assert config.model.kernel_backend == expected_backend


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
    assert "router_aux" not in asdict(config.loss)
    assert config.model.kernel_backend == "custom"
    assert config.model.precision == "bf16"
    assert config.infra.gpu == "RTX-PRO-6000"
    assert training_config_from_dict(asdict(config)) == config


@pytest.mark.parametrize("d_model", [384, 640, 768])
def test_moe64a2_new_widths_use_te_mxfp8(d_model: int) -> None:
    config = load_training_config("configs/moe64a2.py", d_model=d_model)

    assert config.model.kernel_backend == "te"
    assert config.model.precision == "mxfp8"
    assert config.infra.gpu == "B200"


@pytest.mark.parametrize("d_model", [32, 64, 2048])
def test_moe64a2_recipe_rejects_unsupported_widths(
    d_model: int,
) -> None:
    with pytest.raises(ValueError, match="d_model must be"):
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
        (64, 8, 2_048),
        (128, 8, 4_096),
        (256, 8, 8_192),
        (512, 8, 16_384),
        (768, 8, 24_576),
        (1_024, 8, 32_768),
        (1_280, 8, 40_960),
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


@pytest.mark.parametrize("d_model", [128, 256, 512, 768, 1_024])
def test_moe64a2_batch_size_scales_exactly_with_width(d_model: int) -> None:
    config = load_training_config("configs/moe64a2.py", d_model=d_model)

    assert config.run.batch_size == 128 * d_model


@pytest.mark.parametrize(
    ("d_model", "gpu"),
    [
        (64, "RTX-PRO-6000"),
        (128, "RTX-PRO-6000"),
        (256, "RTX-PRO-6000"),
        (512, "B200"),
        (768, "B200"),
        (1024, "B200"),
        (1280, "B200"),
    ],
)
def test_dense_family_selects_cost_efficient_gpu(d_model: int, gpu: str) -> None:
    config = load_training_config("configs/dense.py", d_model=d_model)

    assert config.infra.gpu == gpu


def test_dense_family_requires_aligned_width() -> None:
    with pytest.raises(ValueError, match="multiple of 64"):
        load_training_config("configs/dense.py", d_model=100)


def test_dense_family_rejects_removed_d32_width() -> None:
    with pytest.raises(ValueError, match="multiple of 64"):
        load_training_config("configs/dense.py", d_model=32)


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
