import argparse
from dataclasses import asdict

import pytest

from chess_engine_4.backends.modal.deployment import apply_modal_defaults
from chess_engine_4.training.config import (
    load_training_config,
    training_config_from_dict,
    validate_training_hardware,
    with_overrides,
)
from chess_engine_4.training.launch import add_training_config_arguments, resolve_training_config


@pytest.mark.parametrize(
    ("family", "width", "gpu"),
    [
        ("dense", 64, "RTX-PRO-6000"),
        ("dense", 128, "RTX-PRO-6000"),
        ("dense", 256, "RTX-PRO-6000"),
        ("dense", 512, "B200"),
        ("dense", 768, "B200"),
        ("dense", 1024, "B200"),
        ("dense", 1280, "B200"),
        ("moe64a2", 128, "RTX-PRO-6000"),
        ("moe64a2", 256, "A100"),
        ("moe64a2", 384, "B200"),
        ("moe64a2", 512, "B200"),
        ("moe64a2", 640, "B200"),
        ("moe64a2", 768, "B200"),
        ("moe64a2", 1024, "B200"),
    ],
)
def test_modal_placement_preserves_recipe_numerics(family: str, width: int, gpu: str) -> None:
    recipe = load_training_config(f"configs/{family}.py", d_model=width)
    assert recipe.infra.gpu is None
    deployed = apply_modal_defaults(recipe)
    assert deployed.infra.require_gpu() == gpu
    assert deployed.infra.cpu_cores == 8
    assert deployed.infra.dataloader_threads == 8
    assert deployed.infra.dataloader_prefetch_per_thread == 2
    for field in ("model", "optimizer", "run", "loss"):
        assert getattr(deployed, field) == getattr(recipe, field)
    validate_training_hardware(deployed)
    assert training_config_from_dict(asdict(deployed)) == deployed


def test_unresolved_recipe_cannot_launch() -> None:
    recipe = load_training_config("configs/dense.py", d_model=64)
    with pytest.raises(ValueError, match="GPU is unresolved"):
        validate_training_hardware(recipe)


def test_launch_selects_backend_defaults_without_changing_recipe() -> None:
    parser = argparse.ArgumentParser()
    add_training_config_arguments(parser, include_steps=True, gpu_choices=("RTX-5070", "H100"))
    args = parser.parse_args(["--d-model", "64"])
    local = resolve_training_config(args, default_gpu="RTX-5070")
    cloud = resolve_training_config(args, deployment=apply_modal_defaults)
    assert local.infra.gpu == "RTX-5070"
    assert cloud.infra.gpu == "RTX-PRO-6000"
    assert local.run == cloud.run
    assert local.model == cloud.model


def test_explicit_deployment_overrides_win_and_old_checkpoints_load() -> None:
    parser = argparse.ArgumentParser()
    add_training_config_arguments(parser, include_steps=True, gpu_choices=("H100",))
    parser.add_argument("--cpu-cores", type=int)
    args = parser.parse_args([
        "--d-model", "64", "--gpu", "H100", "--cpu-cores", "2",
        "--dataloader-threads", "3", "--dataloader-prefetch-per-thread", "1",
    ])
    config = resolve_training_config(args, deployment=apply_modal_defaults)
    assert asdict(config.infra) == {
        "gpu": "H100", "cpu_cores": 2,
        "dataloader_threads": 3, "dataloader_prefetch_per_thread": 1,
    }
    assert training_config_from_dict(asdict(config)) == config
    assert apply_modal_defaults(config) == config


@pytest.mark.parametrize("backend", ["te", "custom"])
@pytest.mark.parametrize(("field", "value"), [
    ("precision", "nvfp4"), ("precision", "fp16"),
    ("activation", "gelu"), ("activation", "geglu"),
    ("activation", "silu"), ("activation", "srelu"),
])
def test_unsupported_training_configs_fail_before_hardware(
    backend: str, field: str, value: str,
) -> None:
    values = asdict(load_training_config("configs/dense.py", d_model=64))
    values["model"][field] = value
    values["model"]["kernel_backend"] = backend
    config = training_config_from_dict(values)
    with pytest.raises(ValueError, match="Production training supports only"):
        validate_training_hardware(config)


@pytest.mark.parametrize(("flag", "value"), [
    ("--quantization-recipe", "nvfp4"), ("--activation", "gelu"),
    ("--activation", "geglu"), ("--activation", "silu"), ("--activation", "srelu"),
])
def test_retired_options_rejected_by_cli(flag: str, value: str) -> None:
    parser = argparse.ArgumentParser()
    add_training_config_arguments(parser, include_steps=True, gpu_choices=("RTX-5070",))
    with pytest.raises(SystemExit) as error:
        parser.parse_args([flag, value])
    assert error.value.code == 2


def test_execution_overrides_do_not_mutate_the_recipe() -> None:
    recipe = load_training_config("configs/dense.py", d_model=64)
    deployed = with_overrides(recipe, gpu="RTX-5070", dataloader_threads=2)
    assert recipe.infra.gpu is None
    assert recipe.infra.dataloader_threads == 8
    assert deployed.infra.dataloader_threads == 2
