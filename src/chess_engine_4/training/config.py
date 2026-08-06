"""Training configuration loading and command-line overrides."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from chess_engine_4.model import KernelBackend, ModelConfig, Precision, model_config_from_dict
from chess_engine_4.training.losses import LossWeights

TrainingGpu = Literal["B200", "RTX-PRO-6000"]
TRAINING_GPUS: tuple[TrainingGpu, ...] = ("B200", "RTX-PRO-6000")


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str = "d64"
    seed: int = 1
    steps: int = 1_000
    batch_size: int = 1_024
    training_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.training_ratio <= 0:
            raise ValueError("training_ratio must be positive.")


@dataclass(frozen=True, slots=True)
class InfraConfig:
    gpu: TrainingGpu = "B200"
    cpu_cores: int = 8
    dataloader_threads: int = 4
    dataloader_prefetch_per_thread: int = 2

    def __post_init__(self) -> None:
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive.")


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_warmup_steps: int = 0
    lr_cooldown_frac: float = 0.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run: RunConfig = RunConfig()
    infra: InfraConfig = InfraConfig()
    model: ModelConfig = model_config_from_dict({})
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: LossWeights = LossWeights()


def load_training_config(
    path: str | Path,
    *,
    d_model: int,
    training_ratio: float | None = None,
    history_length: int | None = None,
) -> TrainingConfig:
    config_path = Path(path)
    if config_path.suffix != ".py":
        raise ValueError(f"{config_path}: training configs must be Python files.")

    module_name = f"_chess_engine_4_config_{abs(hash(config_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load training config {config_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    factory = getattr(module, "config", None)
    if not callable(factory):
        raise ValueError(
            f"{config_path}: expected callable config(*, d_model: int, training_ratio: float)."
        )
    kwargs: dict[str, Any] = {"d_model": d_model}
    if training_ratio is not None:
        kwargs["training_ratio"] = training_ratio
    if history_length is not None:
        kwargs["history_length"] = history_length
    result = factory(**kwargs)
    if not isinstance(result, TrainingConfig):
        raise ValueError(f"{config_path}: config() must return TrainingConfig.")
    return result


def training_config_from_dict(values: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        run=RunConfig(**values.get("run", {})),
        infra=InfraConfig(**values.get("infra", {})),
        model=model_config_from_dict(values.get("model", {})),
        optimizer=OptimizerConfig(**values.get("optimizer", {})),
        loss=LossWeights(**values.get("loss", {})),
    )


def validate_training_hardware(config: TrainingConfig) -> None:
    if config.infra.gpu == "RTX-PRO-6000" and config.model.precision != "bf16":
        raise ValueError(
            "RTX-PRO-6000 training requires precision='bf16'; "
            "Transformer Engine 2.17 does not support MXFP8 or NVFP4 on SM120."
        )
    if config.model.kernel_backend != "custom":
        return
    if config.model.kind == "dense" and config.infra.gpu == "B200":
        return
    if (
        config.model.kind == "moe64a2"
        and config.model.d_model in {128, 256, 512}
        and config.model.precision == "bf16"
        and config.infra.gpu == "RTX-PRO-6000"
    ):
        return
    raise ValueError(
        "custom kernels require either a dense model on B200 or "
        "a supported BF16 moe64a2 model on RTX-PRO-6000"
    )


def with_overrides(
    config: TrainingConfig,
    *,
    seed: int | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    depth: int | None = None,
    expansion_ratio: float | None = None,
    history_length: int | None = None,
    activation: str | None = None,
    lr: float | None = None,
    max_grad_norm: float | None = None,
    lr_warmup_steps: int | None = None,
    lr_cooldown_frac: float | None = None,
    gpu: TrainingGpu | None = None,
    dataloader_threads: int | None = None,
    dataloader_prefetch_per_thread: int | None = None,
    quantization_recipe: Precision | None = None,
    kernel_backend: KernelBackend | None = None,
) -> TrainingConfig:
    if seed is not None:
        config = replace(config, run=replace(config.run, seed=seed))
    if steps is not None:
        config = replace(config, run=replace(config.run, steps=steps))
    if batch_size is not None:
        config = replace(config, run=replace(config.run, batch_size=batch_size))
    if depth is not None:
        config = replace(config, model=replace(config.model, depth=depth))
    if expansion_ratio is not None:
        config = replace(
            config,
            model=replace(config.model, expansion_ratio=expansion_ratio),
        )
    if history_length is not None:
        config = replace(config, model=replace(config.model, history_length=history_length))
    if activation is not None:
        config = replace(config, model=replace(config.model, activation=activation))
    if lr is not None:
        config = replace(config, optimizer=replace(config.optimizer, lr=lr))
    if max_grad_norm is not None:
        config = replace(
            config,
            optimizer=replace(config.optimizer, max_grad_norm=max_grad_norm),
        )
    if lr_warmup_steps is not None:
        config = replace(
            config,
            optimizer=replace(config.optimizer, lr_warmup_steps=lr_warmup_steps),
        )
    if lr_cooldown_frac is not None:
        config = replace(
            config,
            optimizer=replace(config.optimizer, lr_cooldown_frac=lr_cooldown_frac),
        )
    if gpu is not None:
        config = replace(config, infra=replace(config.infra, gpu=gpu))
    if dataloader_threads is not None:
        config = replace(
            config,
            infra=replace(config.infra, dataloader_threads=dataloader_threads),
        )
    if dataloader_prefetch_per_thread is not None:
        config = replace(
            config,
            infra=replace(
                config.infra,
                dataloader_prefetch_per_thread=dataloader_prefetch_per_thread,
            ),
        )
    if quantization_recipe is not None:
        config = replace(
            config,
            model=replace(config.model, precision=quantization_recipe),
        )
    if kernel_backend is not None:
        config = replace(
            config,
            model=replace(config.model, kernel_backend=kernel_backend),
        )
    return config
