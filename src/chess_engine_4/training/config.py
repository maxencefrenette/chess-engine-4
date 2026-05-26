"""Training configuration loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from chess_engine_4.model import ModelConfig, model_config_from_dict
from chess_engine_4.training.losses import LossWeights


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str = "1e14"
    seed: int = 1
    compute_budget: float = 1e14
    step_penalty_k: float = 1.0


@dataclass(frozen=True, slots=True)
class InfraConfig:
    gpu_type: str = "l4"
    dataloader_threads: int = 4
    dataloader_prefetch_per_thread: int = 2


@dataclass(frozen=True, slots=True)
class DataConfig:
    batch_size: int = 1024


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run: RunConfig = RunConfig()
    infra: InfraConfig = InfraConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = model_config_from_dict({})
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: LossWeights = LossWeights()


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    _reject_unknown_sections(raw, config_path)
    return TrainingConfig(
        run=_build_section(RunConfig, raw.get("run", {}), config_path, "run"),
        infra=_build_section(InfraConfig, raw.get("infra", {}), config_path, "infra"),
        data=_build_section(DataConfig, raw.get("data", {}), config_path, "data"),
        model=_build_model_config(raw.get("model", {}), config_path),
        optimizer=_build_section(
            OptimizerConfig,
            raw.get("optimizer", {}),
            config_path,
            "optimizer",
        ),
        loss=_build_section(LossWeights, raw.get("loss", {}), config_path, "loss"),
    )


def with_overrides(
    config: TrainingConfig,
    *,
    compute_budget: float | None = None,
    step_penalty_k: float | None = None,
    batch_size: int | None = None,
    d_model: int | None = None,
    depth: int | None = None,
    num_heads: int | None = None,
    lr: float | None = None,
    router_aux: float | None = None,
    dataloader_threads: int | None = None,
    dataloader_prefetch_per_thread: int | None = None,
) -> TrainingConfig:
    if compute_budget is not None:
        config = replace(config, run=replace(config.run, compute_budget=compute_budget))
    if step_penalty_k is not None:
        config = replace(config, run=replace(config.run, step_penalty_k=step_penalty_k))
    if batch_size is not None:
        config = replace(config, data=replace(config.data, batch_size=batch_size))
    if d_model is not None:
        config = replace(config, model=replace(config.model, d_model=d_model))
    if depth is not None:
        config = replace(config, model=replace(config.model, depth=depth))
    if num_heads is not None:
        if not hasattr(config.model, "num_heads"):
            raise ValueError(f"model kind {config.model.kind!r} does not support num_heads.")
        config = replace(config, model=replace(config.model, num_heads=num_heads))
    if lr is not None:
        config = replace(config, optimizer=replace(config.optimizer, lr=lr))
    if router_aux is not None:
        config = replace(config, loss=replace(config.loss, router_aux=router_aux))
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
    return config


def _build_model_config(values: object, path: Path) -> ModelConfig:
    if not isinstance(values, dict):
        raise ValueError(f"{path}: [model] must be a table.")
    try:
        return model_config_from_dict(values)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _build_section[ConfigT](
    section_type: type[ConfigT],
    values: object,
    path: Path,
    section_name: str,
) -> ConfigT:
    if not isinstance(values, dict):
        raise ValueError(f"{path}: [{section_name}] must be a table.")

    field_names = {field.name for field in fields(section_type)}
    unknown = sorted(set(values) - field_names)
    if unknown:
        unknown_names = ", ".join(unknown)
        raise ValueError(f"{path}: [{section_name}] has unknown key(s): {unknown_names}.")

    return section_type(**values)


def _reject_unknown_sections(raw: dict[str, Any], path: Path) -> None:
    allowed = {"run", "infra", "data", "model", "optimizer", "loss"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        unknown_names = ", ".join(unknown)
        raise ValueError(f"{path}: unknown section(s): {unknown_names}.")
