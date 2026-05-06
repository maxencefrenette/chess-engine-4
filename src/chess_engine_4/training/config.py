"""Training configuration loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from chess_engine_4.model import MlpChessNetConfig
from chess_engine_4.training.losses import LossWeights


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str = "1e14"
    seed: int = 1
    flops_target: float = 1e14
    log_every: int = 10
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class DataConfig:
    batch_size: int = 1024
    max_records: int | None = None


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run: RunConfig = RunConfig()
    data: DataConfig = DataConfig()
    model: MlpChessNetConfig = MlpChessNetConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: LossWeights = LossWeights()


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    _reject_unknown_sections(raw, config_path)
    return TrainingConfig(
        run=_build_section(RunConfig, raw.get("run", {}), config_path, "run"),
        data=_build_section(DataConfig, raw.get("data", {}), config_path, "data"),
        model=_build_section(MlpChessNetConfig, raw.get("model", {}), config_path, "model"),
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
    flops_target: float | None = None,
    batch_size: int | None = None,
    d_model: int | None = None,
    depth: int | None = None,
    device: str | None = None,
) -> TrainingConfig:
    if flops_target is not None:
        config = replace(config, run=replace(config.run, flops_target=flops_target))
    if batch_size is not None:
        config = replace(config, data=replace(config.data, batch_size=batch_size))
    if d_model is not None:
        config = replace(config, model=replace(config.model, d_model=d_model))
    if depth is not None:
        config = replace(config, model=replace(config.model, depth=depth))
    if device is not None:
        config = replace(config, run=replace(config.run, device=device))
    return config


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
    allowed = {"run", "data", "model", "optimizer", "loss"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        unknown_names = ", ".join(unknown)
        raise ValueError(f"{path}: unknown section(s): {unknown_names}.")
