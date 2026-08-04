"""Canonical dense model-family recipe."""

from __future__ import annotations

import math

from chess_engine_4.model import DenseChessNetConfig, dense_parameter_count
from chess_engine_4.training.config import (
    InfraConfig,
    OptimizerConfig,
    PrecisionConfig,
    RunConfig,
    TrainingConfig,
)
from chess_engine_4.training.losses import LossWeights

_BASE_WIDTH = 32
_DEPTH_INTERCEPT = 2.5
_DEPTH_PER_WIDTH_DOUBLING = 0.85
_SAMPLES_PER_PARAMETER = 50.0
_BATCH_PER_WIDTH = 32
_LR_PARAMETER_COEFFICIENT = 0.96
_LR_PARAMETER_EXPONENT = -0.45
_LR_TRAINING_RATIO_EXPONENT = -0.3


def config(*, d_model: int, training_ratio: float = 1.0) -> TrainingConfig:
    """Generate the current dense scaling recipe for one residual width."""

    if d_model < _BASE_WIDTH or d_model % _BASE_WIDTH != 0:
        raise ValueError("d_model must be a positive multiple of 32.")
    if training_ratio <= 0:
        raise ValueError("training_ratio must be positive.")

    width_scale = d_model / _BASE_WIDTH
    depth = max(
        2,
        math.floor(_DEPTH_INTERCEPT + _DEPTH_PER_WIDTH_DOUBLING * math.log2(width_scale)),
    )
    batch_size = _round_batch_size(_BATCH_PER_WIDTH * d_model)
    parameter_count = dense_parameter_count(
        d_model=d_model,
        depth=depth,
        expansion_ratio=4.0,
        activation="swiglu",
    )
    return TrainingConfig(
        run=RunConfig(
            name=_run_name(d_model, training_ratio),
            seed=1,
            steps=round(
                training_ratio * _SAMPLES_PER_PARAMETER * parameter_count / batch_size
            ),
            batch_size=batch_size,
            training_ratio=training_ratio,
        ),
        infra=InfraConfig(
            cpu_cores=8,
            dataloader_threads=8,
            dataloader_prefetch_per_thread=2,
        ),
        precision=PrecisionConfig(recipe="mxfp8"),
        model=DenseChessNetConfig(
            d_model=d_model,
            depth=depth,
            expansion_ratio=4.0,
            activation="swiglu",
            rms_norm_eps=1e-6,
        ),
        optimizer=OptimizerConfig(
            lr=_round_significant(
                _LR_PARAMETER_COEFFICIENT
                * parameter_count**_LR_PARAMETER_EXPONENT
                * training_ratio**_LR_TRAINING_RATIO_EXPONENT,
                digits=2,
            ),
            weight_decay=0.01,
            max_grad_norm=1.0,
            lr_warmup_steps=0,
            lr_cooldown_frac=0.1,
        ),
        loss=LossWeights(policy=1.0, value=1.0, moves_left=1.0),
    )


def _run_name(d_model: int, training_ratio: float) -> str:
    if training_ratio == 1.0:
        return f"d{d_model}"
    return f"d{d_model}-r{training_ratio:g}"


def _round_batch_size(value: float) -> int:
    ladder = [round(multiplier * 2**power) for power in range(5, 23) for multiplier in (1.0, 1.5)]
    return min(ladder, key=lambda candidate: abs(math.log(candidate / value)))


def _round_significant(value: float, *, digits: int) -> float:
    places = digits - 1 - math.floor(math.log10(abs(value)))
    return round(value, places)
