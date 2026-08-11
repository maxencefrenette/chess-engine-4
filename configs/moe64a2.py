"""Canonical alternating 64-expert, 2-active MoE model-family recipe."""

from __future__ import annotations

import math

from chess_engine_4.hardware import TrainingGpu
from chess_engine_4.model import (
    InputPipeline,
    KernelBackend,
    Moe64A2ChessNetConfig,
    moe64a2_parameter_count,
)
from chess_engine_4.training.config import (
    InfraConfig,
    OptimizerConfig,
    RunConfig,
    TrainingConfig,
)
from chess_engine_4.training.losses import LossWeights

_BASE_WIDTH = 64
_DEPTH = 8
_SAMPLES_PER_PARAMETER = 50.0
_BATCH_PER_WIDTH = 128
_LR_PARAMETER_COEFFICIENT = 518.0
_LR_PARAMETER_EXPONENT = -0.74
_LR_TRAINING_RATIO_EXPONENT = -0.18
_KERNEL_BACKEND_BY_WIDTH: dict[int, KernelBackend] = {
    128: "custom",
    256: "custom",
    384: "te",
    512: "te",
    640: "te",
    768: "te",
    1024: "te",
}
_GPU_BY_WIDTH: dict[int, TrainingGpu] = {
    128: "RTX-PRO-6000",
    256: "A100",
    384: "B200",
    512: "B200",
    640: "B200",
    768: "B200",
    1024: "B200",
}
_INPUT_PIPELINE_BY_WIDTH: dict[int, InputPipeline] = {
    128: "overlap",
    256: "overlap",
    384: "overlap",
    512: "overlap",
    640: "overlap",
    768: "overlap",
    1024: "overlap",
}


def config(
    *,
    d_model: int,
    training_ratio: float = 0.05,
    history_length: int = 8,
) -> TrainingConfig:
    """Generate the canonical moe64a2 scaling recipe for one residual width."""

    if d_model < _BASE_WIDTH or d_model % _BASE_WIDTH != 0:
        raise ValueError("d_model must be a positive multiple of 64.")
    if d_model not in _KERNEL_BACKEND_BY_WIDTH:
        choices = ", ".join(str(width) for width in _KERNEL_BACKEND_BY_WIDTH)
        raise ValueError(f"d_model must be one of: {choices}.")
    if training_ratio <= 0:
        raise ValueError("training_ratio must be positive.")
    if not 1 <= history_length <= 8:
        raise ValueError("history_length must be in [1, 8].")

    batch_size = _BATCH_PER_WIDTH * d_model
    kernel_backend = _KERNEL_BACKEND_BY_WIDTH[d_model]
    parameter_count = moe64a2_parameter_count(
        d_model=d_model,
        depth=_DEPTH,
        history_length=history_length,
        expansion_ratio=2.0,
    )
    return TrainingConfig(
        run=RunConfig(
            name=_run_name(d_model, training_ratio),
            seed=1,
            steps=round(training_ratio * _SAMPLES_PER_PARAMETER * parameter_count / batch_size),
            batch_size=batch_size,
            training_ratio=training_ratio,
        ),
        infra=InfraConfig(
            gpu=_GPU_BY_WIDTH[d_model],
            cpu_cores=8,
            dataloader_threads=8,
            dataloader_prefetch_per_thread=2,
        ),
        model=Moe64A2ChessNetConfig(
            d_model=d_model,
            depth=_DEPTH,
            history_length=history_length,
            expansion_ratio=2.0,
            activation="swiglu",
            rms_norm_eps=1e-6,
            precision="bf16" if kernel_backend == "custom" else "mxfp8",
            kernel_backend=kernel_backend,
            input_pipeline=_INPUT_PIPELINE_BY_WIDTH[d_model],
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
    suffix = "" if training_ratio == 1.0 else f"-r{training_ratio:g}"
    return f"moe64a2-d{d_model}{suffix}"


def _round_significant(value: float, *, digits: int) -> float:
    places = digits - 1 - math.floor(math.log10(abs(value)))
    return round(value, places)
