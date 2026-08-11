"""Canonical dense model-family recipe."""

from __future__ import annotations

from chess_engine_4.hardware import TrainingGpu
from chess_engine_4.model import DenseChessNetConfig, InputPipeline, dense_parameter_count
from chess_engine_4.training.config import (
    InfraConfig,
    OptimizerConfig,
    RunConfig,
    TrainingConfig,
)
from chess_engine_4.training.losses import LossWeights

_BASE_WIDTH = 64
_BF16_MAX_WIDTH = 512
_DEPTH = 8
_SAMPLES_PER_PARAMETER = 50.0
_BATCH32_PER_WIDTH = 32
_MINIMUM_STEPS_COEFFICIENT = 62.7575303963433
_MINIMUM_STEPS_WIDTH_EXPONENT = 0.8073049254601639
_ADAMH_LR_BY_WIDTH = {
    64: 0.0071,
    128: 0.005,
    256: 0.005,
    512: 0.0035,
    768: 0.0035,
    1024: 0.0025,
    1280: 0.0025,
}
_INPUT_PIPELINE_BY_WIDTH: dict[int, InputPipeline] = {
    64: "pageable",
    128: "pageable",
    256: "overlap",
    512: "overlap",
    768: "overlap",
    1024: "overlap",
    1280: "overlap",
}
_GPU_BY_WIDTH: dict[int, TrainingGpu] = {
    64: "RTX-PRO-6000",
    128: "RTX-PRO-6000",
    256: "RTX-PRO-6000",
    512: "B200",
    768: "B200",
    1024: "B200",
    1280: "B200",
}


def config(
    *,
    d_model: int,
    training_ratio: float = 0.2,
    history_length: int = 8,
) -> TrainingConfig:
    """Generate the current dense scaling recipe for one residual width."""

    if d_model < _BASE_WIDTH or d_model % _BASE_WIDTH != 0:
        raise ValueError("d_model must be a positive multiple of 64.")
    if d_model not in _INPUT_PIPELINE_BY_WIDTH:
        choices = ", ".join(str(width) for width in _INPUT_PIPELINE_BY_WIDTH)
        raise ValueError(f"d_model must be one of: {choices}.")
    if training_ratio <= 0:
        raise ValueError("training_ratio must be positive.")
    if not 1 <= history_length <= 8:
        raise ValueError("history_length must be in [1, 8].")

    depth = _DEPTH
    parameter_count = dense_parameter_count(
        d_model=d_model,
        depth=depth,
        history_length=history_length,
        expansion_ratio=4.0,
        activation="swiglu",
    )
    batch32 = _BATCH32_PER_WIDTH * d_model
    steps32 = round(training_ratio * _SAMPLES_PER_PARAMETER * parameter_count / batch32)
    minimum_steps = _minimum_steps(d_model)
    if steps32 >= minimum_steps:
        batch_size = batch32
        steps = steps32
    else:
        batch_size = batch32 // 2
        steps = steps32 * 2
        if steps < minimum_steps:
            raise ValueError(
                f"training_ratio={training_ratio:g} gives only {steps:,} optimizer "
                f"steps at the minimum dense batch for d_model={d_model}; "
                f"the fitted minimum is {minimum_steps:,.1f}."
            )
    return TrainingConfig(
        run=RunConfig(
            name=_run_name(d_model, training_ratio),
            seed=1,
            steps=steps,
            batch_size=batch_size,
            training_ratio=training_ratio,
        ),
        infra=InfraConfig(
            gpu=_GPU_BY_WIDTH[d_model],
            cpu_cores=8,
            dataloader_threads=8,
            dataloader_prefetch_per_thread=2,
        ),
        model=DenseChessNetConfig(
            d_model=d_model,
            depth=depth,
            history_length=history_length,
            expansion_ratio=4.0,
            activation="swiglu",
            rms_norm_eps=1e-6,
            precision="bf16" if d_model <= _BF16_MAX_WIDTH else "mxfp8",
            input_pipeline=_INPUT_PIPELINE_BY_WIDTH[d_model],
        ),
        optimizer=OptimizerConfig(
            kind="adamh",
            lr=_ADAMH_LR_BY_WIDTH[d_model],
            weight_decay=None,
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


def _minimum_steps(d_model: int) -> float:
    return _MINIMUM_STEPS_COEFFICIENT * d_model**_MINIMUM_STEPS_WIDTH_EXPONENT
