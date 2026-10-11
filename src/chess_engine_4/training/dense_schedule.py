"""Shared dense training budgets and the fitted minimum-step schedule."""

from __future__ import annotations

DENSE_SAMPLES_PER_PARAMETER = 50.0
DENSE_BATCH_PER_WIDTH = 32
_MINIMUM_STEPS_COEFFICIENT = 62.7575303963433
_MINIMUM_STEPS_WIDTH_EXPONENT = 0.8073049254601639


def dense_training_schedule(
    *, d_model: int, parameter_count: int, training_ratio: float
) -> tuple[int, int]:
    """Return batch size and steps, preserving the rounded 32d sample budget."""
    batch_size = DENSE_BATCH_PER_WIDTH * d_model
    steps = round(training_ratio * DENSE_SAMPLES_PER_PARAMETER * parameter_count / batch_size)
    minimum_steps = _MINIMUM_STEPS_COEFFICIENT * d_model**_MINIMUM_STEPS_WIDTH_EXPONENT
    if steps < minimum_steps:
        batch_size //= 2
        steps *= 2
    if steps < minimum_steps:
        raise ValueError(
            f"training_ratio={training_ratio:g} gives only {steps:,} optimizer "
            f"steps at the minimum dense batch for d_model={d_model}; "
            f"the fitted minimum is {minimum_steps:,.1f}."
        )
    return batch_size, steps
