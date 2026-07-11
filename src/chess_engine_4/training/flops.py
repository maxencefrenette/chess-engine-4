"""Training compute accounting."""

from __future__ import annotations

import math

import torch

from chess_engine_4.data.leela import COMPACT_POLICY_SIZE
from chess_engine_4.model import ModelConfig
from chess_engine_4.model.export import PortableChessNet
from chess_engine_4.training.losses import lczero_loss


def measure_training_flops_per_sample(config: ModelConfig, *, batch_size: int) -> int:
    """Measure physical train-step FLOPs using the portable equivalent model on meta.

    Transformer Engine does not implement meta kernels for all fused operations. The portable
    model has the same matrix operations while using standard PyTorch operators that the FLOPs
    profiler can inspect without a GPU allocation.
    """

    profile_batch_size = min(batch_size, 8)
    with torch.device("meta"):
        model = PortableChessNet(config).train()
        planes = torch.zeros(profile_batch_size, 112, 8, 8)
        policy_indices = torch.full(
            (profile_batch_size, COMPACT_POLICY_SIZE),
            -1,
            dtype=torch.int16,
        )
        policy_indices[:, 0] = 0
        policy_probs = torch.zeros(profile_batch_size, COMPACT_POLICY_SIZE)
        policy_probs[:, 0] = 1.0
        values = torch.zeros(profile_batch_size, 6, 3)
        values[:, 0, 1] = 1.0

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
        acc_events=True,
    ) as profiler:
        loss = lczero_loss(
            model(planes),
            (policy_indices, policy_probs),
            values,
        ).task
        loss.backward()

    flops = sum(event.flops or 0 for event in profiler.key_averages())
    if flops <= 0:
        raise RuntimeError("PyTorch profiler did not report FLOPs for the training step.")
    return math.ceil(flops / profile_batch_size)


def steps_for_compute_budget(
    *,
    compute_budget: float,
    flops_per_sample: int,
    batch_size: int,
    step_penalty_k: float = 1.0,
) -> int:
    if compute_budget <= 0:
        raise ValueError("compute_budget must be positive.")
    if flops_per_sample <= 0:
        raise ValueError("flops_per_sample must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if step_penalty_k < 1.0:
        raise ValueError("step_penalty_k must be at least 1.0.")
    return math.ceil((compute_budget / (flops_per_sample * batch_size)) ** (1.0 / step_penalty_k))


def step_adjusted_compute(
    *,
    flops_per_sample: int,
    batch_size: int,
    steps: int,
    step_penalty_k: float,
) -> float:
    if flops_per_sample <= 0:
        raise ValueError("flops_per_sample must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if step_penalty_k < 1.0:
        raise ValueError("step_penalty_k must be at least 1.0.")
    return flops_per_sample * batch_size * steps**step_penalty_k
