"""Training FLOPs measurement."""

from __future__ import annotations

import math

import torch

from chess_engine_4.data.leela import COMPACT_POLICY_SIZE
from chess_engine_4.training.losses import lczero_loss


def measure_training_flops_per_sample(
    model: torch.nn.Module,
    *,
    batch_size: int,
) -> int:
    """Measure train-step FLOPs per sample on the PyTorch meta device."""

    profile_batch_size = min(batch_size, 8)
    devices = {parameter.device.type for parameter in model.parameters()}
    if devices != {"meta"}:
        raise ValueError("FLOPs measurement expects a model on the meta device.")

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    device = torch.device("meta")
    profile_dtype = getattr(model, "flops_profile_dtype", torch.float32)
    planes = torch.zeros(profile_batch_size, 112, 8, 8, device=device, dtype=profile_dtype)
    policy_indices = torch.full(
        (profile_batch_size, COMPACT_POLICY_SIZE),
        -1,
        device=device,
        dtype=torch.int16,
    )
    policy_indices[:, 0] = 0
    policy_probs = torch.zeros(
        profile_batch_size,
        COMPACT_POLICY_SIZE,
        device=device,
        dtype=profile_dtype,
    )
    policy_probs[:, 0] = 1.0
    policy = (policy_indices, policy_probs)
    values = torch.zeros(profile_batch_size, 6, 3, device=device, dtype=profile_dtype)
    values[:, 0, 1] = 1.0

    activities = [torch.profiler.ProfilerActivity.CPU]

    with torch.profiler.profile(
        activities=activities,
        with_flops=True,
        acc_events=True,
    ) as profiler:
        loss = lczero_loss(model(planes), policy, values).total
        loss.backward()

    model.zero_grad(set_to_none=True)
    model.train(was_training)

    flops = sum(event.flops or 0 for event in profiler.key_averages())
    extra_flops_per_sample = _extra_training_flops_per_sample(model)
    flops += extra_flops_per_sample * profile_batch_size
    if flops <= 0:
        raise RuntimeError("PyTorch profiler did not report FLOPs for the training step.")
    return math.ceil(flops / profile_batch_size)


def _extra_training_flops_per_sample(model: torch.nn.Module) -> int:
    extra_flops = getattr(model, "extra_training_flops_per_sample", None)
    if extra_flops is None:
        return 0
    if not callable(extra_flops):
        raise TypeError("extra_training_flops_per_sample must be callable.")
    measured = int(extra_flops())
    if measured < 0:
        raise ValueError("extra_training_flops_per_sample must be non-negative.")
    return measured


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
