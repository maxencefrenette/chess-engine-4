"""Training FLOPs measurement."""

from __future__ import annotations

import math

import torch

from chess_engine_4.data.leela import POLICY_SIZE
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
    planes = torch.zeros(profile_batch_size, 112, 8, 8, device=device)
    policy = torch.full((profile_batch_size, POLICY_SIZE), -1.0, device=device)
    policy[:, 0] = 1.0
    values = torch.zeros(profile_batch_size, 6, 3, device=device)
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
    if flops <= 0:
        raise RuntimeError("PyTorch profiler did not report FLOPs for the training step.")
    return math.ceil(flops / profile_batch_size)


def steps_for_flops_target(
    *,
    flops_target: float,
    flops_per_sample: int,
    batch_size: int,
) -> int:
    if flops_target <= 0:
        raise ValueError("flops_target must be positive.")
    if flops_per_sample <= 0:
        raise ValueError("flops_per_sample must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return math.ceil(flops_target / (flops_per_sample * batch_size))
