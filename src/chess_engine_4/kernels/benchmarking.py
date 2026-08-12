"""Shared correctness and CUDA timing primitives for kernel benchmarks."""

from __future__ import annotations

from typing import Any


def tensor_metrics(output: Any, reference: Any, functional: Any) -> dict[str, float]:
    difference = output.float() - reference.float()
    return {
        "mean_absolute_error": difference.abs().mean().item(),
        "max_absolute_error": difference.abs().max().item(),
        "cosine_similarity": functional.cosine_similarity(
            output.float().flatten(),
            reference.float().flatten(),
            dim=0,
        ).item(),
    }


def compare_gradients(
    names: tuple[str, ...],
    custom_gradients: tuple[Any, ...],
    reference_gradients: tuple[Any, ...],
    functional: Any,
) -> dict[str, dict[str, float | bool]]:
    return {
        name: {
            "mean_absolute_error": (custom.float() - reference.float()).abs().mean().item(),
            "cosine_similarity": functional.cosine_similarity(
                custom.float().flatten(),
                reference.float().flatten(),
                dim=0,
            ).item(),
            "custom_finite": bool(custom.isfinite().all().item()),
            "reference_finite": bool(reference.isfinite().all().item()),
            "custom_abs_max": custom.float().abs().max().item(),
            "reference_abs_max": reference.float().abs().max().item(),
        }
        for name, custom, reference in zip(
            names,
            custom_gradients,
            reference_gradients,
            strict=True,
        )
    }


def named_tensor_metrics(
    names: tuple[str, ...],
    custom_values: tuple[Any, ...],
    reference_values: tuple[Any, ...],
    functional: Any,
) -> dict[str, dict[str, float]]:
    return {
        name: tensor_metrics(custom, reference, functional)
        for name, custom, reference in zip(
            names,
            custom_values,
            reference_values,
            strict=True,
        )
    }


def cuda_time(function: Any, *, warmup: int, iterations: int) -> float:
    import statistics

    import torch

    with torch.no_grad():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return float(statistics.median(samples))


def cuda_time_backward(
    build_output: Any,
    inputs: tuple[Any, ...],
    gradient: Any,
    *,
    warmup: int,
    iterations: int,
) -> float:
    import statistics

    import torch

    for _ in range(warmup):
        torch.autograd.grad(build_output(), inputs, gradient)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        output = build_output()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.autograd.grad(output, inputs, gradient)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return float(statistics.median(samples))
