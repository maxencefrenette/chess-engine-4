"""Correctness and latency benchmarks for project-owned CUDA kernels."""

from __future__ import annotations

from typing import Any

KERNEL_NAME = "dense-mxfp8"
SUPPORTED_WIDTHS = (128, 256, 512, 1024, 2048)
MIN_COSINE_SIMILARITY = 0.999
MAX_MEAN_ABSOLUTE_ERROR = 1e-3
MIN_GRADIENT_COSINE_SIMILARITY = 0.99


def benchmark_dense_layer(
    *,
    d_model: int,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    from chess_engine_4.kernels import dense_mxfp8_forward, dense_mxfp8_trainable
    from chess_engine_4.kernels.dense import _dense_bf16_forward
    from chess_engine_4.model.dense import DenseBlock
    from chess_engine_4.model.transformer_engine import autocast_context

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    x = torch.randn(batch_size, d_model, device="cuda", dtype=torch.bfloat16)
    block = DenseBlock(
        d_model=d_model,
        hidden_dim=4 * d_model,
        rms_norm_eps=1e-6,
        activation="swiglu",
    ).cuda().eval()
    layer = block.layer
    norm_weight = layer.layer_norm_weight.detach()
    gate_up_weight = layer.fc1_weight.detach()
    down_weight = layer.fc2_weight.detach()

    with torch.no_grad(), autocast_context("mxfp8"):
        te_output = block(x)
    with torch.no_grad():
        custom_output = dense_mxfp8_forward(
            x,
            norm_weight,
            gate_up_weight,
            down_weight,
        )
    torch.cuda.synchronize()

    difference = custom_output.float() - te_output.float()
    mean_abs_error = difference.abs().mean().item()
    max_abs_error = difference.abs().max().item()
    cosine_similarity = F.cosine_similarity(
        custom_output.float().flatten(),
        te_output.float().flatten(),
        dim=0,
    ).item()
    if not torch.isfinite(custom_output).all():
        raise RuntimeError("custom kernel produced non-finite output")
    if cosine_similarity < MIN_COSINE_SIMILARITY:
        raise RuntimeError(
            f"custom kernel cosine similarity {cosine_similarity:.8f} is below "
            f"{MIN_COSINE_SIMILARITY}"
        )
    if mean_abs_error > MAX_MEAN_ABSOLUTE_ERROR:
        raise RuntimeError(
            f"custom kernel mean absolute error {mean_abs_error:.8f} exceeds "
            f"{MAX_MEAN_ABSOLUTE_ERROR}"
        )

    gradient = torch.randn_like(custom_output)
    custom_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (x, norm_weight, gate_up_weight, down_weight)
    )
    custom_gradients = torch.autograd.grad(
        dense_mxfp8_trainable(*custom_inputs),
        custom_inputs,
        gradient,
    )
    bf16_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (x, norm_weight, gate_up_weight, down_weight)
    )
    bf16_gradients = torch.autograd.grad(
        _dense_bf16_forward(*bf16_inputs, eps=1e-6),
        bf16_inputs,
        gradient,
    )
    te_x = x.detach().clone().requires_grad_(True)
    with autocast_context("mxfp8"):
        te_train_output = block(te_x)
    te_gradients = torch.autograd.grad(
        te_train_output,
        (te_x, layer.layer_norm_weight, layer.fc1_weight, layer.fc2_weight),
        gradient,
    )
    gradient_metrics_vs_te = _gradient_metrics(custom_gradients, te_gradients, F)
    gradient_metrics_vs_bf16 = _gradient_metrics(custom_gradients, bf16_gradients, F)
    gradient_cosine_similarity = min(
        metrics["cosine_similarity"] for metrics in gradient_metrics_vs_te.values()
    )
    if gradient_cosine_similarity < MIN_GRADIENT_COSINE_SIMILARITY:
        raise RuntimeError(
            f"custom kernel gradient cosine similarity {gradient_cosine_similarity:.8f} "
            f"is below {MIN_GRADIENT_COSINE_SIMILARITY}: {gradient_metrics_vs_te}"
        )

    def run_custom() -> None:
        dense_mxfp8_forward(
            x,
            norm_weight,
            gate_up_weight,
            down_weight,
        )

    def run_te() -> None:
        with autocast_context("mxfp8"):
            block(x)

    custom_ms = _cuda_time(run_custom, warmup=warmup, iterations=iterations)
    te_ms = _cuda_time(run_te, warmup=warmup, iterations=iterations)
    backward_custom_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (x, norm_weight, gate_up_weight, down_weight)
    )
    te_backward_x = x.detach().clone().requires_grad_(True)

    def build_custom_backward() -> Any:
        return dense_mxfp8_trainable(*backward_custom_inputs)

    def build_te_backward() -> Any:
        with autocast_context("mxfp8"):
            return block(te_backward_x)

    custom_backward_ms = _cuda_time_backward(
        build_custom_backward,
        backward_custom_inputs,
        gradient,
        warmup=warmup,
        iterations=iterations,
    )
    te_backward_ms = _cuda_time_backward(
        build_te_backward,
        (te_backward_x, layer.layer_norm_weight, layer.fc1_weight, layer.fc2_weight),
        gradient,
        warmup=warmup,
        iterations=iterations,
    )
    return {
        "kernel": KERNEL_NAME,
        "d_model": d_model,
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations": iterations,
        "custom_ms": custom_ms,
        "te_ms": te_ms,
        "speedup_vs_te": te_ms / custom_ms,
        "custom_backward_ms": custom_backward_ms,
        "te_backward_ms": te_backward_ms,
        "backward_speedup_vs_te": te_backward_ms / custom_backward_ms,
        "mean_abs_error_vs_te": mean_abs_error,
        "max_abs_error_vs_te": max_abs_error,
        "cosine_similarity_vs_te": cosine_similarity,
        "gradient_cosine_similarity_vs_te": gradient_cosine_similarity,
        "gradient_metrics_vs_te": gradient_metrics_vs_te,
        "gradient_metrics_vs_bf16": gradient_metrics_vs_bf16,
        "device_name": torch.cuda.get_device_name(),
    }


def _gradient_metrics(
    custom_gradients: tuple[Any, ...],
    reference_gradients: tuple[Any, ...],
    functional: Any,
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean_absolute_error": (custom.float() - reference.float()).abs().mean().item(),
            "cosine_similarity": functional.cosine_similarity(
                custom.float().flatten(),
                reference.float().flatten(),
                dim=0,
            ).item(),
        }
        for name, custom, reference in zip(
            ("input", "norm_weight", "gate_up_weight", "down_weight"),
            custom_gradients,
            reference_gradients,
            strict=True,
        )
    }


def _cuda_time(function: Any, *, warmup: int, iterations: int) -> float:
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
    return start.elapsed_time(end) / iterations


def _cuda_time_backward(
    build_output: Any,
    inputs: tuple[Any, ...],
    gradient: Any,
    *,
    warmup: int,
    iterations: int,
) -> float:
    import torch

    for _ in range(warmup):
        torch.autograd.grad(build_output(), inputs, gradient)
    torch.cuda.synchronize()
    total_ms = 0.0
    for _ in range(iterations):
        output = build_output()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.autograd.grad(output, inputs, gradient)
        end.record()
        end.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / iterations
