"""Correctness and latency benchmarks for project-owned CUDA kernels."""

from __future__ import annotations

from typing import Any

from chess_engine_4.kernels.benchmarking import (
    compare_gradients,
    cuda_time,
    cuda_time_backward,
    tensor_metrics,
)

KERNEL_NAME = "dense-custom"
MIN_COSINE_SIMILARITY = 0.999
MAX_MEAN_ABSOLUTE_ERROR = 1e-3
MIN_GRADIENT_COSINE_SIMILARITY = 0.99


def benchmark_dense_layer(
    *,
    d_model: int,
    batch_size: int,
    precision: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    from chess_engine_4.kernels import dense_block_forward, dense_block_trainable
    from chess_engine_4.model.dense import DenseBlock
    from chess_engine_4.model.transformer_engine import (
        autocast_context,
        quantization_recipe,
        te,
    )

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    x = torch.randn(batch_size, d_model, device="cuda", dtype=torch.bfloat16)
    block = (
        DenseBlock(
            d_model=d_model,
            hidden_dim=4 * d_model,
            rms_norm_eps=1e-6,
            activation="swiglu",
            precision=precision,
        )
        .cuda()
        .eval()
    )
    layer = block.layer
    norm_weight = layer.layer_norm_weight.detach()
    gate_up_weight = layer.fc1_weight.detach()
    down_weight = layer.fc2_weight.detach()

    with torch.no_grad(), autocast_context(precision):
        te_output = block(x)
    with torch.no_grad():
        custom_output = dense_block_forward(
            x,
            norm_weight,
            gate_up_weight,
            down_weight,
            precision=precision,
        )
    torch.cuda.synchronize()

    reference_name = f"te_{precision}"
    output_metrics = tensor_metrics(custom_output, te_output, F)
    if not torch.isfinite(custom_output).all():
        raise RuntimeError("custom kernel produced non-finite output")
    if output_metrics["cosine_similarity"] < MIN_COSINE_SIMILARITY:
        raise RuntimeError(
            f"custom kernel cosine similarity against {reference_name} "
            f"{output_metrics['cosine_similarity']:.8f} is below {MIN_COSINE_SIMILARITY}"
        )
    if output_metrics["mean_absolute_error"] > MAX_MEAN_ABSOLUTE_ERROR:
        raise RuntimeError(
            f"custom kernel mean absolute error against {reference_name} "
            f"{output_metrics['mean_absolute_error']:.8f} exceeds {MAX_MEAN_ABSOLUTE_ERROR}"
        )

    gradient = torch.randn_like(custom_output)
    custom_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (x, norm_weight, gate_up_weight, down_weight)
    )
    custom_gradients = torch.autograd.grad(
        dense_block_trainable(*custom_inputs, precision=precision),
        custom_inputs,
        gradient,
    )
    te_x = x.detach().clone().requires_grad_(True)
    with autocast_context(precision):
        te_train_output = block(te_x)
    te_gradients = torch.autograd.grad(
        te_train_output,
        (te_x, layer.layer_norm_weight, layer.fc1_weight, layer.fc2_weight),
        gradient,
    )
    gradient_metrics = compare_gradients(
        ("input", "norm_weight", "gate_up_weight", "down_weight"),
        custom_gradients,
        te_gradients,
        F,
    )
    non_finite_gradients = [
        name
        for name, metrics in gradient_metrics.items()
        if not metrics["custom_finite"] or not metrics["reference_finite"]
    ]
    if non_finite_gradients:
        raise RuntimeError(f"non-finite gradients for {non_finite_gradients}: {gradient_metrics}")
    gradient_cosine_similarity = min(
        metrics["cosine_similarity"] for metrics in gradient_metrics.values()
    )
    if gradient_cosine_similarity < MIN_GRADIENT_COSINE_SIMILARITY:
        raise RuntimeError(
            f"custom kernel gradient cosine similarity against {reference_name} "
            f"{gradient_cosine_similarity:.8f} is below "
            f"{MIN_GRADIENT_COSINE_SIMILARITY}: {gradient_metrics}"
        )

    graph_inputs = {
        "te": x.detach().clone().requires_grad_(True),
        "custom": x.detach().clone().requires_grad_(True),
    }
    graph_blocks = {
        name: DenseBlock(
            d_model=d_model,
            hidden_dim=4 * d_model,
            rms_norm_eps=1e-6,
            activation="swiglu",
            precision=precision,
        )
        .cuda()
        .train()
        for name in graph_inputs
    }
    for graph_block in graph_blocks.values():
        graph_block.load_state_dict(block.state_dict())
    graph_blocks["custom"].enable_custom_kernels()
    recipe = quantization_recipe(precision)
    with autocast_context(precision):
        if recipe is None:
            graphs = {
                name: torch.cuda.make_graphed_callables(
                    graph_block,
                    (graph_inputs[name],),
                    allow_unused_input=True,
                )
                for name, graph_block in graph_blocks.items()
            }
        else:
            graphs = {
                name: te().make_graphed_callables(
                    graph_block,
                    (graph_inputs[name],),
                    allow_unused_input=True,
                    enabled=True,
                    recipe=recipe,
                )
                for name, graph_block in graph_blocks.items()
            }

    def run_custom() -> None:
        graphs["custom"](graph_inputs["custom"])

    def run_te() -> None:
        graphs["te"](graph_inputs["te"])

    custom_ms = cuda_time(run_custom, warmup=warmup, iterations=iterations)
    te_ms = cuda_time(run_te, warmup=warmup, iterations=iterations)
    backward_custom_inputs = (
        graph_inputs["custom"],
        graph_blocks["custom"].layer.layer_norm_weight,
        graph_blocks["custom"].layer.fc1_weight,
        graph_blocks["custom"].layer.fc2_weight,
    )
    backward_te_inputs = (
        graph_inputs["te"],
        graph_blocks["te"].layer.layer_norm_weight,
        graph_blocks["te"].layer.fc1_weight,
        graph_blocks["te"].layer.fc2_weight,
    )

    def build_custom_backward() -> Any:
        return graphs["custom"](graph_inputs["custom"])

    def build_te_backward() -> Any:
        return graphs["te"](graph_inputs["te"])

    custom_backward_ms = cuda_time_backward(
        build_custom_backward,
        backward_custom_inputs,
        gradient,
        warmup=warmup,
        iterations=iterations,
    )
    te_backward_ms = cuda_time_backward(
        build_te_backward,
        backward_te_inputs,
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
        "execution": "cuda_graph",
        "custom_ms": custom_ms,
        "te_ms": te_ms,
        "speedup_vs_te": te_ms / custom_ms,
        "custom_backward_ms": custom_backward_ms,
        "te_backward_ms": te_backward_ms,
        "backward_speedup_vs_te": te_backward_ms / custom_backward_ms,
        "reference": reference_name,
        "output_metrics": output_metrics,
        "gradient_metrics": gradient_metrics,
        "device_name": torch.cuda.get_device_name(),
    }
