"""Paired Modal benchmarks for specialized MoE expert implementations."""

from __future__ import annotations

import argparse
import json
from typing import Any

import modal

from chess_engine_4.hardware import gpu_spec, hardware_dollars_per_second
from chess_engine_4.kernels.benchmarking import (
    cuda_time,
    cuda_time_backward,
    named_tensor_metrics,
    tensor_metrics,
)
from chess_engine_4.kernels.modal import with_cuda_kernels
from chess_engine_4.modal_train import app, base_image

ACTIVE_EXPERTS = 2
EXPERT_COUNT = 64
TOKEN_ALIGNMENT = 128
CPU_CORES = 8
benchmark_image = with_cuda_kernels(base_image)


def benchmark_moe_kernels_modal() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a custom MoE kernel with TE MXFP8 on B200."
    )
    parser.add_argument("--d-model", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument(
        "--custom-gpu",
        choices=("A100", "B200", "RTX-PRO-6000"),
        default="B200",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("iterations must be positive")

    custom_function = custom_benchmark_function(args.custom_gpu)
    with modal.enable_output(), app.run():
        custom_call = custom_function.spawn(
            args.custom_gpu, args.d_model, args.warmup, args.iterations
        )
        baseline_call = _benchmark_te.spawn(args.d_model, args.warmup, args.iterations)
        custom = custom_call.get()
        baseline = baseline_call.get()
    result = _comparison(custom, baseline)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_result(result)


def _benchmark_custom(
    gpu: str,
    d_model: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    from chess_engine_4.kernels import moe_trainable

    expected = gpu_spec(gpu)
    capability = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()
    if capability != expected.capability or expected.device_name not in device_name:
        raise RuntimeError(
            f"configured gpu={gpu!r}, but Modal provided {device_name} "
            f"SM{capability[0]}{capability[1]}"
        )

    torch.manual_seed(2026)
    correctness = _make_inputs(torch, d_model, [128] * EXPERT_COUNT)
    correctness_tensors = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in correctness[:4]
    )
    custom_output = moe_trainable(
        *correctness_tensors,
        correctness[4],
    )
    reference_tensors = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in correctness[:4]
    )
    reference_output = _reference_moe(
        *reference_tensors,
        splits=correctness[5],
        functional=F,
    )
    gradient = torch.randn_like(custom_output)
    custom_gradients = torch.autograd.grad(
        custom_output,
        correctness_tensors,
        gradient,
    )
    reference_gradients = torch.autograd.grad(
        reference_output,
        reference_tensors,
        gradient,
    )
    output_metrics = tensor_metrics(custom_output, reference_output, F)
    gradient_metrics = named_tensor_metrics(
        ("input", "gate_up_weight", "down_weight", "route_probs"),
        custom_gradients,
        reference_gradients,
        F,
    )
    if not all(tensor.isfinite().all() for tensor in (*custom_gradients, *reference_gradients)):
        raise RuntimeError(f"custom MoE produced non-finite gradients: {gradient_metrics}")
    if output_metrics["cosine_similarity"] < 0.999:
        raise RuntimeError(f"custom MoE output cosine similarity is too low: {output_metrics}")
    if min(metric["cosine_similarity"] for metric in gradient_metrics.values()) < 0.99:
        raise RuntimeError(f"custom MoE gradient cosine similarity is too low: {gradient_metrics}")

    class CustomExperts(torch.nn.Module):
        def __init__(self, gate_up_weight: Any, down_weight: Any) -> None:
            super().__init__()
            self.gate_up_weight = torch.nn.Parameter(gate_up_weight)
            self.down_weight = torch.nn.Parameter(down_weight)

        def forward(self, x: Any, probs: Any, offsets: Any) -> Any:
            return moe_trainable(
                x,
                self.gate_up_weight,
                self.down_weight,
                probs,
                offsets,
            )

    batch_size = 128 * d_model
    x, gate_up, down, probs, offsets, _ = _inputs(
        torch,
        d_model,
        batch_size,
        assign_slack=False,
    )
    x.requires_grad_(True)
    probs.requires_grad_(True)
    experts = CustomExperts(gate_up, down).cuda().train()
    graph = torch.cuda.make_graphed_callables(
        experts,
        (x, probs, offsets),
        allow_unused_input=True,
    )

    def run_forward() -> Any:
        return graph(x, probs, offsets)

    graph_output = run_forward()
    backward_inputs = (x, experts.gate_up_weight, experts.down_weight, probs)
    backward_gradient = torch.randn_like(graph_output)
    del graph_output

    return {
        "implementation": "custom-bf16",
        "gpu": gpu,
        "device_name": device_name,
        "d_model": d_model,
        "batch_size": batch_size,
        "padded_tokens": x.shape[0],
        "forward_ms": cuda_time(run_forward, warmup=warmup, iterations=iterations),
        "backward_ms": cuda_time_backward(
            run_forward,
            backward_inputs,
            backward_gradient,
            warmup=warmup,
            iterations=iterations,
        ),
        "output_metrics": output_metrics,
        "gradient_metrics": gradient_metrics,
    }


def custom_benchmark_function(gpu: str) -> modal.Function:
    return app.function(
        image=benchmark_image,
        gpu=gpu,
        cpu=CPU_CORES,
        timeout=60 * 60,
        name=f"moe_custom_benchmark_{gpu.lower().replace('-', '_')}",
    )(_benchmark_custom)


@app.function(image=benchmark_image, gpu="B200", cpu=CPU_CORES, timeout=60 * 60)
def _benchmark_te(d_model: int, warmup: int, iterations: int) -> dict[str, Any]:
    import torch

    from chess_engine_4.model import Moe64A2ChessNetConfig
    from chess_engine_4.model.moe import MoeBlock
    from chess_engine_4.model.transformer_engine import autocast_context, mxfp8_recipe, te

    torch.manual_seed(2026)
    batch_size = 128 * d_model
    x, _, _, probs, _, splits = _inputs(
        torch,
        d_model,
        batch_size,
        assign_slack=True,
    )
    x.requires_grad_(True)
    probs.requires_grad_(True)
    split_tensor = torch.tensor(splits, device="cuda", dtype=torch.int64)

    class TeExperts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = MoeBlock(Moe64A2ChessNetConfig(d_model=d_model)).experts

        def forward(self, x: Any, probs: Any, splits: Any) -> Any:
            return self.experts(x, splits, probs, splits)

    experts = TeExperts().cuda().train()
    with autocast_context("mxfp8"):
        graph = te().make_graphed_callables(
            experts,
            (x, probs, split_tensor),
            allow_unused_input=True,
            enabled=True,
            recipe=mxfp8_recipe(),
        )

    def run_forward() -> Any:
        return graph(x, probs, split_tensor)

    graph_output = run_forward()
    backward_inputs = (x, *tuple(experts.parameters()), probs)
    backward_gradient = torch.randn_like(graph_output)
    del graph_output
    return {
        "implementation": "te-mxfp8",
        "gpu": "B200",
        "device_name": torch.cuda.get_device_name(),
        "d_model": d_model,
        "batch_size": batch_size,
        "padded_tokens": x.shape[0],
        "forward_ms": cuda_time(run_forward, warmup=warmup, iterations=iterations),
        "backward_ms": cuda_time_backward(
            run_forward,
            backward_inputs,
            backward_gradient,
            warmup=warmup,
            iterations=iterations,
        ),
    }


def _inputs(
    torch: Any,
    d_model: int,
    batch_size: int,
    *,
    assign_slack: bool,
) -> tuple[Any, Any, Any, Any, Any, list[int]]:
    routed_tokens = batch_size * ACTIVE_EXPERTS
    tokens_per_expert = routed_tokens // EXPERT_COUNT
    padded_tokens = batch_size * ACTIVE_EXPERTS + EXPERT_COUNT * (TOKEN_ALIGNMENT - 1)
    padded_tokens = _round_up(padded_tokens, TOKEN_ALIGNMENT)
    splits = [tokens_per_expert] * EXPERT_COUNT
    if assign_slack:
        splits[-1] += padded_tokens - sum(splits)
    return _make_inputs(torch, d_model, splits, rows=padded_tokens)


def _make_inputs(
    torch: Any,
    d_model: int,
    splits: list[int],
    *,
    rows: int | None = None,
) -> tuple[Any, Any, Any, Any, Any, list[int]]:
    padded_tokens = rows or sum(splits)
    hidden_dim = 2 * d_model
    offsets = [0]
    for split in splits:
        offsets.append(offsets[-1] + split)
    return (
        torch.randn(padded_tokens, d_model, device="cuda", dtype=torch.bfloat16),
        torch.randn(
            EXPERT_COUNT,
            2 * hidden_dim,
            d_model,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / d_model**0.5,
        torch.randn(
            EXPERT_COUNT,
            d_model,
            hidden_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / hidden_dim**0.5,
        torch.rand(padded_tokens, device="cuda", dtype=torch.bfloat16),
        torch.tensor(offsets, device="cuda", dtype=torch.int32),
        splits,
    )


def _reference_moe(
    x: Any,
    gate_up: Any,
    down: Any,
    probs: Any,
    *,
    splits: list[int],
    functional: Any,
) -> Any:
    import torch

    outputs = []
    start = 0
    for expert, split in enumerate(splits):
        end = start + split
        projected = functional.linear(x[start:end], gate_up[expert])
        gate, up = projected.chunk(2, dim=-1)
        outputs.append(
            functional.linear(functional.silu(gate) * up, down[expert]) * probs[start:end, None]
        )
        start = end
    return torch.cat(outputs)


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _comparison(custom: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    for measurement in (custom, baseline):
        gpu = measurement["gpu"]
        measurement["training_ms"] = measurement["forward_ms"] + measurement["backward_ms"]
        dollars_per_second = hardware_dollars_per_second(gpu, CPU_CORES)
        measurement["dollars_per_million_batches"] = (
            measurement["training_ms"] / 1_000 * dollars_per_second * 1_000_000
        )
    return {
        "custom": custom,
        "baseline": baseline,
        "forward_speedup": baseline["forward_ms"] / custom["forward_ms"],
        "backward_speedup": baseline["backward_ms"] / custom["backward_ms"],
        "training_speedup": baseline["training_ms"] / custom["training_ms"],
        "cost_efficiency_gain": (
            baseline["dollars_per_million_batches"] / custom["dollars_per_million_batches"]
        ),
    }


def _print_result(result: dict[str, Any]) -> None:
    custom = result["custom"]
    baseline = result["baseline"]
    print(
        f"custom={custom['implementation']} gpu={custom['gpu']} "
        f"forward={custom['forward_ms']:.3f}ms backward={custom['backward_ms']:.3f}ms "
        f"cost_per_million_batches=${custom['dollars_per_million_batches']:.2f}"
    )
    print(
        f"baseline={baseline['implementation']} gpu={baseline['gpu']} "
        f"forward={baseline['forward_ms']:.3f}ms backward={baseline['backward_ms']:.3f}ms "
        f"cost_per_million_batches=${baseline['dollars_per_million_batches']:.2f}"
    )
    print(
        f"training_speedup={result['training_speedup']:.3f}x "
        f"cost_efficiency_gain={result['cost_efficiency_gain']:.3f}x"
    )
    print(f"correctness={custom['output_metrics']}")
