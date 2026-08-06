"""Paired Modal benchmarks for the d128 MoE expert implementation."""

from __future__ import annotations

import argparse
import json
from typing import Any

import modal

from chess_engine_4.kernels.modal import with_cuda_kernels
from chess_engine_4.modal_train import app, base_image

BATCH_SIZE = 16_384
ACTIVE_EXPERTS = 2
EXPERT_COUNT = 64
TOKEN_ALIGNMENT = 128
CPU_CORES = 8
GPU_DOLLARS_PER_SECOND = {
    "B200": 0.001736,
    "RTX-PRO-6000": 0.000842,
}
CPU_DOLLARS_PER_CORE_SECOND = 0.0000131
benchmark_image = with_cuda_kernels(base_image)


def benchmark_moe_kernels_modal() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the d128 SM120 MoE kernel with TE MXFP8 on B200."
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("iterations must be positive")

    with modal.enable_output(), app.run():
        custom_call = _benchmark_custom.spawn(args.warmup, args.iterations)
        baseline_call = _benchmark_te.spawn(args.warmup, args.iterations)
        custom = custom_call.get()
        baseline = baseline_call.get()
    result = _comparison(custom, baseline)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_result(result)


@app.function(image=benchmark_image, gpu="RTX-PRO-6000", cpu=CPU_CORES, timeout=60 * 60)
def _benchmark_custom(warmup: int, iterations: int) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    from chess_engine_4.kernels import moe_d128_forward

    torch.manual_seed(2026)
    x, gate_up, down, probs, offsets, splits = _inputs(torch)
    reference_parts = []
    for expert, split in enumerate(splits):
        start = offsets[expert].item()
        end = start + split
        projected = F.linear(x[start:end], gate_up[expert])
        gate, up = projected.chunk(2, dim=-1)
        reference_parts.append(
            F.linear(F.silu(gate) * up, down[expert]) * probs[start:end, None]
        )
    reference = torch.cat(reference_parts)
    output = moe_d128_forward(x, gate_up, down, probs, offsets)
    torch.cuda.synchronize()
    metrics = _output_metrics(output, reference, F)
    if metrics["cosine_similarity"] < 0.999:
        raise RuntimeError(f"custom MoE output cosine similarity is too low: {metrics}")
    if metrics["mean_absolute_error"] > 0.01:
        raise RuntimeError(f"custom MoE output mean absolute error is too high: {metrics}")

    graph = torch.cuda.CUDAGraph()
    for _ in range(3):
        output = moe_d128_forward(x, gate_up, down, probs, offsets)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        output = moe_d128_forward(x, gate_up, down, probs, offsets)

    return {
        "implementation": "custom-bf16",
        "gpu": "RTX-PRO-6000",
        "device_name": torch.cuda.get_device_name(),
        "batch_size": BATCH_SIZE,
        "padded_tokens": x.shape[0],
        "median_ms": _cuda_time(graph.replay, warmup=warmup, iterations=iterations),
        "output_metrics": metrics,
    }


@app.function(image=benchmark_image, gpu="B200", cpu=CPU_CORES, timeout=60 * 60)
def _benchmark_te(warmup: int, iterations: int) -> dict[str, Any]:
    import torch

    from chess_engine_4.model import Moe64A2ChessNetConfig
    from chess_engine_4.model.moe import MoeBlock
    from chess_engine_4.model.transformer_engine import autocast_context, mxfp8_recipe, te

    torch.manual_seed(2026)
    x, _, _, probs, _, splits = _inputs(torch)
    split_tensor = torch.tensor(splits, device="cuda", dtype=torch.int64)
    experts = MoeBlock(Moe64A2ChessNetConfig(d_model=128)).experts.cuda().eval()
    with autocast_context("mxfp8"):
        graph = te().make_graphed_callables(
            experts,
            (x, split_tensor, probs, split_tensor),
            allow_unused_input=True,
            enabled=True,
            recipe=mxfp8_recipe(),
        )

    return {
        "implementation": "te-mxfp8",
        "gpu": "B200",
        "device_name": torch.cuda.get_device_name(),
        "batch_size": BATCH_SIZE,
        "padded_tokens": x.shape[0],
        "median_ms": _cuda_time(
            lambda: graph(x, split_tensor, probs, split_tensor),
            warmup=warmup,
            iterations=iterations,
        ),
    }


def _inputs(torch: Any) -> tuple[Any, Any, Any, Any, Any, list[int]]:
    routed_tokens = BATCH_SIZE * ACTIVE_EXPERTS
    tokens_per_expert = routed_tokens // EXPERT_COUNT
    padded_tokens = BATCH_SIZE * ACTIVE_EXPERTS + EXPERT_COUNT * (TOKEN_ALIGNMENT - 1)
    padded_tokens = _round_up(padded_tokens, TOKEN_ALIGNMENT)
    splits = [tokens_per_expert] * EXPERT_COUNT
    splits[-1] += padded_tokens - sum(splits)
    offsets = [0]
    for split in splits:
        offsets.append(offsets[-1] + split)
    return (
        torch.randn(padded_tokens, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(64, 512, 128, device="cuda", dtype=torch.bfloat16) / 128**0.5,
        torch.randn(64, 128, 256, device="cuda", dtype=torch.bfloat16) / 256**0.5,
        torch.rand(padded_tokens, device="cuda", dtype=torch.bfloat16),
        torch.tensor(offsets, device="cuda", dtype=torch.int32),
        splits,
    )


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _cuda_time(function: Any, *, warmup: int, iterations: int) -> float:
    import statistics

    import torch

    with torch.no_grad():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _output_metrics(output: Any, reference: Any, functional: Any) -> dict[str, float]:
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


def _comparison(custom: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    for measurement in (custom, baseline):
        gpu = measurement["gpu"]
        dollars_per_second = (
            GPU_DOLLARS_PER_SECOND[gpu] + CPU_CORES * CPU_DOLLARS_PER_CORE_SECOND
        )
        measurement["dollars_per_million_batches"] = (
            measurement["median_ms"] / 1_000 * dollars_per_second * 1_000_000
        )
    return {
        "custom": custom,
        "baseline": baseline,
        "latency_speedup": baseline["median_ms"] / custom["median_ms"],
        "cost_efficiency_gain": (
            baseline["dollars_per_million_batches"]
            / custom["dollars_per_million_batches"]
        ),
    }


def _print_result(result: dict[str, Any]) -> None:
    custom = result["custom"]
    baseline = result["baseline"]
    print(
        f"custom={custom['implementation']} gpu={custom['gpu']} "
        f"latency={custom['median_ms']:.3f}ms "
        f"cost_per_million_batches=${custom['dollars_per_million_batches']:.2f}"
    )
    print(
        f"baseline={baseline['implementation']} gpu={baseline['gpu']} "
        f"latency={baseline['median_ms']:.3f}ms "
        f"cost_per_million_batches=${baseline['dollars_per_million_batches']:.2f}"
    )
    print(
        f"latency_speedup={result['latency_speedup']:.3f}x "
        f"cost_efficiency_gain={result['cost_efficiency_gain']:.3f}x"
    )
    print(f"correctness={custom['output_metrics']}")
