"""Paired TE/custom training benchmarks on isolated Modal B200s."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import modal

from chess_engine_4.hardware import TRAINING_GPUS
from chess_engine_4.kernels.capabilities import SUPPORTED_DENSE_WIDTHS
from chess_engine_4.kernels.modal import with_cuda_kernels
from chess_engine_4.modal_kernels import benchmark_dense_layer
from chess_engine_4.modal_train import (
    DEFAULT_CONFIG_PATH,
    REMOTE_DATA_PATH,
    app,
    base_image,
    data_volume,
)
from chess_engine_4.training.config import load_training_config, with_overrides

LEVELS = ("layer", "step", "production")
benchmark_image = with_cuda_kernels(base_image)


def benchmark_training_modal() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark paired TE and custom dense training paths on Modal."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    widths = parser.add_mutually_exclusive_group()
    supported_widths = sorted(SUPPORTED_DENSE_WIDTHS)
    widths.add_argument("--d-model", type=int, choices=supported_widths)
    widths.add_argument("--widths", type=int, nargs="+", choices=supported_widths)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gpu", choices=TRAINING_GPUS, default="B200")
    parser.add_argument("--level", choices=("all", *LEVELS), default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.batch_size is not None and args.widths is not None:
        parser.error("batch-size can only be used with one d-model")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("iterations must be positive")

    selected_widths = args.widths or [args.d_model or 256]
    configs = []
    for width in selected_widths:
        config = load_training_config(args.config, d_model=width)
        config = with_overrides(config, gpu=args.gpu, quantization_recipe="bf16")
        if args.batch_size is not None:
            config = with_overrides(config, batch_size=args.batch_size)
        configs.append(asdict(config))

    levels = list(LEVELS) if args.level == "all" else [args.level]
    benchmark_function = training_benchmark_function(args.gpu)
    with modal.enable_output(), app.run():
        results = list(
            benchmark_function.starmap(
                (config, levels, args.warmup, args.iterations) for config in configs
            )
        )
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return
    _print_results(results)


def _benchmark_training(
    config_values: dict[str, Any],
    levels: list[str],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import gc

    import torch

    from chess_engine_4.training.config import training_config_from_dict

    torch.set_float32_matmul_precision("high")
    config = training_config_from_dict(config_values)
    result: dict[str, Any] = {
        "d_model": config.model.d_model,
        "batch_size": config.run.batch_size,
        "warmup": warmup,
        "iterations": iterations,
        "device_name": torch.cuda.get_device_name(),
    }
    if "layer" in levels:
        result["layer"] = benchmark_dense_layer(
            d_model=config.model.d_model,
            batch_size=config.run.batch_size,
            precision=config.model.precision,
            warmup=warmup,
            iterations=iterations,
        )
        gc.collect()
        torch.cuda.empty_cache()
    if "step" in levels or "production" in levels:
        te_runner, custom_runner = _build_training_runners(config)
        if "step" in levels:
            synthetic_batch = _synthetic_batch(config.run.batch_size)
            result["step"] = _paired_measure(
                partial(te_runner.step, synthetic_batch),
                partial(custom_runner.step, synthetic_batch),
                warmup=warmup,
                iterations=iterations,
            )
        if "production" in levels:
            result["production"] = _benchmark_production(
                config,
                te_runner=te_runner,
                custom_runner=custom_runner,
                warmup=warmup,
                iterations=iterations,
            )
    return result


def training_benchmark_function(gpu: str):
    return app.function(
        image=benchmark_image,
        gpu=gpu,
        cpu=8,
        volumes={REMOTE_DATA_PATH: data_volume},
        timeout=2 * 60 * 60,
        name=f"training_benchmark_{gpu.lower().replace('-', '_')}",
    )(_benchmark_training)


class _TrainingRunner:
    def __init__(self, config: Any, *, custom: bool, state: dict[str, Any]) -> None:
        import torch

        from chess_engine_4.model import build_model
        from chess_engine_4.training.cli import _build_optimizer
        from chess_engine_4.training.packed_input import build_training_model

        self.config = config
        self.model = build_model(config.model).cuda()
        self.model.load_state_dict(state)
        if custom:
            self.model.enable_custom_kernels()
        self.optimizer = _build_optimizer(self.model, config=config)
        self.training_model = build_training_model(
            self.model,
            batch_size=config.run.batch_size,
            precision=config.model.precision,
        )
        self.training_model.train()
        torch.cuda.synchronize()

    def step(self, batch: tuple[Any, Any, Any]) -> None:
        from chess_engine_4.model.transformer_engine import autocast_context
        from chess_engine_4.training.cli import _clip_gradient_norm
        from chess_engine_4.training.losses import lczero_loss

        planes, policy, value = batch
        self.optimizer.zero_grad(set_to_none=True)
        with autocast_context(self.config.model.precision):
            output = self.training_model(planes)
            loss = lczero_loss(output, policy, value, weights=self.config.loss)
        loss.total.backward()
        _clip_gradient_norm(
            self.model,
            max_grad_norm=self.config.optimizer.max_grad_norm,
        )
        self.optimizer.step()


def _build_training_runners(config: Any) -> tuple[_TrainingRunner, _TrainingRunner]:
    import torch

    from chess_engine_4.model import build_model

    torch.manual_seed(config.run.seed)
    torch.cuda.manual_seed_all(config.run.seed)
    reference = build_model(config.model).cuda()
    state = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
    del reference
    te_runner = _TrainingRunner(config, custom=False, state=state)
    custom_runner = _TrainingRunner(config, custom=True, state=state)
    del state
    return te_runner, custom_runner


def _synthetic_batch(batch_size: int) -> tuple[Any, Any, Any]:
    import torch

    from chess_engine_4.data.leela import (
        COMPACT_POLICY_SIZE,
        HISTORY_PLANE_COUNT,
        VALUE_FIELDS,
        VALUE_TYPE_COUNT,
    )

    packed_planes = torch.randint(
        0,
        256,
        (batch_size, HISTORY_PLANE_COUNT, 8),
        device="cuda",
        dtype=torch.uint8,
    )
    plane_scalars = torch.zeros(batch_size, 8, device="cuda", dtype=torch.bfloat16)
    policy_indices = torch.arange(
        COMPACT_POLICY_SIZE,
        device="cuda",
        dtype=torch.int16,
    ).expand(batch_size, -1)
    policy_probs = torch.full(
        (batch_size, COMPACT_POLICY_SIZE),
        1.0 / COMPACT_POLICY_SIZE,
        device="cuda",
        dtype=torch.float16,
    )
    values = torch.zeros(
        batch_size,
        VALUE_TYPE_COUNT,
        VALUE_FIELDS,
        device="cuda",
        dtype=torch.float32,
    )
    values[:, :, 1] = 1.0
    return (packed_planes, plane_scalars), (policy_indices, policy_probs), values


def _benchmark_production(
    config: Any,
    *,
    te_runner: _TrainingRunner,
    custom_runner: _TrainingRunner,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch

    from chess_engine_4.data.leela import LeelaParquetDataset
    from chess_engine_4.training.input_pipeline import TrainingBatchPipeline

    dataset = LeelaParquetDataset(
        batch_size=config.run.batch_size,
        prefetch_per_thread=config.infra.dataloader_prefetch_per_thread,
        threads=config.infra.dataloader_threads,
    )
    iterator = iter(dataset)
    pipelines = {
        runner: TrainingBatchPipeline(
            kind=config.model.input_pipeline,
            device=torch.device("cuda"),
        )
        for runner in (te_runner, custom_runner)
    }

    def production_step(runner: _TrainingRunner) -> None:
        pipeline = pipelines[runner]
        batch = pipeline.transfer(pipeline.stage(next(iterator)))
        runner.step(batch)

    return _paired_measure(
        lambda: production_step(te_runner),
        lambda: production_step(custom_runner),
        warmup=warmup,
        iterations=iterations,
    )


def _paired_measure(
    run_te: Any,
    run_custom: Any,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch

    for index in range(warmup):
        pair = (run_te, run_custom) if index % 2 == 0 else (run_custom, run_te)
        for function in pair:
            function()
    torch.cuda.synchronize()

    gpu_samples: dict[str, list[float]] = {"te": [], "custom": []}
    wall_samples: dict[str, list[float]] = {"te": [], "custom": []}
    functions = {"te": run_te, "custom": run_custom}
    for index in range(iterations):
        order = ("te", "custom") if index % 2 == 0 else ("custom", "te")
        for implementation in order:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start_event.record()
            functions[implementation]()
            end_event.record()
            end_event.synchronize()
            wall_samples[implementation].append((time.perf_counter() - wall_start) * 1000.0)
            gpu_samples[implementation].append(start_event.elapsed_time(end_event))

    te_gpu = _summarize(gpu_samples["te"])
    custom_gpu = _summarize(gpu_samples["custom"])
    te_wall = _summarize(wall_samples["te"])
    custom_wall = _summarize(wall_samples["custom"])
    return {
        "te": {"gpu_ms": te_gpu, "wall_ms": te_wall},
        "custom": {"gpu_ms": custom_gpu, "wall_ms": custom_wall},
        "gpu_speedup_vs_te": te_gpu["median"] / custom_gpu["median"],
        "wall_speedup_vs_te": te_wall["median"] / custom_wall["median"],
    }


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p10": _percentile(ordered, 0.1),
        "p90": _percentile(ordered, 0.9),
        "stddev": statistics.pstdev(samples),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _print_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(
            f"d_model={result['d_model']} batch_size={result['batch_size']} "
            f"device={result['device_name']}"
        )
        print(f"{'level':<18} {'TE time':>12} {'custom time':>14} {'speedup':>10}")
        layer = result.get("layer")
        if layer is not None:
            _print_row("layer graph fwd", layer["te_ms"], layer["custom_ms"])
            _print_row(
                "layer graph bwd",
                layer["te_backward_ms"],
                layer["custom_backward_ms"],
            )
        for level in ("step", "production"):
            measurement = result.get(level)
            if measurement is None:
                continue
            _print_row(
                f"{level} GPU",
                measurement["te"]["gpu_ms"]["median"],
                measurement["custom"]["gpu_ms"]["median"],
            )
            _print_row(
                f"{level} wall",
                measurement["te"]["wall_ms"]["median"],
                measurement["custom"]["wall_ms"]["median"],
            )
        print("")


def _print_row(name: str, te_ms: float, custom_ms: float) -> None:
    print(f"{name:<18} {te_ms:>10.3f} ms {custom_ms:>12.3f} ms {te_ms / custom_ms:>9.3f}x")
