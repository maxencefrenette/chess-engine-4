"""Modal training profiler."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import modal

from chess_engine_4.modal_train import (
    GPU_CHOICES,
    REMOTE_CONFIG_PATH,
    REMOTE_DATA_PATH,
    app,
    data_volume,
    image,
)
from chess_engine_4.training.config import load_training_config


def profile_training() -> None:
    parser = argparse.ArgumentParser(description="Profile a training loop on Modal.")
    parser.add_argument("--config", default=REMOTE_CONFIG_PATH, type=Path)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--compute-budget", type=float, default=None)
    parser.add_argument("--step-penalty-k", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--router-aux", type=float, default=None)
    parser.add_argument("--gpu", default=None, choices=sorted(GPU_CHOICES))
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument("--json", action="store_true", help="Print only the JSON result.")
    args = parser.parse_args()

    if args.warmup_steps < 0:
        parser.error("warmup-steps must be non-negative.")
    if args.profile_steps <= 0:
        parser.error("profile-steps must be positive.")

    payload = {
        "config": str(args.config),
        "batch_size": args.batch_size,
        "compute_budget": args.compute_budget,
        "step_penalty_k": args.step_penalty_k,
        "d_model": args.d_model,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "router_aux": args.router_aux,
        "num_workers": args.num_workers,
        "warmup_steps": args.warmup_steps,
        "profile_steps": args.profile_steps,
    }

    config = load_training_config(args.config)
    gpu = args.gpu or config.infra.gpu_type
    if gpu not in GPU_CHOICES:
        parser.error(f"config infra.gpu_type must be one of: {', '.join(sorted(GPU_CHOICES))}")

    profile_function = _profile_function_for_gpu(gpu)
    with app.run():
        result = profile_function.remote(payload)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_profile(result)


def _run_profile_remote(payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from chess_engine_4.data.leela import LeelaTarDataset
    from chess_engine_4.model import build_model
    from chess_engine_4.training.cli import (
        _MATMUL_PRECISION,
        _autocast_context,
        _build_optimizer,
        _compile_model_for_training,
        _move_batch_to_device,
        _seed_everything,
        _theoretical_tflops,
        _training_precision,
    )
    from chess_engine_4.training.config import load_training_config, with_overrides
    from chess_engine_4.training.flops import measure_training_flops_per_sample
    from chess_engine_4.training.losses import lczero_loss
    from chess_engine_4.training.packed_input import PackedInputTrainingModel

    config = with_overrides(
        load_training_config(Path(payload["config"])),
        compute_budget=payload["compute_budget"],
        step_penalty_k=payload["step_penalty_k"],
        batch_size=payload["batch_size"],
        d_model=payload["d_model"],
        depth=payload["depth"],
        num_heads=payload["num_heads"],
        lr=payload["lr"],
        router_aux=payload["router_aux"],
    )
    device = torch.device("cuda")
    precision = _training_precision(device)
    torch.set_float32_matmul_precision(_MATMUL_PRECISION)
    _seed_everything(config.run.seed)

    with torch.device("meta"):
        flops_model = build_model(config.model)
    flops_per_sample = measure_training_flops_per_sample(
        flops_model,
        batch_size=config.data.batch_size,
    )

    num_workers = payload["num_workers"] if payload["num_workers"] is not None else 0
    dataset = LeelaTarDataset(
        REMOTE_DATA_PATH,
        batch_size=config.data.batch_size,
        max_records=config.data.max_records,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        **({"persistent_workers": True, "prefetch_factor": 2} if num_workers > 0 else {}),
    )
    iterator = iter(dataloader)

    model = build_model(config.model).to(device)
    training_model = _compile_model_for_training(
        PackedInputTrainingModel(model).to(device),
        device=device,
    )
    optimizer = _build_optimizer(model, config=config, device=device)
    training_model.train()

    warmup_steps = int(payload["warmup_steps"])
    profile_steps = int(payload["profile_steps"])
    total_steps = warmup_steps + profile_steps
    records: list[dict[str, Any]] = []
    final_loss = 0.0

    overall_start = time.perf_counter()
    measured_wall_start = None
    for step in range(1, total_steps + 1):
        if step == warmup_steps + 1:
            measured_wall_start = time.perf_counter()

        fetch_start = time.perf_counter()
        batch = next(iterator)
        fetch_end = time.perf_counter()

        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        train_start = torch.cuda.Event(enable_timing=True)
        train_end = torch.cuda.Event(enable_timing=True)

        enqueue_start = time.perf_counter()
        copy_start.record()
        planes, policy, value = _move_batch_to_device(batch, device=device)
        copy_end.record()

        train_start.record()
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, precision=precision):
            output = training_model(planes)
            loss = lczero_loss(output, policy, value, weights=config.loss)
        loss.total.backward()
        optimizer.step()
        train_end.record()
        enqueue_end = time.perf_counter()
        final_loss = float(loss.task.detach().item())

        records.append(
            {
                "step": step,
                "fetch_wall_ms": (fetch_end - fetch_start) * 1000.0,
                "enqueue_wall_ms": (enqueue_end - enqueue_start) * 1000.0,
                "copy_start": copy_start,
                "copy_end": copy_end,
                "train_start": train_start,
                "train_end": train_end,
            }
        )

    torch.cuda.synchronize(device)
    measured_wall_end = time.perf_counter()
    overall_end = measured_wall_end

    measured = records[warmup_steps:]
    if warmup_steps > 0:
        previous_for_gap = records[warmup_steps - 1 : -1]
        current_for_gap = measured
    else:
        previous_for_gap = measured[:-1]
        current_for_gap = measured[1:]

    copy_ms = [record["copy_start"].elapsed_time(record["copy_end"]) for record in measured]
    train_ms = [record["train_start"].elapsed_time(record["train_end"]) for record in measured]
    stream_step_ms = [
        record["copy_start"].elapsed_time(record["train_end"]) for record in measured
    ]
    idle_gap_ms = [
        previous["train_end"].elapsed_time(current["copy_start"])
        for previous, current in zip(previous_for_gap, current_for_gap, strict=True)
    ]
    fetch_ms = [record["fetch_wall_ms"] for record in measured]
    enqueue_ms = [record["enqueue_wall_ms"] for record in measured]

    measured_wall_ms_per_step = (
        (measured_wall_end - measured_wall_start) * 1000.0 / profile_steps
        if measured_wall_start is not None
        else 0.0
    )
    theoretical_tflops = _theoretical_tflops(device, precision=precision)
    train_gpu_mean_ms = statistics.fmean(train_ms)
    achieved_train_only_tflops = (
        config.data.batch_size * flops_per_sample / (train_gpu_mean_ms / 1000.0) / 1e12
    )
    gpu_work_mean_ms = statistics.fmean(copy_ms) + train_gpu_mean_ms
    gpu_idle_gap_mean_ms = statistics.fmean(idle_gap_ms) if idle_gap_ms else 0.0

    return {
        "config": str(payload["config"]),
        "run_name": config.run.name,
        "model_kind": config.model.kind,
        "device_name": torch.cuda.get_device_name(device),
        "precision": precision,
        "batch_size": config.data.batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "warmup_steps": warmup_steps,
        "profile_steps": profile_steps,
        "overall_wall_sec": overall_end - overall_start,
        "measured_wall_ms_per_step": measured_wall_ms_per_step,
        "final_loss": final_loss,
        "flops_per_sample": flops_per_sample,
        "theoretical_tflops": theoretical_tflops,
        "achieved_train_only_tflops": achieved_train_only_tflops,
        "train_only_mfu": (
            achieved_train_only_tflops / theoretical_tflops if theoretical_tflops else None
        ),
        "end_to_end_mfu": (
            config.data.batch_size
            * flops_per_sample
            / (measured_wall_ms_per_step / 1000.0)
            / 1e12
            / theoretical_tflops
            if theoretical_tflops and measured_wall_ms_per_step > 0
            else None
        ),
        "gpu_work_mean_ms": gpu_work_mean_ms,
        "gpu_idle_gap_mean_ms": gpu_idle_gap_mean_ms,
        "gpu_idle_fraction_of_step": (
            gpu_idle_gap_mean_ms / measured_wall_ms_per_step
            if measured_wall_ms_per_step > 0
            else 0.0
        ),
        "data_fetch_wall": _summarize(fetch_ms),
        "enqueue_wall": _summarize(enqueue_ms),
        "h2d_copy_gpu": _summarize(copy_ms),
        "train_gpu": _summarize(train_ms),
        "gpu_idle_gap": _summarize(idle_gap_ms),
        "stream_copy_plus_train": _summarize(stream_step_ms),
    }


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "max_ms": 0.0}
    sorted_values = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": sorted_values[int(0.9 * (len(sorted_values) - 1))],
        "max_ms": max(values),
    }


def _print_profile(result: dict[str, Any]) -> None:
    print(
        f"profile_complete config={result['config']} gpu={result['device_name']} "
        f"steps={result['profile_steps']} warmup={result['warmup_steps']} "
        f"batch_size={result['batch_size']} num_workers={result['num_workers']}"
    )
    print("")
    rows = [
        ("total wall", result["measured_wall_ms_per_step"]),
        ("CPU/dataloader fetch wall", result["data_fetch_wall"]["mean_ms"]),
        ("Python enqueue wall", result["enqueue_wall"]["mean_ms"]),
        ("exposed GPU idle gap", result["gpu_idle_gap_mean_ms"]),
        ("H2D copy on GPU stream", result["h2d_copy_gpu"]["mean_ms"]),
        ("train GPU kernels", result["train_gpu"]["mean_ms"]),
        ("copy + train GPU work", result["gpu_work_mean_ms"]),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value:8.2f} ms/step")
    print("")
    train_mfu = result["train_only_mfu"]
    end_to_end_mfu = result["end_to_end_mfu"]
    if train_mfu is not None:
        print(f"train-only MFU     {train_mfu:.3f}")
    if end_to_end_mfu is not None:
        print(f"end-to-end MFU     {end_to_end_mfu:.3f}")
    print(f"final loss         {result['final_loss']:.4f}")
    print("")
    print(json.dumps(result, indent=2, sort_keys=True))


@app.function(
    image=image,
    gpu="any",
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=24 * 60 * 60,
)
def _profile_any(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_remote(payload)


@app.function(
    image=image,
    gpu="T4",
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=24 * 60 * 60,
)
def _profile_t4(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_remote(payload)


@app.function(
    image=image,
    gpu="L4",
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=24 * 60 * 60,
)
def _profile_l4(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_remote(payload)


@app.function(
    image=image,
    gpu="A10G",
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=24 * 60 * 60,
)
def _profile_a10g(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_remote(payload)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=24 * 60 * 60,
)
def _profile_a100_40gb(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_remote(payload)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=24 * 60 * 60,
)
def _profile_a100_80gb(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_remote(payload)


@app.function(
    image=image,
    gpu="L40S",
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=24 * 60 * 60,
)
def _profile_l40s(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_remote(payload)


@app.function(
    image=image,
    gpu="H100",
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=24 * 60 * 60,
)
def _profile_h100(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_remote(payload)


@app.function(
    image=image,
    gpu="H200",
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=24 * 60 * 60,
)
def _profile_h200(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_remote(payload)


@app.function(
    image=image,
    gpu="B200",
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=24 * 60 * 60,
)
def _profile_b200(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_profile_remote(payload)


def _profile_function_for_gpu(gpu: str) -> modal.Function:
    return {
        "any": _profile_any,
        "t4": _profile_t4,
        "l4": _profile_l4,
        "a10g": _profile_a10g,
        "a100-40gb": _profile_a100_40gb,
        "a100-80gb": _profile_a100_80gb,
        "l40s": _profile_l40s,
        "h100": _profile_h100,
        "h200": _profile_h200,
        "b200": _profile_b200,
    }[gpu]
