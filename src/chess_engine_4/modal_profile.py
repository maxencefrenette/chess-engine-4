"""Modal training profiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chess_engine_4.modal_train import (
    REMOTE_CONFIG_PATH,
    _train,
    app,
)


def profile_training() -> None:
    parser = argparse.ArgumentParser(description="Profile a training loop on Modal.")
    parser.add_argument("--config", default=REMOTE_CONFIG_PATH, type=Path)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--quantization-recipe",
        choices=("bf16", "mxfp8", "nvfp4"),
        default=None,
    )
    parser.add_argument("--dataloader-threads", type=int, default=None)
    parser.add_argument("--dataloader-prefetch-per-thread", type=int, default=None)
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
        "d_model": args.d_model,
        "depth": args.depth,
        "lr": args.lr,
        "quantization_recipe": args.quantization_recipe,
        "dataloader_threads": args.dataloader_threads,
        "dataloader_prefetch_per_thread": args.dataloader_prefetch_per_thread,
        "wandb": False,
        "profile": {
            "warmup_steps": args.warmup_steps,
            "profile_steps": args.profile_steps,
        },
    }

    with app.run():
        result = _train.remote(payload)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_profile(result)


def _print_profile(result: dict[str, Any]) -> None:
    print(
        f"profile_complete config={result['config']} gpu={result['device_name']} "
        f"steps={result['profile_steps']} warmup={result['warmup_steps']} "
        f"batch_size={result['batch_size']} "
        f"threads={result['dataloader_threads']} "
        f"prefetch_per_thread={result['dataloader_prefetch_per_thread']}"
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
