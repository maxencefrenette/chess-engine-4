"""Modal training profiler."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chess_engine_4.kernels.config import KernelBackend
from chess_engine_4.modal_train import (
    ARTIFACT_VOLUME_NAME,
    REMOTE_ARTIFACT_PATH,
    add_training_config_arguments,
    app,
    print_launch_summary,
    resolve_training_config,
    training_function,
)

REMOTE_TRACE_PATH = Path(REMOTE_ARTIFACT_PATH) / "profiles" / "traces"


def profile_training() -> None:
    parser = argparse.ArgumentParser(description="Profile a training loop on Modal.")
    add_training_config_arguments(parser, include_steps=False)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Capture a PyTorch CPU/CUDA trace and download it to this JSON path.",
    )
    parser.add_argument("--json", action="store_true", help="Print only the JSON result.")
    args = parser.parse_args()

    if args.warmup_steps < 0:
        parser.error("warmup-steps must be non-negative.")
    if args.profile_steps <= 0:
        parser.error("profile-steps must be positive.")
    if args.trace_output is not None and args.trace_output.suffix != ".json":
        parser.error("trace-output must end in .json")
    config = resolve_training_config(args)
    if not args.json:
        print_launch_summary(
            config,
            steps=args.warmup_steps + args.profile_steps,
            kernel_backend=args.kernel_backend,
        )

    remote_trace_path = (
        REMOTE_TRACE_PATH / _trace_name(config.run.name, args.kernel_backend)
        if args.trace_output is not None
        else None
    )
    payload: dict[str, Any] = {
        "config": asdict(config),
        "wandb": False,
        "profile": {
            "warmup_steps": args.warmup_steps,
            "profile_steps": args.profile_steps,
        },
        "kernel_backend": args.kernel_backend,
    }
    if remote_trace_path is not None:
        payload["trace_path"] = str(remote_trace_path)

    profile_function = training_function(
        config.infra.cpu_cores,
        kernel_backend=args.kernel_backend,
    )
    with app.run():
        result = profile_function.remote(payload)

    if remote_trace_path is not None and args.trace_output is not None:
        _download_trace(remote_trace_path, args.trace_output, quiet=args.json)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_profile(result)
    if args.trace_output is not None:
        print(
            f"profile_trace={args.trace_output.resolve()}",
            file=sys.stderr if args.json else sys.stdout,
        )


def _print_profile(result: dict[str, Any]) -> None:
    print(
        f"profile_complete run={result['run_name']} gpu={result['device_name']} "
        f"steps={result['profile_steps']} warmup={result['warmup_steps']} "
        f"batch_size={result['batch_size']} "
        f"kernel_backend={result['kernel_backend']} "
        f"input_pipeline={result['input_pipeline']} "
        f"threads={result['dataloader_threads']} "
        f"prefetch_per_thread={result['dataloader_prefetch_per_thread']}"
    )
    print("")
    rows = [
        ("total wall", result["measured_wall_ms_per_step"]),
        ("loader + tensor wrapping wall", result["data_fetch_wall"]["mean_ms"]),
        ("pin memory wall", result["pin_memory_wall"]["mean_ms"]),
        ("H2D enqueue wall", result["h2d_enqueue_wall"]["mean_ms"]),
        ("exposed GPU idle gap", result["gpu_idle_gap_mean_ms"]),
        ("H2D copy on GPU stream", result["h2d_copy_gpu"]["mean_ms"]),
        ("train GPU kernels", result["train_gpu"]["mean_ms"]),
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


def _download_trace(remote_path: Path, local_path: Path, *, quiet: bool) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    volume_path = "/" + str(remote_path).removeprefix(REMOTE_ARTIFACT_PATH).lstrip("/")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            "--force",
            ARTIFACT_VOLUME_NAME,
            volume_path,
            str(local_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
    )


def _trace_name(run_name: str, kernel_backend: KernelBackend) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{run_name}-{kernel_backend}.json"
