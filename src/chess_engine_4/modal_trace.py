"""Capture a production training timeline with PyTorch profiler on Modal."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

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
DEFAULT_LOCAL_TRACE_PATH = Path("profiles/traces")


def profile_training_trace() -> None:
    parser = argparse.ArgumentParser(
        description="Capture the production Modal training loop as a Chrome trace."
    )
    add_training_config_arguments(parser, include_steps=False)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("warmup-steps must be non-negative")
    if args.profile_steps <= 0:
        parser.error("profile-steps must be positive")

    config = resolve_training_config(args)
    print_launch_summary(config, steps=args.warmup_steps + args.profile_steps)
    stem = _trace_stem(config.run.name, experimental=args.experimental_dense_kernel)
    local_path = args.output or DEFAULT_LOCAL_TRACE_PATH / f"{stem}.json"
    if local_path.suffix != ".json":
        parser.error("output must end in .json")
    remote_path = REMOTE_TRACE_PATH / f"{stem}.json"

    payload = {
        "config": asdict(config),
        "wandb": False,
        "profile": {
            "warmup_steps": args.warmup_steps,
            "profile_steps": args.profile_steps,
        },
        "trace_path": str(remote_path),
        "experimental_dense_kernel": args.experimental_dense_kernel,
    }
    profile_function = training_function(
        config.infra.cpu_cores,
        experimental_dense_kernel=args.experimental_dense_kernel,
    )
    with app.run():
        profile_function.remote(payload)

    _download_trace(remote_path, local_path)
    print(f"profile_trace={local_path.resolve()}")


def _download_trace(remote_path: Path, local_path: Path) -> None:
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
    )


def _trace_stem(run_name: str, *, experimental: bool) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    implementation = "custom" if experimental else "te"
    return f"{timestamp}-{run_name}-{implementation}"
