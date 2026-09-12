"""Modal training entrypoint."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal
from dotenv import load_dotenv

from chess_engine_4.hardware import TRAINING_GPUS, TrainingGpu, modal_gpu_identifier
from chess_engine_4.model import KernelBackend
from chess_engine_4.training.config import training_config_from_dict
from chess_engine_4.training.launch import (
    CHECKPOINT_EVERY_STEPS,
    add_training_config_arguments,
    print_launch_summary,
    print_training_result,
    resolve_training_config,
)

APP_NAME = "chess-engine-4-train"
DATA_VOLUME_NAME = "chess-engine-4-training-data"
ARTIFACT_VOLUME_NAME = "chess-engine-4-artifacts"
WANDB_SECRET_NAME = "chess-engine-4-wandb"
REMOTE_DATA_PATH = "/data/training_data"
REMOTE_PARQUET_DATA_PATH = f"{REMOTE_DATA_PATH}/parquet"
REMOTE_ARTIFACT_PATH = "/artifacts"
REMOTE_CHECKPOINT_PATH = Path(REMOTE_ARTIFACT_PATH) / "checkpoints"
type TrainingResultValue = float | int | str | None | list[list[int]] | dict[str, float]

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)

base_image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands(
        "curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain 1.95.0",
        "PATH=/root/.cargo/bin:$PATH rustc --version",
    )
    .uv_sync(
        extra_options="--extra cuda --no-build-isolation-package transformer-engine-torch",
        env={
            "NVTE_BUILD_USE_NVIDIA_WHEELS": "1",
            "NVTE_FRAMEWORK": "pytorch",
            "NVTE_WITH_NCCL_EP": "0",
            "PATH": (
                "/.uv/.venv/lib/python3.14/site-packages/nvidia/cu13/bin:"
                "/usr/local/bin:/usr/bin:/bin"
            ),
        },
    )
    .run_commands("uv pip install --no-deps nvidia-cublas==13.6.0.2")
    .run_commands(
        "find /.uv/.venv/lib/python3.14/site-packages/nvidia -type d -name lib "
        "> /etc/ld.so.conf.d/nvidia-python.conf && ldconfig"
    )
    .env(
        {
            "CHESS_ENGINE_4_DATA_PATH": REMOTE_PARQUET_DATA_PATH,
            "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM": "1",
        }
    )
    .workdir("/root")
    .add_local_dir("crates", remote_path="/root/crates", copy=True)
    .run_commands(
        "PATH=/root/.cargo/bin:$PATH uv run maturin develop "
        "--manifest-path /root/crates/leela_loader/Cargo.toml --release",
        "uv run python -c 'import chess_engine_4_native'",
    )
)
image = base_image.add_local_python_source("chess_engine_4")


def train_modal() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env")

    parser = argparse.ArgumentParser(description="Train a chess neural network on Modal.")
    add_training_config_arguments(
        parser,
        include_steps=True,
        gpu_choices=TRAINING_GPUS,
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()
    if not 0.0 < args.sampling_rate <= 1.0:
        parser.error("--sampling-rate must be greater than 0 and at most 1")
    config = resolve_training_config(args)
    print_launch_summary(config, sampling_rate=args.sampling_rate)
    if args.dry_run:
        return

    payload = {
        "config": asdict(config),
        "wandb": args.wandb,
        "wandb_project": os.environ.get("WANDB_PROJECT"),
        "wandb_entity": os.environ.get("WANDB_ENTITY"),
        "wandb_mode": os.environ.get("WANDB_MODE"),
        "wandb_name": args.wandb_name,
        "sampling_rate": args.sampling_rate,
        "checkpoint_dir": str(REMOTE_CHECKPOINT_PATH),
        "checkpoint_every": CHECKPOINT_EVERY_STEPS,
    }

    train_function = training_function(
        config.infra.cpu_cores,
        gpu=config.infra.gpu,
        kernel_backend=config.model.kernel_backend,
    )
    with app.run():
        result = train_function.remote(payload)
    print_training_result("modal", result)


def _run_training_remote(payload: dict[str, Any]) -> dict[str, TrainingResultValue]:
    import os

    from chess_engine_4.training.cli import TrainOptions, run_training
    from chess_engine_4.training.profiling import TrainingProfileConfig

    for env_key, payload_key in (
        ("WANDB_PROJECT", "wandb_project"),
        ("WANDB_ENTITY", "wandb_entity"),
        ("WANDB_MODE", "wandb_mode"),
    ):
        value = payload.get(payload_key)
        if value:
            os.environ[env_key] = value

    profile = payload.get("profile")
    result = run_training(
        TrainOptions(
            config=training_config_from_dict(payload["config"]),
            data=REMOTE_PARQUET_DATA_PATH,
            sampling_rate=float(payload.get("sampling_rate", 1.0)),
            wandb=payload.get("wandb", True),
            wandb_name=payload.get("wandb_name"),
            checkpoint_dir=(
                Path(payload["checkpoint_dir"]) if payload.get("checkpoint_dir") else None
            ),
            checkpoint_every=payload.get("checkpoint_every"),
            checkpoint_commit=artifact_volume.commit,
            profile=(TrainingProfileConfig(**profile) if profile is not None else None),
            trace_path=(Path(payload["trace_path"]) if payload.get("trace_path") else None),
        )
    )
    if payload.get("trace_path"):
        artifact_volume.commit()
    return result


def training_function(
    cpu_cores: int,
    *,
    gpu: TrainingGpu,
    kernel_backend: KernelBackend = "te",
) -> modal.Function:
    selected_image = image
    function_name = f"train_{gpu.lower().replace('-', '_')}_cpu_{cpu_cores}"
    if kernel_backend == "custom":
        from chess_engine_4.kernels.modal import with_cuda_kernels

        selected_image = with_cuda_kernels(base_image)
        function_name += "_custom_kernels"
    return app.function(
        image=selected_image,
        gpu=modal_gpu_identifier(gpu),
        cpu=cpu_cores,
        volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
        secrets=[wandb_secret],
        timeout=24 * 60 * 60,
        name=function_name,
    )(_run_training_remote)
