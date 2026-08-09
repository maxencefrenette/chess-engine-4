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
from chess_engine_4.model import KernelBackend, model_parameter_count
from chess_engine_4.training.config import (
    TrainingConfig,
    load_training_config,
    resolve_training_kernel,
    training_config_from_dict,
    validate_training_hardware,
    with_overrides,
)
from chess_engine_4.training.flops import measure_training_flops_per_sample

APP_NAME = "chess-engine-4-train"
DATA_VOLUME_NAME = "chess-engine-4-training-data"
ARTIFACT_VOLUME_NAME = "chess-engine-4-artifacts"
WANDB_SECRET_NAME = "chess-engine-4-wandb"
REMOTE_DATA_PATH = "/data/training_data"
REMOTE_PARQUET_DATA_PATH = f"{REMOTE_DATA_PATH}/parquet"
REMOTE_ARTIFACT_PATH = "/artifacts"
REMOTE_CHECKPOINT_PATH = Path(REMOTE_ARTIFACT_PATH) / "checkpoints"
DEFAULT_CONFIG_PATH = Path("configs/dense.py")
CHECKPOINT_EVERY_STEPS = 50_000

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
        extra_options="--no-build-isolation-package transformer-engine-torch",
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
    add_training_config_arguments(parser, include_steps=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()
    config = resolve_training_config(args)
    print_launch_summary(config)

    payload = {
        "config": asdict(config),
        "wandb": args.wandb,
        "wandb_project": os.environ.get("WANDB_PROJECT"),
        "wandb_entity": os.environ.get("WANDB_ENTITY"),
        "wandb_mode": os.environ.get("WANDB_MODE"),
        "wandb_name": args.wandb_name,
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
    print(
        f"modal_run_complete run={result['run_name']} "
        f"steps={result['steps']} "
        f"samples_seen={result['samples_seen']} "
        f"flops_seen={result['flops_seen']:.3e} "
        f"final_loss={result['final_loss']:.4f} "
        f"device={result['device']} "
        f"precision={result['precision']} "
        f"checkpoint_path={result['checkpoint_path']}"
    )


def _run_training_remote(payload: dict[str, Any]) -> dict[str, float | int | str]:
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


def add_training_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_steps: bool,
) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--training-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    if include_steps:
        parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--expansion-ratio", type=float, default=None)
    parser.add_argument("--history-length", type=int, choices=range(1, 9), default=None)
    parser.add_argument(
        "--activation",
        choices=("geglu", "gelu", "silu", "srelu", "swiglu"),
        default=None,
    )
    parser.add_argument("--lr", type=float, default=None)
    if include_steps:
        parser.add_argument("--max-grad-norm", type=float, default=None)
        parser.add_argument("--lr-warmup-steps", type=int, default=None)
        parser.add_argument("--lr-cooldown-frac", type=float, default=None)
    parser.add_argument(
        "--quantization-recipe",
        choices=("bf16", "mxfp8", "nvfp4"),
        default=None,
    )
    parser.add_argument("--dataloader-threads", type=int, default=None)
    parser.add_argument("--dataloader-prefetch-per-thread", type=int, default=None)
    parser.add_argument("--gpu", choices=TRAINING_GPUS, default=None)
    parser.add_argument(
        "--kernel-backend",
        choices=("te", "custom"),
        default=None,
    )


def resolve_training_config(args: argparse.Namespace) -> TrainingConfig:
    config = with_overrides(
        load_training_config(
            args.config,
            d_model=args.d_model,
            training_ratio=args.training_ratio,
            history_length=args.history_length,
        ),
        seed=args.seed,
        steps=getattr(args, "steps", None),
        batch_size=args.batch_size,
        depth=args.depth,
        expansion_ratio=args.expansion_ratio,
        activation=args.activation,
        lr=args.lr,
        max_grad_norm=getattr(args, "max_grad_norm", None),
        lr_warmup_steps=getattr(args, "lr_warmup_steps", None),
        lr_cooldown_frac=getattr(args, "lr_cooldown_frac", None),
        gpu=args.gpu,
        quantization_recipe=args.quantization_recipe,
        dataloader_threads=args.dataloader_threads,
        dataloader_prefetch_per_thread=args.dataloader_prefetch_per_thread,
        kernel_backend=args.kernel_backend,
    )
    if args.gpu in {"H100", "H200"} and args.kernel_backend is None:
        config = with_overrides(
            config,
            kernel_backend=_explicit_gpu_default_backend(
                gpu=args.gpu,
                model_kind=config.model.kind,
            ),
        )
    validate_training_hardware(config)
    return config


def _explicit_gpu_default_backend(*, gpu: TrainingGpu, model_kind: str) -> KernelBackend:
    """Select only Hopper backends established by end-to-end measurements."""

    if gpu == "H100" and model_kind == "moe64a2":
        return "custom"
    if gpu in {"H100", "H200"}:
        return "te"
    # Non-Hopper callers retain the backend from their canonical recipe.
    raise ValueError(f"No explicit GPU backend override is defined for {gpu}.")


def print_launch_summary(
    config: TrainingConfig,
    *,
    steps: int | None = None,
) -> None:
    run_steps = config.run.steps if steps is None else steps
    flops_per_sample = measure_training_flops_per_sample(
        config.model,
        batch_size=config.run.batch_size,
    )
    samples = config.run.batch_size * run_steps
    kernel_selection = resolve_training_kernel(config)
    print(
        "launch_summary "
        f"run={config.run.name} "
        f"model={config.model.kind}-d{config.model.d_model}x{config.model.depth} "
        f"expansion={config.model.expansion_ratio:g} "
        f"history={config.model.history_length} "
        f"activation={config.model.activation} "
        f"params={model_parameter_count(config.model):,} "
        f"training_ratio={config.run.training_ratio:g} "
        f"seed={config.run.seed} "
        f"batch_size={config.run.batch_size:,} "
        f"steps={run_steps:,} "
        f"samples={samples:,} "
        f"flops={flops_per_sample * samples:.3e} "
        f"lr={config.optimizer.lr:g} "
        f"precision={config.model.precision} "
        f"gpu={config.infra.gpu} "
        f"kernel_backend={config.model.kernel_backend} "
        f"kernel_variant={kernel_selection.variant} "
        f"input_pipeline={config.model.input_pipeline} "
        f"cpu_cores={config.infra.cpu_cores} "
        f"dataloader_threads={config.infra.dataloader_threads}"
    )


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
