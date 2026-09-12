"""Shared local and remote training launch helpers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv

from chess_engine_4.hardware import GPU_SPECS, TrainingGpu, identify_training_gpu
from chess_engine_4.model import model_parameter_count
from chess_engine_4.training.cli import TrainingResult, TrainOptions, run_training
from chess_engine_4.training.config import (
    TrainingConfig,
    load_training_config,
    resolve_training_kernel,
    validate_training_hardware,
    with_overrides,
)
from chess_engine_4.training.flops import measure_training_flops_per_sample

DEFAULT_CONFIG_PATH = Path("configs/dense.py")
DEFAULT_CHECKPOINT_PATH = Path("checkpoints")
CHECKPOINT_EVERY_STEPS = 50_000


def train_local() -> None:
    """Train directly on the CUDA GPU and Parquet data available on this host."""

    load_dotenv(dotenv_path=Path.cwd() / ".env")
    parser = argparse.ArgumentParser(description="Train a chess neural network locally.")
    add_training_config_arguments(
        parser,
        include_steps=True,
        gpu_choices=tuple(GPU_SPECS),
    )
    parser.add_argument("--data", default=None, help="Parquet path, directory, or glob.")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY_STEPS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sampling-rate", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 < args.sampling_rate <= 1.0:
        parser.error("--sampling-rate must be greater than 0 and at most 1")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    detected_gpu = detect_local_training_gpu()
    config = resolve_training_config(args, default_gpu=detected_gpu)
    print_launch_summary(config, sampling_rate=args.sampling_rate)
    if args.dry_run:
        return

    result = run_training(
        TrainOptions(
            config=config,
            data=args.data,
            sampling_rate=args.sampling_rate,
            wandb=args.wandb,
            wandb_name=args.wandb_name,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
        )
    )
    print_training_result("local", result)


def detect_local_training_gpu() -> TrainingGpu:
    """Identify the local CUDA device using the shared hardware catalog."""

    if not torch.cuda.is_available():
        raise RuntimeError("local training requires a CUDA GPU visible to PyTorch")
    device = torch.device("cuda")
    return identify_training_gpu(
        device_name=torch.cuda.get_device_name(device),
        capability=torch.cuda.get_device_capability(device),
    )


def add_training_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_steps: bool,
    gpu_choices: Sequence[str],
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
    parser.add_argument("--optimizer", choices=("adamw", "adamh"), default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
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
    parser.add_argument("--gpu", choices=gpu_choices, default=None)
    parser.add_argument("--kernel-backend", choices=("te", "custom"), default=None)


def resolve_training_config(
    args: argparse.Namespace,
    *,
    default_gpu: TrainingGpu | None = None,
) -> TrainingConfig:
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
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=getattr(args, "max_grad_norm", None),
        lr_warmup_steps=getattr(args, "lr_warmup_steps", None),
        lr_cooldown_frac=getattr(args, "lr_cooldown_frac", None),
        gpu=args.gpu if args.gpu is not None else default_gpu,
        quantization_recipe=args.quantization_recipe,
        dataloader_threads=args.dataloader_threads,
        dataloader_prefetch_per_thread=args.dataloader_prefetch_per_thread,
        kernel_backend=args.kernel_backend,
    )
    validate_training_hardware(config)
    return config


def print_launch_summary(
    config: TrainingConfig,
    *,
    steps: int | None = None,
    sampling_rate: float = 1.0,
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
        f"optimizer={config.optimizer.kind} "
        f"lr={config.optimizer.lr:g} "
        f"weight_decay={config.optimizer.weight_decay} "
        f"precision={config.model.precision} "
        f"gpu={config.infra.gpu} "
        f"kernel_backend={config.model.kernel_backend} "
        f"kernel_variant={kernel_selection.variant} "
        f"input_pipeline={config.model.input_pipeline} "
        f"cpu_cores={config.infra.cpu_cores} "
        f"dataloader_threads={config.infra.dataloader_threads}"
        f" sampling_rate={sampling_rate:g}"
    )


def print_training_result(prefix: str, result: TrainingResult | dict[str, Any]) -> None:
    print(
        f"{prefix}_run_complete run={result['run_name']} "
        f"steps={result['steps']} "
        f"samples_seen={result['samples_seen']} "
        f"flops_seen={result['flops_seen']:.3e} "
        f"final_loss={result['final_loss']:.4f} "
        f"device={result['device']} "
        f"precision={result['precision']} "
        f"checkpoint_path={result['checkpoint_path']}"
    )
