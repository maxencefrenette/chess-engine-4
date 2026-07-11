"""Modal training entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import modal
from dotenv import load_dotenv

APP_NAME = "chess-engine-4-train"
DATA_VOLUME_NAME = "chess-engine-4-training-data"
ARTIFACT_VOLUME_NAME = "chess-engine-4-artifacts"
WANDB_SECRET_NAME = "chess-engine-4-wandb"
REMOTE_DATA_PATH = "/data/training_data"
REMOTE_ARTIFACT_PATH = "/artifacts"
REMOTE_CHECKPOINT_PATH = Path(REMOTE_ARTIFACT_PATH) / "checkpoints"
REMOTE_CONFIG_PATH = Path("configs/mlp/1e18.toml")

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands(
        "curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal",
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
    .env({"CHESS_ENGINE_4_DATA_PATH": REMOTE_DATA_PATH})
    .workdir("/root")
    .add_local_dir("crates", remote_path="/root/crates", copy=True)
    .run_commands(
        "PATH=/root/.cargo/bin:$PATH uv run maturin develop "
        "--manifest-path /root/crates/leela_loader/Cargo.toml --release",
        "uv run python -c 'import chess_engine_4_native'",
    )
    .add_local_python_source("chess_engine_4")
    .add_local_dir("configs", remote_path="/root/configs")
)


def train_modal() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env")

    parser = argparse.ArgumentParser(description="Train a chess neural network on Modal.")
    parser.add_argument("--config", default=REMOTE_CONFIG_PATH, type=Path)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--compute-budget", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument("--lr-cooldown-frac", type=float, default=None)
    parser.add_argument("--router-aux", type=float, default=None)
    parser.add_argument(
        "--quantization-recipe",
        choices=("bf16", "mxfp8", "nvfp4"),
        default=None,
    )
    parser.add_argument("--dataloader-threads", type=int, default=None)
    parser.add_argument("--dataloader-prefetch-per-thread", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=REMOTE_CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    args = parser.parse_args()

    payload = {
        "config": str(args.config),
        "batch_size": args.batch_size,
        "compute_budget": args.compute_budget,
        "d_model": args.d_model,
        "depth": args.depth,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_cooldown_frac": args.lr_cooldown_frac,
        "router_aux": args.router_aux,
        "quantization_recipe": args.quantization_recipe,
        "dataloader_threads": args.dataloader_threads,
        "dataloader_prefetch_per_thread": args.dataloader_prefetch_per_thread,
        "max_steps": args.max_steps,
        "wandb": args.wandb,
        "wandb_project": os.environ.get("WANDB_PROJECT"),
        "wandb_entity": os.environ.get("WANDB_ENTITY"),
        "wandb_mode": os.environ.get("WANDB_MODE"),
        "wandb_name": args.wandb_name,
        "checkpoint_dir": str(args.checkpoint_dir) if args.save_checkpoints else None,
        "checkpoint_every": args.checkpoint_every,
    }

    with app.run():
        result = _train.remote(payload)
    print(
        f"modal_run_complete run={result['run_name']} "
        f"steps={result['steps']} "
        f"samples_seen={result['samples_seen']} "
        f"flops_seen={result['flops_seen']:.3e} "
        f"compute_seen={result['compute_seen']:.3e} "
        f"step_penalty_k={result['step_penalty_k']:.3f} "
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
            config=Path(payload["config"]),
            data=REMOTE_DATA_PATH,
            batch_size=payload.get("batch_size"),
            compute_budget=payload.get("compute_budget"),
            d_model=payload.get("d_model"),
            depth=payload.get("depth"),
            lr=payload.get("lr"),
            max_grad_norm=payload.get("max_grad_norm"),
            lr_warmup_steps=payload.get("lr_warmup_steps"),
            lr_cooldown_frac=payload.get("lr_cooldown_frac"),
            router_aux=payload.get("router_aux"),
            quantization_recipe=payload.get("quantization_recipe"),
            dataloader_threads=payload.get("dataloader_threads"),
            dataloader_prefetch_per_thread=payload.get(
                "dataloader_prefetch_per_thread"
            ),
            max_steps=payload.get("max_steps"),
            wandb=payload.get("wandb", True),
            wandb_name=payload.get("wandb_name"),
            checkpoint_dir=(
                Path(payload["checkpoint_dir"]) if payload.get("checkpoint_dir") else None
            ),
            checkpoint_every=payload.get("checkpoint_every"),
            profile=(TrainingProfileConfig(**profile) if profile is not None else None),
        )
    )
    if result["checkpoint_path"]:
        artifact_volume.commit()
    return result


@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)
