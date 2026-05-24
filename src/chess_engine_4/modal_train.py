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
REMOTE_CONFIG_PATH = Path("configs/mlp/1e15.toml")

GPU_CHOICES = {
    "any": "any",
    "t4": "T4",
    "l4": "L4",
    "a10g": "A10G",
    "a100-40gb": "A100-40GB",
    "a100-80gb": "A100-80GB",
    "l40s": "L40S",
    "h100": "H100",
    "h200": "H200",
    "b200": "B200",
}

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_sync()
    .env({"CHESS_ENGINE_4_DATA_PATH": REMOTE_DATA_PATH})
    .workdir("/root")
    .add_local_python_source("chess_engine_4")
    .add_local_dir("configs", remote_path="/root/configs")
)


def train_modal() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env")

    parser = argparse.ArgumentParser(description="Train a chess neural network on Modal.")
    parser.add_argument("--config", default=REMOTE_CONFIG_PATH, type=Path)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--compute-budget", type=float, default=None)
    parser.add_argument("--step-penalty-k", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu", default="t4", choices=sorted(GPU_CHOICES))
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=REMOTE_CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    args = parser.parse_args()

    payload = {
        "config": str(args.config),
        "batch_size": args.batch_size,
        "compute_budget": args.compute_budget,
        "step_penalty_k": args.step_penalty_k,
        "d_model": args.d_model,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
        "wandb_name": args.wandb_name,
        "checkpoint_dir": str(args.checkpoint_dir) if args.save_checkpoints else None,
        "checkpoint_every": args.checkpoint_every,
    }

    train_function = _remote_function_for_gpu(args.gpu)
    with app.run():
        result = train_function.remote(payload)
    print(
        f"modal_run_complete run={result['run_name']} "
        f"steps={result['steps']} "
        f"samples_seen={result['samples_seen']} "
        f"flops_seen={result['flops_seen']:.3e} "
        f"compute_seen={result['compute_seen']:.3e} "
        f"step_penalty_k={result['step_penalty_k']:.3f} "
        f"final_loss={result['final_loss']:.4f} "
        f"device={result['device']} "
        f"checkpoint_path={result['checkpoint_path']}"
    )


def _run_training_remote(payload: dict[str, Any]) -> dict[str, float | int | str]:
    import os

    from chess_engine_4.training.cli import TrainOptions, run_training

    for env_key, payload_key in (
        ("WANDB_PROJECT", "wandb_project"),
        ("WANDB_ENTITY", "wandb_entity"),
        ("WANDB_MODE", "wandb_mode"),
    ):
        value = payload.get(payload_key)
        if value:
            os.environ[env_key] = value

    result = run_training(
        TrainOptions(
            config=Path(payload["config"]),
            data=REMOTE_DATA_PATH,
            batch_size=payload["batch_size"],
            compute_budget=payload["compute_budget"],
            step_penalty_k=payload["step_penalty_k"],
            d_model=payload["d_model"],
            depth=payload["depth"],
            num_heads=payload["num_heads"],
            lr=payload["lr"],
            device="cuda",
            wandb=payload["wandb"],
            wandb_name=payload["wandb_name"],
            checkpoint_dir=(
                Path(payload["checkpoint_dir"]) if payload.get("checkpoint_dir") else None
            ),
            checkpoint_every=payload["checkpoint_every"],
        )
    )
    if result["checkpoint_path"]:
        artifact_volume.commit()
    return result


@app.function(
    image=image,
    gpu="any",
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train_any(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)


@app.function(
    image=image,
    gpu="T4",
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train_t4(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)


@app.function(
    image=image,
    gpu="L4",
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train_l4(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)


@app.function(
    image=image,
    gpu="A10G",
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train_a10g(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train_a100_40gb(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train_a100_80gb(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)


@app.function(
    image=image,
    gpu="L40S",
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train_l40s(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)


@app.function(
    image=image,
    gpu="H100",
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train_h100(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)


@app.function(
    image=image,
    gpu="H200",
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train_h200(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)


@app.function(
    image=image,
    gpu="B200",
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    secrets=[wandb_secret],
    timeout=24 * 60 * 60,
)
def _train_b200(payload: dict[str, Any]) -> dict[str, float | int | str]:
    return _run_training_remote(payload)


def _remote_function_for_gpu(gpu: str) -> modal.Function:
    return {
        "any": _train_any,
        "t4": _train_t4,
        "l4": _train_l4,
        "a10g": _train_a10g,
        "a100-40gb": _train_a100_40gb,
        "a100-80gb": _train_a100_80gb,
        "l40s": _train_l40s,
        "h100": _train_h100,
        "h200": _train_h200,
        "b200": _train_b200,
    }[gpu]
