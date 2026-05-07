"""Command-line entrypoints for local training workflows."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch

from chess_engine_4.data.leela import (
    DEFAULT_DATA_ENV_VAR,
    LeelaTarDataset,
)
from chess_engine_4.model import MlpChessNet
from chess_engine_4.training.config import TrainingConfig, load_training_config, with_overrides
from chess_engine_4.training.flops import (
    measure_training_flops_per_sample,
    steps_for_flops_target,
)
from chess_engine_4.training.losses import lczero_loss

_DATA_HELP = f"Leela tar path, directory, or glob. Defaults to ${DEFAULT_DATA_ENV_VAR}."
_DEFAULT_CONFIG_PATH = Path("configs/1e14.toml")


@dataclass(frozen=True, slots=True)
class TrainOptions:
    config: Path = _DEFAULT_CONFIG_PATH
    data: str | None = None
    batch_size: int | None = None
    flops_target: float | None = None
    d_model: int | None = None
    depth: int | None = None
    lr: float | None = None
    device: str | None = None
    wandb: bool = True
    wandb_name: str | None = None


def train() -> None:
    parser = argparse.ArgumentParser(description="Train the MLP-only chess network.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--flops-target", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    run_training(
        TrainOptions(
            config=args.config,
            data=args.data,
            batch_size=args.batch_size,
            flops_target=args.flops_target,
            d_model=args.d_model,
            depth=args.depth,
            lr=args.lr,
            device=args.device,
            wandb=args.wandb,
            wandb_name=args.wandb_name,
        )
    )


def run_training(options: TrainOptions) -> dict[str, float | int | str]:
    config = with_overrides(
        load_training_config(options.config),
        flops_target=options.flops_target,
        batch_size=options.batch_size,
        d_model=options.d_model,
        depth=options.depth,
        lr=options.lr,
        device=options.device,
    )
    _seed_everything(config.run.seed)

    with torch.device("meta"):
        flops_model = MlpChessNet(config.model)
    flops_per_sample = measure_training_flops_per_sample(
        flops_model,
        batch_size=config.data.batch_size,
    )
    steps = steps_for_flops_target(
        flops_target=config.run.flops_target,
        flops_per_sample=flops_per_sample,
        batch_size=config.data.batch_size,
    )

    dataset = LeelaTarDataset(
        options.data,
        batch_size=config.data.batch_size,
        max_records=config.data.max_records,
    )
    device = _resolve_device(config.run.device)
    model = MlpChessNet(config.model).to(device)
    optimizer = torch.optim.AdamW(
        _adamw_parameter_groups(model, weight_decay=config.optimizer.weight_decay),
        lr=config.optimizer.lr,
    )
    wandb_run = (
        _init_wandb(
            config,
            options.wandb_name,
            model,
            device,
            steps=steps,
            flops_per_sample=flops_per_sample,
        )
        if options.wandb
        else None
    )

    model.train()
    start = time.perf_counter()
    seen = 0
    last_metrics: dict[str, float | int] = {}
    for step, (planes, policy, value) in enumerate(islice(dataset, steps), start=1):
        planes = planes.to(device)
        policy = policy.to(device)
        value = value.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(planes)
        loss = lczero_loss(output, policy, value, weights=config.loss)
        loss.total.backward()
        grad_norm = _gradient_norm(model)
        optimizer.step()

        seen += planes.shape[0]
        elapsed = time.perf_counter() - start
        metrics = _training_metrics(
            output=output,
            policy_target=policy,
            values=value,
            loss=loss,
            grad_norm=grad_norm,
            samples_seen=seen,
            elapsed=elapsed,
            lr=config.optimizer.lr,
        )
        estimated_flops_seen = seen * flops_per_sample
        metrics["perf/measured_flops_per_sample"] = flops_per_sample
        metrics["perf/estimated_flops_seen"] = estimated_flops_seen
        metrics["perf/flops_target"] = config.run.flops_target
        should_log = step == 1 or step % config.run.log_every == 0 or step == steps
        if wandb_run is not None and should_log:
            wandb_run.log(metrics, step=step)
        if should_log:
            print(
                f"step={step} "
                f"loss={loss.total.item():.4f} "
                f"policy={loss.policy.item():.4f} "
                f"value={loss.value.item():.4f} "
                f"mlh={loss.moves_left.item():.4f} "
                f"grad_norm={grad_norm:.2f} "
                f"flops_seen={estimated_flops_seen:.3e} "
                f"samples_per_sec={seen / elapsed:.1f}"
            )
        last_metrics = metrics
    if wandb_run is not None:
        wandb_run.finish()
    return {
        "run_name": config.run.name,
        "steps": step if seen > 0 else 0,
        "samples_seen": seen,
        "final_loss": float(last_metrics.get("loss/total", 0.0)),
        "flops_target": config.run.flops_target,
        "estimated_flops_seen": int(last_metrics.get("perf/estimated_flops_seen", 0)),
        "measured_flops_per_sample": flops_per_sample,
        "device": str(device),
    }


def inspect_data() -> None:
    parser = argparse.ArgumentParser(description="Inspect Leela tar training records.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--records", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    dataset = LeelaTarDataset(args.data, batch_size=args.batch_size, max_records=args.records)
    seen = 0
    for planes, policy, value in dataset:
        batch_size = planes.shape[0]
        legal = policy >= 0
        legal_policy_sum = policy.masked_fill(~legal, 0).sum(dim=1)
        illegal_count = (~legal).sum(dim=1)
        seen += batch_size
        print(
            f"records={seen} "
            f"planes={tuple(planes.shape)} "
            f"policy={tuple(policy.shape)} "
            f"value={tuple(value.shape)} "
            f"legal_policy_sum=[{legal_policy_sum.min():.4f}, {legal_policy_sum.max():.4f}] "
            f"illegal_moves=[{illegal_count.min()}, {illegal_count.max()}] "
            f"plies_left=[{value[:, 0, 2].min():.1f}, {value[:, 0, 2].max():.1f}]"
        )


def sample_batch() -> None:
    parser = argparse.ArgumentParser(description="Load and print one Leela training batch.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    dataset = LeelaTarDataset(args.data, batch_size=args.batch_size, max_records=args.batch_size)
    planes, policy, value = next(iter(dataset))
    print(f"planes: {tuple(planes.shape)}")
    print(f"policy: {tuple(policy.shape)}")
    print(f"value: {tuple(value.shape)}")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _adamw_parameter_groups(
    model: torch.nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _init_wandb(
    config: TrainingConfig,
    run_name: str | None,
    model: torch.nn.Module,
    device: torch.device,
    *,
    steps: int,
    flops_per_sample: int,
) -> Any:
    import wandb

    wandb_config = {
        "run_name": config.run.name,
        "seed": config.run.seed,
        "flops_target": config.run.flops_target,
        "computed_steps": steps,
        "measured_flops_per_sample": flops_per_sample,
        "log_every": config.run.log_every,
        "batch_size": config.data.batch_size,
        "max_records": config.data.max_records,
        "device": str(device),
        "d_model": config.model.d_model,
        "depth": config.model.depth,
        "mlp_ratio": config.model.mlp_ratio,
        "rms_norm_eps": config.model.rms_norm_eps,
        "lr": config.optimizer.lr,
        "weight_decay": config.optimizer.weight_decay,
        "policy_loss_weight": config.loss.policy,
        "value_loss_weight": config.loss.value,
        "moves_left_loss_weight": config.loss.moves_left,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "non_embedding_parameter_count": sum(
            parameter.numel() for block in model.blocks for parameter in block.parameters()
        ),
    }
    return wandb.init(
        name=run_name or config.run.name,
        config=wandb_config,
    )


def _gradient_norm(model: torch.nn.Module) -> float:
    norms = [
        parameter.grad.detach().norm(2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return torch.linalg.vector_norm(torch.stack(norms), 2).item()


def _training_metrics(
    *,
    output: Any,
    policy_target: torch.Tensor,
    values: torch.Tensor,
    loss: Any,
    grad_norm: float,
    samples_seen: int,
    elapsed: float,
    lr: float,
) -> dict[str, float | int]:
    legal = policy_target >= 0
    legal_targets = policy_target.relu()
    policy_entropy = -(legal_targets * legal_targets.clamp_min(1e-30).log()).sum(dim=-1).mean()
    policy_logits = output.policy_logits.masked_fill(
        ~legal,
        torch.finfo(output.policy_logits.dtype).min,
    )
    policy_top1 = (policy_logits.argmax(dim=-1) == legal_targets.argmax(dim=-1)).float().mean()
    wdl_probs = torch.softmax(output.wdl_logits, dim=-1)
    q_pred = wdl_probs[:, 0] - wdl_probs[:, 2]
    result = values[:, 0]
    q_target = result[:, 0]
    q_mse = torch.square(q_pred - q_target).mean()
    moves_left_mae = torch.abs(output.moves_left - result[:, 2]).mean()
    samples_per_sec = samples_seen / elapsed if elapsed > 0 else 0.0
    return {
        "loss/total": loss.total.item(),
        "loss/policy": loss.policy.item(),
        "loss/value": loss.value.item(),
        "loss/moves_left": loss.moves_left.item(),
        "metrics/policy_entropy": policy_entropy.item(),
        "metrics/policy_top1": policy_top1.item(),
        "metrics/value_q_mse": q_mse.item(),
        "metrics/moves_left_mae": moves_left_mae.item(),
        "optim/lr": lr,
        "optim/grad_norm": grad_norm,
        "perf/samples_per_sec": samples_per_sec,
        "perf/samples_seen": samples_seen,
    }
