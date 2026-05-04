"""Command-line entrypoints for local training workflows."""

from __future__ import annotations

import argparse
import time
from itertools import islice
from typing import Any

import torch

from chess_engine_4.data.leela import (
    DEFAULT_DATA_ENV_VAR,
    LeelaTarDataset,
)
from chess_engine_4.model import MlpChessNet, MlpChessNetConfig
from chess_engine_4.training.losses import LossWeights, lczero_loss

_DATA_HELP = f"Leela tar path, directory, or glob. Defaults to ${DEFAULT_DATA_ENV_VAR}."


def train() -> None:
    parser = argparse.ArgumentParser(description="Train the MLP-only chess network.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--moves-left-loss-weight", type=float, default=1.0)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    dataset = LeelaTarDataset(args.data, batch_size=args.batch_size)
    device = _resolve_device(args.device)
    model = MlpChessNet(
        MlpChessNetConfig(
            d_model=args.d_model,
            depth=args.depth,
            mlp_ratio=args.mlp_ratio,
            rms_norm_eps=args.rms_norm_eps,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        _adamw_parameter_groups(model, weight_decay=args.weight_decay),
        lr=args.lr,
    )
    weights = LossWeights(
        policy=args.policy_loss_weight,
        value=args.value_loss_weight,
        moves_left=args.moves_left_loss_weight,
    )
    wandb_run = _init_wandb(args, model, device) if args.wandb else None

    model.train()
    start = time.perf_counter()
    seen = 0
    for step, (planes, policy, value) in enumerate(islice(dataset, args.steps), start=1):
        planes = planes.to(device)
        policy = policy.to(device)
        value = value.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(planes)
        loss = lczero_loss(output, policy, value, weights=weights)
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
            lr=args.lr,
        )
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        print(
            f"step={step} "
            f"loss={loss.total.item():.4f} "
            f"policy={loss.policy.item():.4f} "
            f"value={loss.value.item():.4f} "
            f"mlh={loss.moves_left.item():.4f} "
            f"grad_norm={grad_norm:.2f} "
            f"samples_per_sec={seen / elapsed:.1f}"
        )
    if wandb_run is not None:
        wandb_run.finish()


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
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> Any:
    import wandb

    config = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "device": str(device),
        "d_model": args.d_model,
        "depth": args.depth,
        "mlp_ratio": args.mlp_ratio,
        "rms_norm_eps": args.rms_norm_eps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "policy_loss_weight": args.policy_loss_weight,
        "value_loss_weight": args.value_loss_weight,
        "moves_left_loss_weight": args.moves_left_loss_weight,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return wandb.init(
        name=args.wandb_name,
        config=config,
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
