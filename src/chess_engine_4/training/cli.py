"""Command-line entrypoints for local training workflows."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch

from chess_engine_4.data.leela import (
    DEFAULT_DATA_ENV_VAR,
    LeelaTarDataset,
)
from chess_engine_4.model import build_model
from chess_engine_4.training.config import TrainingConfig, load_training_config, with_overrides
from chess_engine_4.training.flops import (
    measure_training_flops_per_sample,
    step_adjusted_compute,
    steps_for_compute_budget,
)
from chess_engine_4.training.losses import lczero_loss

_DATA_HELP = f"Leela tar path, directory, or glob. Defaults to ${DEFAULT_DATA_ENV_VAR}."
_DEFAULT_CONFIG_PATH = Path("configs/1e15.toml")


@dataclass(frozen=True, slots=True)
class TrainOptions:
    config: Path = _DEFAULT_CONFIG_PATH
    data: str | None = None
    batch_size: int | None = None
    compute_budget: float | None = None
    step_penalty_k: float | None = None
    d_model: int | None = None
    depth: int | None = None
    num_heads: int | None = None
    lr: float | None = None
    device: str | None = None
    wandb: bool = True
    wandb_name: str | None = None
    checkpoint_dir: Path | None = None
    checkpoint_every: int | None = None


def train() -> None:
    parser = argparse.ArgumentParser(description="Train a chess neural network.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--compute-budget", type=float, default=None)
    parser.add_argument("--step-penalty-k", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    args = parser.parse_args()

    run_training(
        TrainOptions(
            config=args.config,
            data=args.data,
            batch_size=args.batch_size,
            compute_budget=args.compute_budget,
            step_penalty_k=args.step_penalty_k,
            d_model=args.d_model,
            depth=args.depth,
            num_heads=args.num_heads,
            lr=args.lr,
            device=args.device,
            wandb=args.wandb,
            wandb_name=args.wandb_name,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
        )
    )


def run_training(options: TrainOptions) -> dict[str, float | int | str]:
    config = with_overrides(
        load_training_config(options.config),
        compute_budget=options.compute_budget,
        step_penalty_k=options.step_penalty_k,
        batch_size=options.batch_size,
        d_model=options.d_model,
        depth=options.depth,
        num_heads=options.num_heads,
        lr=options.lr,
        device=options.device,
    )
    _seed_everything(config.run.seed)

    with torch.device("meta"):
        flops_model = build_model(config.model)
    flops_per_sample = measure_training_flops_per_sample(
        flops_model,
        batch_size=config.data.batch_size,
    )
    steps = steps_for_compute_budget(
        compute_budget=config.run.compute_budget,
        flops_per_sample=flops_per_sample,
        batch_size=config.data.batch_size,
        step_penalty_k=config.run.step_penalty_k,
    )

    dataset = LeelaTarDataset(
        options.data,
        batch_size=config.data.batch_size,
        max_records=config.data.max_records,
    )
    device = _resolve_device(config.run.device)
    model = build_model(config.model).to(device)
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
    completed_steps = 0
    last_metrics: dict[str, float | int] = {}
    checkpoint_paths: list[Path] = []
    final_checkpoint_saved = False
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
        completed_steps = step
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
        flops_seen = seen * flops_per_sample
        compute_seen = step_adjusted_compute(
            flops_per_sample=flops_per_sample,
            batch_size=config.data.batch_size,
            steps=step,
            step_penalty_k=config.run.step_penalty_k,
        )
        metrics["perf/flops_seen"] = flops_seen
        metrics["perf/compute_seen"] = compute_seen
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
                f"flops_seen={flops_seen:.3e} "
                f"compute_seen={compute_seen:.3e} "
                f"samples_per_sec={seen / elapsed:.1f}"
            )
        last_metrics = metrics
        if _should_save_checkpoint(options.checkpoint_dir, options.checkpoint_every, step, steps):
            final_checkpoint_saved = step == steps
            checkpoint_paths.append(
                _save_checkpoint(
                    checkpoint_dir=options.checkpoint_dir,
                    run_name=options.wandb_name or config.run.name,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    samples_seen=seen,
                    flops_per_sample=flops_per_sample,
                    metrics=metrics,
                    final=step == steps,
                )
            )
    if options.checkpoint_dir is not None and completed_steps > 0 and not final_checkpoint_saved:
        checkpoint_paths.append(
            _save_checkpoint(
                checkpoint_dir=options.checkpoint_dir,
                run_name=options.wandb_name or config.run.name,
                config=config,
                model=model,
                optimizer=optimizer,
                step=completed_steps,
                samples_seen=seen,
                flops_per_sample=flops_per_sample,
                metrics=last_metrics,
                final=True,
            )
        )
    if wandb_run is not None:
        wandb_run.finish()
    return {
        "run_name": config.run.name,
        "steps": completed_steps,
        "samples_seen": seen,
        "final_loss": float(last_metrics.get("loss/total", 0.0)),
        "compute_budget": config.run.compute_budget,
        "flops_seen": int(last_metrics.get("perf/flops_seen", 0)),
        "compute_seen": float(last_metrics.get("perf/compute_seen", 0.0)),
        "step_penalty_k": config.run.step_penalty_k,
        "flops_per_sample": flops_per_sample,
        "device": str(device),
        "checkpoint_path": str(checkpoint_paths[-1]) if checkpoint_paths else "",
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
            f"root_m=[{value[:, 4, 2].min():.1f}, {value[:, 4, 2].max():.1f}]"
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


def _should_save_checkpoint(
    checkpoint_dir: Path | None,
    checkpoint_every: int | None,
    step: int,
    total_steps: int,
) -> bool:
    if checkpoint_dir is None:
        return False
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive.")
    if step == total_steps:
        return True
    return checkpoint_every is not None and step % checkpoint_every == 0


def _save_checkpoint(
    *,
    checkpoint_dir: Path | None,
    run_name: str,
    config: TrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    samples_seen: int,
    flops_per_sample: int,
    metrics: dict[str, float | int],
    final: bool,
) -> Path:
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = "final" if final else f"step{step:08d}"
    path = checkpoint_dir / f"{_checkpoint_name(run_name)}-{suffix}.pt"
    torch.save(
        {
            "format_version": 1,
            "run_name": run_name,
            "step": step,
            "samples_seen": samples_seen,
            "flops_per_sample": flops_per_sample,
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    return path


def _checkpoint_name(run_name: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "-" for char in run_name)
    return safe.strip(".-_") or "checkpoint"


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
        "compute_budget": config.run.compute_budget,
        "computed_steps": steps,
        "flops_per_sample": flops_per_sample,
        "step_penalty_k": config.run.step_penalty_k,
        "log_every": config.run.log_every,
        "batch_size": config.data.batch_size,
        "max_records": config.data.max_records,
        "device": str(device),
        "model_kind": config.model.kind,
        "d_model": config.model.d_model,
        "depth": config.model.depth,
        **({"num_heads": config.model.num_heads} if hasattr(config.model, "num_heads") else {}),
        "mlp_ratio": config.model.mlp_ratio,
        "rms_norm_eps": config.model.rms_norm_eps,
        "policy_kind": config.model.policy.kind,
        **(
            {
                "policy_embedding_size": config.model.policy.embedding_size,
                "policy_d_model": config.model.policy.d_model,
            }
            if hasattr(config.model.policy, "embedding_size")
            else {}
        ),
        "lr": config.optimizer.lr,
        "weight_decay": config.optimizer.weight_decay,
        "policy_loss_weight": config.loss.policy,
        "value_loss_weight": config.loss.value,
        "moves_left_loss_weight": config.loss.moves_left,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
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
    root = values[:, 4]
    q_target = root[:, 0]
    q_mse = torch.square(q_pred - q_target).mean()
    moves_left_mae = torch.abs(output.moves_left - root[:, 2]).mean()
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
