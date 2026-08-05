"""Training runtime and local data diagnostics."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from chess_engine_4.data.leela import (
    DEFAULT_DATA_ENV_VAR,
    LeelaParquetDataset,
)
from chess_engine_4.model import build_model
from chess_engine_4.model.transformer_engine import autocast_context, te
from chess_engine_4.training.config import TrainingConfig
from chess_engine_4.training.flops import measure_training_flops_per_sample
from chess_engine_4.training.losses import PolicyTarget, lczero_loss
from chess_engine_4.training.packed_input import (
    PackedPlaneInput,
    build_training_model,
)
from chess_engine_4.training.profiling import TrainingProfileConfig, summarize_profile
from chess_engine_4.training.stability import LossSpikeDetector

_DATA_HELP = f"Parquet path, directory, or glob. Defaults to ${DEFAULT_DATA_ENV_VAR}."
_LOG_EVERY = 10
_MATMUL_PRECISION = "high"
_LOSS_EMA_DECAY = 0.99
_POLICY_TOP1_EMA_DECAY = 0.9
_LOSS_TASK_EMA_KEY = "loss/task[ema=0.99]"
_POLICY_TOP1_EMA_KEY = "metrics/policy_top1[ema=0.9]"
_B200_TFLOPS = {"bf16": 2250.0, "mxfp8": 4500.0, "nvfp4": 9000.0}

type NativeBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass(frozen=True, slots=True)
class TrainOptions:
    config: TrainingConfig
    data: str | None = None
    wandb: bool = True
    wandb_name: str | None = None
    checkpoint_dir: Path | None = None
    checkpoint_every: int | None = None
    checkpoint_commit: Callable[[], None] | None = None
    profile: TrainingProfileConfig | None = None
    experimental_dense_kernel: bool = False


def run_training(options: TrainOptions) -> dict[str, Any]:
    config = options.config
    _seed_everything(config.run.seed)
    torch.set_float32_matmul_precision(_MATMUL_PRECISION)

    flops_per_sample = measure_training_flops_per_sample(
        config.model,
        batch_size=config.run.batch_size,
    )
    steps = options.profile.total_steps if options.profile is not None else config.run.steps

    device = torch.device("cuda")
    _require_blackwell(device)
    dataset = LeelaParquetDataset(
        options.data,
        batch_size=config.run.batch_size,
        prefetch_per_thread=config.infra.dataloader_prefetch_per_thread,
        threads=config.infra.dataloader_threads,
    )
    iterator = iter(dataset)
    model = build_model(config.model).to(device)
    if options.experimental_dense_kernel:
        _enable_experimental_dense_kernel(model)
    optimizer = _build_optimizer(model, config=config)
    training_model = build_training_model(
        model,
        batch_size=config.run.batch_size,
        precision=config.precision.recipe,
    )
    theoretical_tflops = _theoretical_tflops(device, precision=config.precision.recipe)
    wandb_run = (
        _init_wandb(
            config,
            options.wandb_name,
            model,
            device,
            steps=steps,
            flops_per_sample=flops_per_sample,
            theoretical_tflops=theoretical_tflops,
        )
        if options.wandb
        else None
    )

    training_model.train()
    start = time.perf_counter()
    profile_records: list[dict[str, Any]] = []
    profile_measured_wall_start: float | None = None
    interval_start = start
    interval_step = 0
    interval_seen = 0
    seen = 0
    completed_steps = 0
    last_metrics: dict[str, float | int] = {}
    ema_metrics: dict[str, float] = {}
    loss_spike_detector = LossSpikeDetector()
    pending_losses: list[torch.Tensor] = []
    checkpoint_paths: list[Path] = []
    final_checkpoint_saved = False
    for step in range(1, steps + 1):
        if options.profile is not None:
            if step == options.profile.warmup_steps + 1:
                profile_measured_wall_start = time.perf_counter()
            fetch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if options.profile is not None:
            fetch_end = time.perf_counter()
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            train_start = torch.cuda.Event(enable_timing=True)
            train_end = torch.cuda.Event(enable_timing=True)
            copy_start.record()
        planes, policy, value = _move_batch_to_device(batch, device=device)
        if options.profile is not None:
            copy_end.record()
            train_start.record()
        current_lr = _set_scheduled_lr(
            optimizer,
            base_lr=config.optimizer.lr,
            warmup_steps=config.optimizer.lr_warmup_steps,
            cooldown_frac=config.optimizer.lr_cooldown_frac,
            step=step,
            total_steps=steps,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(config.precision.recipe):
            output = training_model(planes)
            loss = lczero_loss(output, policy, value, weights=config.loss)
        pending_losses.append(loss.task.detach())
        loss.total.backward()
        grad_norm_tensor = _clip_gradient_norm(
            model,
            max_grad_norm=config.optimizer.max_grad_norm,
        )
        should_log = step == 1 or step % _LOG_EVERY == 0 or step == steps
        should_checkpoint = _should_save_checkpoint(
            options.checkpoint_dir,
            options.checkpoint_every,
            step,
            steps,
        )
        grad_norm = grad_norm_tensor.item() if should_log or should_checkpoint else 0.0
        optimizer.step()
        if options.profile is not None:
            train_end.record()
            profile_records.append(
                {
                    "fetch_wall_ms": (fetch_end - fetch_start) * 1000.0,
                    "enqueue_wall_ms": (time.perf_counter() - fetch_end) * 1000.0,
                    "copy_start": copy_start,
                    "copy_end": copy_end,
                    "train_start": train_start,
                    "train_end": train_end,
                }
            )

        seen += _input_batch_size(planes)
        completed_steps = step
        flops_seen = seen * flops_per_sample

        if should_log or should_checkpoint:
            pending_loss_values = torch.stack(pending_losses)
            torch.cuda.synchronize(device)
            pending_loss_numbers = pending_loss_values.cpu().tolist()
            loss_spike_detector.update_many(pending_loss_numbers)
            pending_losses.clear()
            now = time.perf_counter()
            elapsed = now - start
            interval_elapsed = now - interval_start
            interval_steps = step - interval_step
            interval_samples = seen - interval_seen
            metrics = _training_metrics(
                output=output,
                policy_target=policy,
                values=value,
                loss=loss,
                grad_norm=grad_norm,
                samples_seen=seen,
                elapsed=elapsed,
                lr=current_lr,
            )
            metrics["perf/flops_seen"] = flops_seen
            metrics["stability/loss_spike_count"] = loss_spike_detector.count
            metrics["perf/step_time_sec"] = interval_elapsed / interval_steps
            metrics["perf/samples_per_sec_interval"] = interval_samples / interval_elapsed
            if theoretical_tflops is not None:
                metrics["perf/mfu"] = _mfu(
                    flops=interval_samples * flops_per_sample,
                    elapsed=interval_elapsed,
                    theoretical_tflops=theoretical_tflops,
                )
            _update_ema_metrics(
                metrics,
                ema_metrics,
                loss_tasks=pending_loss_numbers,
            )

            if wandb_run is not None and should_log:
                wandb_run.log(metrics, step=step)
            if should_log:
                mfu_text = f" mfu={metrics['perf/mfu']:.3f}" if "perf/mfu" in metrics else ""
                print(
                    f"step={step} "
                    f"loss={metrics['loss']:.4f} "
                    f"policy={metrics['loss/task/policy']:.4f} "
                    f"value={metrics['loss/task/value']:.4f} "
                    f"mlh={metrics['loss/task/moves_left']:.4f} "
                    f"grad_norm={grad_norm:.2f} "
                    f"flops_seen={flops_seen:.3e} "
                    f"samples_per_sec={metrics['perf/samples_per_sec_interval']:.1f}"
                    f"{mfu_text}"
                )
                interval_start = now
                interval_step = step
                interval_seen = seen
            last_metrics = metrics

        if should_checkpoint:
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
            if options.checkpoint_commit is not None:
                options.checkpoint_commit()
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
        if options.checkpoint_commit is not None:
            options.checkpoint_commit()
    if wandb_run is not None:
        wandb_run.finish()
    result: dict[str, Any] = {
        "run_name": config.run.name,
        "steps": completed_steps,
        "samples_seen": seen,
        "final_loss": float(last_metrics.get("loss", 0.0)),
        "flops_seen": int(last_metrics.get("perf/flops_seen", 0)),
        "flops_per_sample": flops_per_sample,
        "device": str(device),
        "precision": config.precision.recipe,
        "checkpoint_path": str(checkpoint_paths[-1]) if checkpoint_paths else "",
    }
    if options.profile is not None:
        result.update(
            {
                "model_kind": config.model.kind,
                "device_name": torch.cuda.get_device_name(device),
                "batch_size": config.run.batch_size,
                "dataloader_threads": config.infra.dataloader_threads,
                "dataloader_prefetch_per_thread": (config.infra.dataloader_prefetch_per_thread),
            }
        )
        result.update(
            summarize_profile(
                profile=options.profile,
                records=profile_records,
                measured_wall_start=profile_measured_wall_start,
                overall_wall_start=start,
                device=device,
                batch_size=config.run.batch_size,
                flops_per_sample=flops_per_sample,
                theoretical_tflops=theoretical_tflops,
            )
        )
    return result


def _enable_experimental_dense_kernel(model: torch.nn.Module) -> None:
    from chess_engine_4.model import DenseChessNet

    if not isinstance(model, DenseChessNet):
        raise ValueError("the experimental dense kernel only supports dense models")
    model.enable_experimental_d128_kernel()


def inspect_data() -> None:
    parser = argparse.ArgumentParser(description="Inspect Leela training records.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--batch-size", type=int, default=1024)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument(
        "--batches",
        type=int,
        default=1,
        help="Number of batches to inspect (default: 1).",
    )
    limit.add_argument("--all", action="store_true", help="Inspect the entire dataset.")
    args = parser.parse_args()
    if args.batches <= 0:
        parser.error("--batches must be positive")

    dataset = LeelaParquetDataset(args.data, batch_size=args.batch_size)
    seen = 0
    for batch_number, batch in enumerate(dataset, start=1):
        packed_planes, plane_scalars, policy_indices, policy_probs, value = batch
        batch_size = packed_planes.shape[0]
        legal = policy_indices >= 0
        legal_policy_sum = policy_probs.float().sum(dim=1)
        legal_count = legal.sum(dim=1)
        seen += batch_size
        print(
            f"records={seen} "
            f"packed_planes={tuple(packed_planes.shape)} "
            f"plane_scalars={tuple(plane_scalars.shape)} "
            f"policy_indices={tuple(policy_indices.shape)} "
            f"policy_probs={tuple(policy_probs.shape)} "
            f"value={tuple(value.shape)} "
            f"legal_policy_sum=[{legal_policy_sum.min():.4f}, {legal_policy_sum.max():.4f}] "
            f"legal_moves=[{legal_count.min()}, {legal_count.max()}] "
            f"root_m=[{value[:, 4, 2].min():.1f}, {value[:, 4, 2].max():.1f}]"
        )
        if not args.all and batch_number >= args.batches:
            break


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch_to_device(
    batch: NativeBatch,
    *,
    device: torch.device,
) -> tuple[PackedPlaneInput, PolicyTarget, torch.Tensor]:
    batch = tuple(tensor.pin_memory() for tensor in batch)
    packed_planes, plane_scalars, policy_indices, policy_probs, value = batch
    packed_planes = packed_planes.to(device=device, non_blocking=True)
    plane_scalars = plane_scalars.to(
        device=device,
        dtype=torch.bfloat16,
        non_blocking=True,
    )
    planes = (packed_planes, plane_scalars)
    return (
        planes,
        (
            policy_indices.to(device=device, non_blocking=True),
            policy_probs.to(device=device, non_blocking=True),
        ),
        value.to(device, non_blocking=True),
    )


def _input_batch_size(planes: PackedPlaneInput) -> int:
    return int(planes[0].shape[0])


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
            "format_version": 2,
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


def _build_optimizer(
    model: torch.nn.Module,
    *,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    return te().optimizers.FusedAdam(
        _adamw_parameter_groups(model, weight_decay=config.optimizer.weight_decay),
        lr=config.optimizer.lr,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )


def _theoretical_tflops(device: torch.device, *, precision: str) -> float:
    _require_blackwell(device)
    return _B200_TFLOPS[precision]


def _require_blackwell(device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if capability != (10, 0) or "B200" not in name:
        raise RuntimeError(
            "chess-engine-4 requires an NVIDIA B200 (SM100); "
            f"found {name} SM{capability[0]}{capability[1]}."
        )


def _mfu(*, flops: int, elapsed: float, theoretical_tflops: float) -> float:
    if elapsed <= 0:
        return 0.0
    return flops / elapsed / (theoretical_tflops * 1e12)


def _set_scheduled_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    warmup_steps: int,
    cooldown_frac: float,
    step: int,
    total_steps: int,
) -> float:
    lr = _scheduled_lr(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        cooldown_frac=cooldown_frac,
        step=step,
        total_steps=total_steps,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _scheduled_lr(
    *,
    base_lr: float,
    warmup_steps: int,
    cooldown_frac: float,
    step: int,
    total_steps: int,
) -> float:
    if base_lr <= 0:
        raise ValueError("base_lr must be positive.")
    if warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative.")
    if not 0.0 <= cooldown_frac < 1.0:
        raise ValueError("lr_cooldown_frac must be in [0, 1).")
    if step <= 0 or total_steps <= 0:
        raise ValueError("step and total_steps must be positive.")
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    cooldown_steps = round(total_steps * cooldown_frac)
    if cooldown_steps <= 0:
        return base_lr
    cooldown_start = total_steps - cooldown_steps
    if step <= cooldown_start:
        return base_lr
    progress = 1.0 if cooldown_steps == 1 else (step - cooldown_start) / cooldown_steps
    multiplier = 1.0 - progress
    return base_lr * multiplier


def _init_wandb(
    config: TrainingConfig,
    run_name: str | None,
    model: torch.nn.Module,
    device: torch.device,
    *,
    steps: int,
    flops_per_sample: int,
    theoretical_tflops: float | None,
) -> Any:
    import wandb

    wandb_config = {
        "run_name": config.run.name,
        "seed": config.run.seed,
        "steps": steps,
        "flops_per_sample": flops_per_sample,
        "log_every": _LOG_EVERY,
        "batch_size": config.run.batch_size,
        "training_ratio": config.run.training_ratio,
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(device),
        "precision": config.precision.recipe,
        "matmul_precision": _MATMUL_PRECISION,
        "theoretical_tflops": theoretical_tflops,
        "dataloader_threads": config.infra.dataloader_threads,
        "dataloader_prefetch_per_thread": config.infra.dataloader_prefetch_per_thread,
        "model_kind": config.model.kind,
        "d_model": config.model.d_model,
        "depth": config.model.depth,
        "expansion_ratio": config.model.expansion_ratio,
        "history_length": config.model.history_length,
        "activation": config.model.activation,
        "rms_norm_eps": config.model.rms_norm_eps,
        "lr": config.optimizer.lr,
        "weight_decay": config.optimizer.weight_decay,
        "max_grad_norm": config.optimizer.max_grad_norm,
        "lr_warmup_steps": config.optimizer.lr_warmup_steps,
        "lr_cooldown_frac": config.optimizer.lr_cooldown_frac,
        "fused_adamw": True,
        "policy_loss_weight": config.loss.policy,
        "value_loss_weight": config.loss.value,
        "moves_left_loss_weight": config.loss.moves_left,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if config.model.kind == "moe64a2":
        wandb_config["num_experts"] = config.model.num_experts
        wandb_config["num_active_experts"] = config.model.num_active_experts
        wandb_config["router_aux_loss_weight"] = config.loss.router_aux
    return wandb.init(
        name=run_name or config.run.name,
        config=wandb_config,
    )


def _clip_gradient_norm(
    model: torch.nn.Module,
    *,
    max_grad_norm: float,
) -> torch.Tensor:
    if max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative.")
    if max_grad_norm == 0:
        return _gradient_norm_tensor(model)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def _gradient_norm_tensor(model: torch.nn.Module) -> torch.Tensor:
    norms = [
        parameter.grad.detach().norm(2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms), 2)


def _training_metrics(
    *,
    output: Any,
    policy_target: tuple[torch.Tensor, torch.Tensor],
    values: torch.Tensor,
    loss: Any,
    grad_norm: float,
    samples_seen: int,
    elapsed: float,
    lr: float,
) -> dict[str, float | int]:
    policy_indices, policy_probs = policy_target
    valid_policy = policy_indices >= 0
    policy_targets = policy_probs.float()
    policy_entropy = -(policy_targets * policy_targets.clamp_min(1e-30).log()).sum(dim=-1).mean()
    gathered_logits = output.policy_logits.gather(
        dim=-1,
        index=policy_indices.clamp_min(0).long(),
    )
    gathered_logits = gathered_logits.masked_fill(
        ~valid_policy,
        torch.finfo(gathered_logits.dtype).min,
    )
    policy_top1 = (gathered_logits.argmax(dim=-1) == policy_targets.argmax(dim=-1)).float().mean()
    wdl_probs = torch.softmax(output.wdl_logits, dim=-1)
    q_pred = wdl_probs[:, 0] - wdl_probs[:, 2]
    root = values[:, 4]
    q_target = root[:, 0]
    q_mse = torch.square(q_pred - q_target).mean()
    moves_left_mae = torch.abs(output.moves_left - root[:, 2]).mean()
    samples_per_sec = samples_seen / elapsed if elapsed > 0 else 0.0
    metrics = {
        "loss": loss.task.item(),
        "loss/task": loss.task.item(),
        "loss/task/policy": loss.policy.item(),
        "loss/task/value": loss.value.item(),
        "loss/task/moves_left": loss.moves_left.item(),
        "metrics/policy_entropy": policy_entropy.item(),
        "metrics/policy_top1": policy_top1.item(),
        "metrics/value_q_mse": q_mse.item(),
        "metrics/moves_left_mae": moves_left_mae.item(),
        "optim/lr": lr,
        "optim/grad_norm": grad_norm,
        "perf/samples_per_sec": samples_per_sec,
        "perf/samples_seen": samples_seen,
    }
    if loss.router_aux is not None:
        metrics["loss/aux"] = loss.aux.item()
        metrics["loss/aux/router"] = loss.router_aux.item()
    if output.router_dead_experts is not None:
        metrics["router/dead_experts"] = output.router_dead_experts.item()
    return metrics


def _update_ema_metrics(
    metrics: dict[str, float | int],
    ema_metrics: dict[str, float],
    *,
    loss_tasks: list[float],
) -> None:
    for loss_task in loss_tasks:
        _update_ema_metric(
            metrics,
            ema_metrics,
            _LOSS_TASK_EMA_KEY,
            loss_task,
            decay=_LOSS_EMA_DECAY,
        )
    _update_ema_metric(
        metrics,
        ema_metrics,
        _POLICY_TOP1_EMA_KEY,
        float(metrics["metrics/policy_top1"]),
        decay=_POLICY_TOP1_EMA_DECAY,
    )


def _update_ema_metric(
    metrics: dict[str, float | int],
    ema_metrics: dict[str, float],
    ema_key: str,
    value: float,
    *,
    decay: float,
) -> None:
    previous = ema_metrics.get(ema_key)
    next_value = value if previous is None else (decay * previous + (1.0 - decay) * value)
    ema_metrics[ema_key] = next_value
    metrics[ema_key] = next_value
