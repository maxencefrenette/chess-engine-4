"""Command-line entrypoints for local training workflows."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
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
from chess_engine_4.training.packed_input import PackedInputTrainingModel

_DATA_HELP = f"Leela tar path, directory, or glob. Defaults to ${DEFAULT_DATA_ENV_VAR}."
_DEFAULT_CONFIG_PATH = Path("configs/mlp/1e18.toml")
_LOG_EVERY = 10
_MATMUL_PRECISION = "high"
_METRIC_EMA_DECAY = 0.99
_LOSS_TASK_EMA_KEY = "loss/task[ema=0.99]"
_LOSS_TASK2_EMA_KEY = "loss/task2[ema=0.99]"
_POLICY_TOP1_EMA_KEY = "metrics/policy_top1[ema=0.99]"
_BF16_TFLOPS_BY_GPU = {
    "NVIDIA L4": 121.0,
    "NVIDIA A10G": 125.0,
    "NVIDIA A100": 312.0,
    "NVIDIA L40S": 362.0,
    "NVIDIA H100": 989.0,
    "NVIDIA H200": 989.0,
    "NVIDIA B200": 2250.0,
}
_FP32_TFLOPS_BY_GPU = {
    "Tesla T4": 8.1,
    "NVIDIA T4": 8.1,
    "NVIDIA L4": 30.3,
    "NVIDIA A10G": 31.2,
    "NVIDIA A100": 19.5,
    "NVIDIA L40S": 91.6,
    "NVIDIA H100": 67.0,
    "NVIDIA H200": 67.0,
    "NVIDIA B200": 90.0,
}


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
    router_aux: float | None = None
    device: str | None = None
    max_steps: int | None = None
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
    parser.add_argument("--router-aux", type=float, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-steps", type=int, default=None)
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
            router_aux=args.router_aux,
            device=args.device,
            max_steps=args.max_steps,
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
        router_aux=options.router_aux,
    )
    _seed_everything(config.run.seed)
    torch.set_float32_matmul_precision(_MATMUL_PRECISION)

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
    if options.max_steps is not None:
        if options.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        steps = min(steps, options.max_steps)

    device = _resolve_device(options.device)
    precision = _training_precision(device)
    dataset = LeelaTarDataset(
        options.data,
        batch_size=config.data.batch_size,
        max_records=config.data.max_records,
        prefetch_factor=config.infra.dataloader_prefetch_factor,
    )
    model = build_model(config.model).to(device)
    training_model = _compile_model_for_training(
        PackedInputTrainingModel(model).to(device),
        device=device,
    )
    optimizer = _build_optimizer(model, config=config, device=device)
    theoretical_tflops = _theoretical_tflops(device, precision=precision)
    wandb_run = (
        _init_wandb(
            config,
            options.wandb_name,
            model,
            device,
            steps=steps,
            flops_per_sample=flops_per_sample,
            theoretical_tflops=theoretical_tflops,
            precision=precision,
        )
        if options.wandb
        else None
    )

    training_model.train()
    start = time.perf_counter()
    interval_start = start
    interval_step = 0
    interval_seen = 0
    seen = 0
    completed_steps = 0
    last_metrics: dict[str, float | int] = {}
    ema_metrics: dict[str, float] = {}
    checkpoint_paths: list[Path] = []
    final_checkpoint_saved = False
    for step, batch in enumerate(islice(dataset, steps), start=1):
        planes, policy, value = _move_batch_to_device(batch, device=device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, precision=precision):
            output = training_model(planes)
            loss = lczero_loss(output, policy, value, weights=config.loss)
        loss.total.backward()
        should_log = step == 1 or step % _LOG_EVERY == 0 or step == steps
        should_checkpoint = _should_save_checkpoint(
            options.checkpoint_dir,
            options.checkpoint_every,
            step,
            steps,
        )
        grad_norm = _gradient_norm(model) if should_log or should_checkpoint else 0.0
        optimizer.step()

        seen += _input_batch_size(planes)
        completed_steps = step
        flops_seen = seen * flops_per_sample
        compute_seen = step_adjusted_compute(
            flops_per_sample=flops_per_sample,
            batch_size=config.data.batch_size,
            steps=step,
            step_penalty_k=config.run.step_penalty_k,
        )

        if should_log or should_checkpoint:
            _synchronize_if_cuda(device)
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
                lr=config.optimizer.lr,
            )
            metrics["perf/flops_seen"] = flops_seen
            metrics["perf/compute_seen"] = compute_seen
            metrics["perf/step_time_sec"] = interval_elapsed / interval_steps
            metrics["perf/samples_per_sec_interval"] = interval_samples / interval_elapsed
            if theoretical_tflops is not None:
                metrics["perf/mfu"] = _mfu(
                    flops=interval_samples * flops_per_sample,
                    elapsed=interval_elapsed,
                    theoretical_tflops=theoretical_tflops,
                )
            _update_ema_metrics(metrics, ema_metrics)

            if wandb_run is not None and should_log:
                wandb_run.log(metrics, step=step)
            if should_log:
                mfu_text = (
                    f" mfu={metrics['perf/mfu']:.3f}" if "perf/mfu" in metrics else ""
                )
                print(
                    f"step={step} "
                    f"loss={metrics['loss']:.4f} "
                    f"policy={metrics['loss/task/policy']:.4f} "
                    f"value={metrics['loss/task/value']:.4f} "
                    f"mlh={metrics['loss/task/moves_left']:.4f} "
                    f"aux={metrics['loss/aux']:.4f} "
                    f"grad_norm={grad_norm:.2f} "
                    f"flops_seen={flops_seen:.3e} "
                    f"compute_seen={compute_seen:.3e} "
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
        "final_loss": float(last_metrics.get("loss", 0.0)),
        "compute_budget": config.run.compute_budget,
        "flops_seen": int(last_metrics.get("perf/flops_seen", 0)),
        "compute_seen": float(last_metrics.get("perf/compute_seen", 0.0)),
        "step_penalty_k": config.run.step_penalty_k,
        "flops_per_sample": flops_per_sample,
        "device": str(device),
        "precision": precision,
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
    for packed_planes, plane_scalars, policy_indices, policy_probs, value in dataset:
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


def sample_batch() -> None:
    parser = argparse.ArgumentParser(description="Load and print one Leela training batch.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    dataset = LeelaTarDataset(args.data, batch_size=args.batch_size, max_records=args.batch_size)
    packed_planes, plane_scalars, policy_indices, policy_probs, value = next(iter(dataset))
    print(f"packed_planes: {tuple(packed_planes.shape)}")
    print(f"plane_scalars: {tuple(plane_scalars.shape)}")
    print(f"policy_indices: {tuple(policy_indices.shape)}")
    print(f"policy_probs: {tuple(policy_probs.shape)}")
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


def _training_precision(device: torch.device) -> str:
    if device.type == "cuda":
        if torch.cuda.get_device_capability(device)[0] < 8:
            name = torch.cuda.get_device_name(device)
            raise RuntimeError(f"CUDA training requires native bf16 support; {name} is too old.")
        if not torch.cuda.is_bf16_supported():
            name = torch.cuda.get_device_name(device)
            raise RuntimeError(f"CUDA training requires bf16 support; {name} reports none.")
        return "bf16"
    return "fp32"


def _move_batch_to_device(
    batch: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
) -> tuple[Any, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    non_blocking = device.type == "cuda"
    plane_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        batch = tuple(tensor.pin_memory() for tensor in batch)
    packed_planes, plane_scalars, policy_indices, policy_probs, value = batch
    packed_planes = packed_planes.to(device=device, non_blocking=non_blocking)
    plane_scalars = plane_scalars.to(
        device=device,
        dtype=plane_dtype,
        non_blocking=non_blocking,
    )
    planes = (packed_planes, plane_scalars)
    return (
        planes,
        (
            policy_indices.to(device=device, non_blocking=non_blocking),
            policy_probs.to(device=device, non_blocking=non_blocking),
        ),
        value.to(device, non_blocking=non_blocking),
    )


def _input_batch_size(planes: Any) -> int:
    if isinstance(planes, tuple):
        return int(planes[0].shape[0])
    return int(planes.shape[0])


def _autocast_context(device: torch.device, *, precision: str) -> Any:
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _compile_model_for_training(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> torch.nn.Module:
    if device.type == "cuda":
        return torch.compile(model, mode="reduce-overhead")
    return model


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _build_optimizer(
    model: torch.nn.Module,
    *,
    config: TrainingConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {}
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(
        _adamw_parameter_groups(model, weight_decay=config.optimizer.weight_decay),
        lr=config.optimizer.lr,
        **kwargs,
    )


def _theoretical_tflops(device: torch.device, *, precision: str) -> float | None:
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    table = _BF16_TFLOPS_BY_GPU if precision == "bf16" else _FP32_TFLOPS_BY_GPU
    for prefix, tflops in table.items():
        if name.startswith(prefix):
            return tflops
    return None


def _mfu(*, flops: int, elapsed: float, theoretical_tflops: float) -> float:
    if elapsed <= 0:
        return 0.0
    return flops / elapsed / (theoretical_tflops * 1e12)


def _init_wandb(
    config: TrainingConfig,
    run_name: str | None,
    model: torch.nn.Module,
    device: torch.device,
    *,
    steps: int,
    flops_per_sample: int,
    theoretical_tflops: float | None,
    precision: str,
) -> Any:
    import wandb

    wandb_config = {
        "run_name": config.run.name,
        "seed": config.run.seed,
        "compute_budget": config.run.compute_budget,
        "computed_steps": steps,
        "flops_per_sample": flops_per_sample,
        "step_penalty_k": config.run.step_penalty_k,
        "log_every": _LOG_EVERY,
        "batch_size": config.data.batch_size,
        "max_records": config.data.max_records,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": precision,
        "matmul_precision": _MATMUL_PRECISION,
        "theoretical_tflops": theoretical_tflops,
        "gpu_type": config.infra.gpu_type,
        "dataloader_prefetch_factor": config.infra.dataloader_prefetch_factor,
        "model_kind": config.model.kind,
        "d_model": config.model.d_model,
        "depth": config.model.depth,
        **({"num_heads": config.model.num_heads} if hasattr(config.model, "num_heads") else {}),
        **({"mlp_ratio": config.model.mlp_ratio} if hasattr(config.model, "mlp_ratio") else {}),
        **(
            {"expert_mlp_ratio": config.model.expert_mlp_ratio}
            if hasattr(config.model, "expert_mlp_ratio")
            else {}
        ),
        **(
            {"num_experts": config.model.num_experts}
            if hasattr(config.model, "num_experts")
            else {}
        ),
        **(
            {"num_experts_per_token": config.model.num_experts_per_token}
            if hasattr(config.model, "num_experts_per_token")
            else {}
        ),
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
        "fused_adamw": device.type == "cuda",
        "policy_loss_weight": config.loss.policy,
        "value_loss_weight": config.loss.value,
        "moves_left_loss_weight": config.loss.moves_left,
        "router_aux_loss_weight": config.loss.router_aux,
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
    policy_entropy = -(
        policy_targets * policy_targets.clamp_min(1e-30).log()
    ).sum(dim=-1).mean()
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
        "loss/train": loss.total.item(),
        "loss/task": loss.task.item(),
        "loss/task/policy": loss.policy.item(),
        "loss/task/value": loss.value.item(),
        "loss/task/moves_left": loss.moves_left.item(),
        "loss/aux": loss.aux.item(),
        "loss/aux/router": loss.router_aux.item(),
        "metrics/policy_entropy": policy_entropy.item(),
        "metrics/policy_top1": policy_top1.item(),
        "metrics/value_q_mse": q_mse.item(),
        "metrics/moves_left_mae": moves_left_mae.item(),
        "optim/lr": lr,
        "optim/grad_norm": grad_norm,
        "perf/samples_per_sec": samples_per_sec,
        "perf/samples_seen": samples_seen,
    }
    if output.router_dead_experts is not None:
        metrics["router/dead_experts"] = output.router_dead_experts.item()
    if output.router_dead_experts_max is not None:
        metrics["router/dead_experts_max"] = output.router_dead_experts_max.item()
    return metrics


def _update_ema_metrics(
    metrics: dict[str, float | int],
    ema_metrics: dict[str, float],
) -> None:
    loss_task = float(metrics["loss/task"])
    _update_ema_metric(metrics, ema_metrics, _LOSS_TASK_EMA_KEY, loss_task)
    _update_ema_metric(metrics, ema_metrics, _LOSS_TASK2_EMA_KEY, loss_task * loss_task)
    _update_ema_metric(
        metrics,
        ema_metrics,
        _POLICY_TOP1_EMA_KEY,
        float(metrics["metrics/policy_top1"]),
    )


def _update_ema_metric(
    metrics: dict[str, float | int],
    ema_metrics: dict[str, float],
    ema_key: str,
    value: float,
) -> None:
    previous = ema_metrics.get(ema_key)
    next_value = value if previous is None else (
        _METRIC_EMA_DECAY * previous + (1.0 - _METRIC_EMA_DECAY) * value
    )
    ema_metrics[ema_key] = next_value
    metrics[ema_key] = next_value
