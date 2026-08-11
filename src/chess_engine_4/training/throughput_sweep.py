"""Persistent Modal throughput benchmarks for model-family ladders."""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomli_w

from chess_engine_4.hardware import (
    CPU_DOLLARS_PER_CORE_SECOND,
    TRAINING_GPUS,
    gpu_spec,
    hardware_dollars_per_second,
)
from chess_engine_4.modal_train import (
    DEFAULT_CONFIG_PATH,
    app,
    print_launch_summary,
    training_function,
)
from chess_engine_4.model import model_parameter_count
from chess_engine_4.training.config import (
    TrainingConfig,
    load_training_config,
    validate_training_hardware,
    with_overrides,
)

DEFAULT_WIDTHS = (64, 128, 256, 512, 768, 1024, 1280)
DEFAULT_OUTPUT = Path("experiments/throughput-dense.toml")
TRAINING_RATIOS = (0.25, 0.5, 1.0)


def throughput_sweep() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and cache model-family training throughput on Modal."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--profile-steps", type=int, default=500)
    parser.add_argument("--gpu", choices=TRAINING_GPUS, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--batch-divisor",
        type=int,
        default=1,
        help=(
            "Divide the recipe batch size and multiply its steps by this value, "
            "preserving the 1x sample count."
        ),
    )
    parser.add_argument(
        "--quantization-recipe",
        choices=("bf16", "mxfp8", "nvfp4"),
        default=None,
    )
    parser.add_argument("--kernel-backend", choices=("te", "custom"), default=None)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rerun selected widths even when matching cached results exist.",
    )
    args = parser.parse_args()

    widths = normalize_widths(args.widths)
    if args.warmup_steps < 0:
        parser.error("warmup-steps must be non-negative.")
    if args.profile_steps <= 0:
        parser.error("profile-steps must be positive.")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("batch-size must be positive.")
    if args.batch_divisor <= 0:
        parser.error("batch-divisor must be positive.")
    if args.batch_size is not None and args.batch_divisor != 1:
        parser.error("batch-size and batch-divisor are mutually exclusive.")

    cached = load_results(args.output)
    configs = {
        width: _benchmark_config(
            load_training_config(args.config, d_model=width, training_ratio=1.0),
            gpu=args.gpu,
            batch_size=args.batch_size,
            batch_divisor=args.batch_divisor,
            quantization_recipe=args.quantization_recipe,
            kernel_backend=args.kernel_backend,
        )
        for width in widths
    }
    for config in configs.values():
        validate_training_hardware(config)
    pending = [
        width
        for width in widths
        if args.refresh
        or not entry_matches(
            cached.get(model_key(width)),
            configs[width],
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
        )
    ]
    skipped = [width for width in widths if width not in pending]
    if skipped:
        print("cached widths: " + ", ".join(model_key(width) for width in skipped))

    errors: list[str] = []
    if pending:
        print("profiling widths: " + ", ".join(model_key(width) for width in pending))
        for width in pending:
            print_launch_summary(
                configs[width],
                steps=args.warmup_steps + args.profile_steps,
            )
        results, errors = run_modal_profiles(
            pending,
            configs,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
        )
        commit = git_commit()
        for width, profile in results.items():
            cached[model_key(width)] = make_entry(
                configs[width],
                profile,
                source_commit=commit,
            )
        write_results(
            args.output,
            cached,
            config_path=args.config,
            model_family=_model_family(configs),
            gpu=_results_gpu(cached),
        )
        print(f"wrote {args.output}")
    elif not args.output.exists():
        write_results(
            args.output,
            cached,
            config_path=args.config,
            model_family=_model_family(configs),
            gpu=_results_gpu(cached),
        )

    print_report(widths, cached)
    if errors:
        raise SystemExit("\n".join(errors))


def _benchmark_config(
    config: TrainingConfig,
    *,
    gpu: str | None,
    batch_size: int | None,
    batch_divisor: int,
    quantization_recipe: str | None,
    kernel_backend: str | None,
) -> TrainingConfig:
    if batch_divisor <= 0:
        raise ValueError("batch_divisor must be positive")
    if config.run.batch_size % batch_divisor:
        raise ValueError(
            f"batch size {config.run.batch_size} is not divisible by {batch_divisor}"
        )
    if batch_size is not None:
        selected_batch = batch_size
        selected_steps = config.run.steps
    else:
        selected_batch = config.run.batch_size // batch_divisor
        selected_steps = config.run.steps * batch_divisor
    return with_overrides(
        config,
        gpu=gpu,
        batch_size=selected_batch,
        steps=selected_steps,
        quantization_recipe=quantization_recipe,
        kernel_backend=kernel_backend,
    )


def normalize_widths(widths: list[int]) -> list[int]:
    if any(width <= 0 for width in widths):
        raise ValueError("widths must be positive.")
    return sorted(set(widths))


def model_key(width: int) -> str:
    return f"d{width}"


def profile_payload(
    config: TrainingConfig,
    *,
    warmup_steps: int,
    profile_steps: int,
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "wandb": False,
        "profile": {
            "warmup_steps": warmup_steps,
            "profile_steps": profile_steps,
        },
    }


def run_modal_profiles(
    widths: list[int],
    configs: dict[int, TrainingConfig],
    *,
    warmup_steps: int,
    profile_steps: int,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    function_keys = {
        (
            configs[width].infra.cpu_cores,
            configs[width].infra.gpu,
            configs[width].model.kernel_backend,
        )
        for width in widths
    }
    functions = {
        key: training_function(key[0], gpu=key[1], kernel_backend=key[2]) for key in function_keys
    }
    completed: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    with app.run():
        for width in widths:
            cpu_cores = configs[width].infra.cpu_cores
            gpu = configs[width].infra.gpu
            function_key = (cpu_cores, gpu, configs[width].model.kernel_backend)
            function = functions[function_key]
            try:
                completed[width] = function.remote(
                    profile_payload(
                        configs[width],
                        warmup_steps=warmup_steps,
                        profile_steps=profile_steps,
                    )
                )
            except Exception as exc:
                errors.append(f"{model_key(width)} failed: {exc}")
    return completed, errors


def make_entry(
    config: TrainingConfig,
    profile: dict[str, Any],
    *,
    source_commit: str,
) -> dict[str, Any]:
    model = config.model
    params = model_parameter_count(model)
    milliseconds = float(profile["measured_wall_ms_per_step"])
    runtime_sec = config.run.steps * milliseconds / 1000.0
    gpu_rate = gpu_spec(config.infra.gpu).dollars_per_second
    cpu_rate = config.infra.cpu_cores * CPU_DOLLARS_PER_CORE_SECOND
    entry: dict[str, Any] = {
        "source_commit": source_commit,
        "d_model": model.d_model,
        "depth": model.depth,
        "expansion_ratio": model.expansion_ratio,
        "activation": model.activation,
        "params": params,
        "batch_size": config.run.batch_size,
        "steps_1x": config.run.steps,
        "samples_1x": config.run.batch_size * config.run.steps,
        "precision": config.model.precision,
        "kernel_backend": config.model.kernel_backend,
        "input_pipeline": config.model.input_pipeline,
        "gpu": config.infra.gpu,
        "cpu_cores": config.infra.cpu_cores,
        "dataloader_threads": config.infra.dataloader_threads,
        "dataloader_prefetch_per_thread": config.infra.dataloader_prefetch_per_thread,
        "warmup_steps": int(profile["warmup_steps"]),
        "profile_steps": int(profile["profile_steps"]),
        "flops_per_sample": int(profile["flops_per_sample"]),
        "measured_wall_ms_per_step": milliseconds,
        "samples_per_sec": config.run.batch_size / (milliseconds / 1000.0),
        "train_gpu_ms_per_step": float(profile["train_gpu"]["mean_ms"]),
        "pin_memory_ms_per_step": float(profile["pin_memory_wall"]["mean_ms"]),
        "h2d_enqueue_ms_per_step": float(profile["h2d_enqueue_wall"]["mean_ms"]),
        "h2d_copy_ms_per_step": float(profile["h2d_copy_gpu"]["mean_ms"]),
        "data_fetch_ms_per_step": float(profile["data_fetch_wall"]["mean_ms"]),
        "gpu_idle_gap_ms_per_step": float(profile["gpu_idle_gap_mean_ms"]),
        "peak_memory_allocated_bytes": int(profile["peak_memory_allocated_bytes"]),
        "peak_memory_reserved_bytes": int(profile["peak_memory_reserved_bytes"]),
        "train_only_mfu": float(profile["train_only_mfu"]),
        "end_to_end_mfu": float(profile["end_to_end_mfu"]),
        "gpu_dollars_per_second": gpu_rate,
        "cpu_dollars_per_second": cpu_rate,
        "hardware_dollars_per_second": hardware_dollars_per_second(
            config.infra.gpu,
            config.infra.cpu_cores,
        ),
        "estimated_runtime_sec_1x": runtime_sec,
        "estimated_cost_dollars_1x": runtime_sec * (gpu_rate + cpu_rate),
    }
    return entry


def entry_matches(
    entry: dict[str, Any] | None,
    config: TrainingConfig,
    *,
    warmup_steps: int,
    profile_steps: int,
) -> bool:
    if entry is None:
        return False
    expected = {
        "d_model": config.model.d_model,
        "depth": config.model.depth,
        "expansion_ratio": config.model.expansion_ratio,
        "activation": config.model.activation,
        "batch_size": config.run.batch_size,
        "precision": config.model.precision,
        "kernel_backend": config.model.kernel_backend,
        "input_pipeline": config.model.input_pipeline,
        "gpu": config.infra.gpu,
        "cpu_cores": config.infra.cpu_cores,
        "dataloader_threads": config.infra.dataloader_threads,
        "dataloader_prefetch_per_thread": config.infra.dataloader_prefetch_per_thread,
        "warmup_steps": warmup_steps,
        "profile_steps": profile_steps,
    }
    return all(entry.get(key) == value for key, value in expected.items())


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("rb") as result_file:
        data = tomllib.load(result_file)
    return dict(data.get("models", {}))


def write_results(
    path: Path,
    models: dict[str, dict[str, Any]],
    *,
    config_path: Path,
    model_family: str,
    gpu: str,
) -> None:
    data = {
        "sweep": {
            "model_family": model_family,
            "config": str(config_path),
            "gpu": gpu,
            "updated_at": datetime.now(UTC).isoformat(),
        },
        "models": {key: models[key] for key in sorted(models, key=_key_width)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(tomli_w.dumps(data), encoding="utf-8")
    temporary.replace(path)


def _model_family(configs: dict[int, TrainingConfig]) -> str:
    families = {config.model.kind for config in configs.values()}
    if len(families) != 1:
        raise ValueError(f"throughput sweep configs must use one model family, got {families}")
    return families.pop()


def _results_gpu(models: dict[str, dict[str, Any]]) -> str:
    gpus = {str(row["gpu"]) for row in models.values()}
    return gpus.pop() if len(gpus) == 1 else "mixed"


def print_report(widths: list[int], models: dict[str, dict[str, Any]]) -> None:
    available = [models[model_key(width)] for width in widths if model_key(width) in models]
    if not available:
        return
    print("")
    print("model  params    batch     ms/step  samples/s  MFU    0.25x     0.5x      1x")
    for entry in available:
        runtimes = [
            _format_duration(float(entry["estimated_runtime_sec_1x"]) * ratio)
            for ratio in TRAINING_RATIOS
        ]
        print(
            f"d{entry['d_model']:<5} "
            f"{_format_count(int(entry['params'])):>8} "
            f"{entry['batch_size']:>8,} "
            f"{entry['measured_wall_ms_per_step']:>9.2f} "
            f"{entry['samples_per_sec']:>10,.0f} "
            f"{entry['end_to_end_mfu']:>5.1%} "
            f"{runtimes[0]:>9} {runtimes[1]:>9} {runtimes[2]:>9}"
        )


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _key_width(key: str) -> int:
    return int(key.removeprefix("d"))


def _format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value / 1_000:.0f}K"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"
