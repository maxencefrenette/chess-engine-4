"""Benchmark reference and batched Muon optimizer steps on Modal GPUs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

import modal

from chess_engine_4.hardware import modal_gpu_identifier
from chess_engine_4.modal_train import app, image
from chess_engine_4.training.config import load_training_config

CONFIG = "experiments/2026-08-11.02-muon/config.py"
GPU_BY_WIDTH = {256: "RTX-PRO-6000", 512: "B200", 768: "B200"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", choices=GPU_BY_WIDTH, default=None)
    args = parser.parse_args()
    results = []
    with modal.enable_output(), app.run():
        for width in args.widths or GPU_BY_WIDTH:
            gpu = GPU_BY_WIDTH[width]
            config = load_training_config(CONFIG, d_model=width)
            function = BENCHMARK_FUNCTIONS[gpu]
            results.append(function.remote(asdict(config), 10, 100))
    print(json.dumps(results, indent=2, sort_keys=True))


def _benchmark(
    config_values: dict[str, Any],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch

    from chess_engine_4.model import build_model
    from chess_engine_4.training.cli import _muon_parameter_split
    from chess_engine_4.training.config import training_config_from_dict
    from chess_engine_4.training.muon import BatchedMuon

    config = training_config_from_dict(config_values)
    torch.manual_seed(config.run.seed)
    model = build_model(config.model).cuda()
    reference_model = build_model(config.model).cuda()
    reference_model.load_state_dict(model.state_dict())
    parameters, _ = _muon_parameter_split(model)
    reference_parameters, _ = _muon_parameter_split(reference_model)
    for parameter, reference in zip(parameters, reference_parameters, strict=True):
        gradient = torch.randn_like(parameter)
        parameter.grad = gradient.clone()
        reference.grad = gradient.clone()

    reference_optimizer = torch.optim.Muon(
        reference_parameters,
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        adjust_lr_fn="match_rms_adamw",
    )
    batched_optimizer = BatchedMuon(
        parameters,
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    reference_optimizer.step()
    batched_optimizer.step()
    torch.cuda.synchronize()
    differences = [
        (parameter.float() - reference.float()).abs().max().item()
        for parameter, reference in zip(parameters, reference_parameters, strict=True)
    ]

    reference_ms = _measure(reference_optimizer, warmup=warmup, iterations=iterations)
    batched_ms = _measure(batched_optimizer, warmup=warmup, iterations=iterations)
    return {
        "width": config.model.d_model,
        "gpu": torch.cuda.get_device_name(),
        "eligible_matrices": len(parameters),
        "max_parameter_difference": max(differences),
        "reference_ms": reference_ms,
        "batched_ms": batched_ms,
        "speedup": reference_ms / batched_ms,
    }


def _measure(optimizer: Any, *, warmup: int, iterations: int) -> float:
    import time

    import torch

    for _ in range(warmup):
        optimizer.step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1_000 / iterations


benchmark_rtx_pro_6000 = app.function(
    image=image,
    gpu=modal_gpu_identifier("RTX-PRO-6000"),
    cpu=8,
    timeout=30 * 60,
    name="muon_optimizer_benchmark_rtx_pro_6000",
)(_benchmark)
benchmark_b200 = app.function(
    image=image,
    gpu=modal_gpu_identifier("B200"),
    cpu=8,
    timeout=30 * 60,
    name="muon_optimizer_benchmark_b200",
)(_benchmark)
BENCHMARK_FUNCTIONS = {
    "RTX-PRO-6000": benchmark_rtx_pro_6000,
    "B200": benchmark_b200,
}


if __name__ == "__main__":
    main()
