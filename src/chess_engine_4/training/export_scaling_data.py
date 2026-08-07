"""Export canonical scaling-law data for the static website."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from functools import cache
from pathlib import Path
from typing import Any

from chess_engine_4.model import model_parameter_count
from chess_engine_4.training.config import load_training_config
from chess_engine_4.training.flops import measure_training_flops_per_sample
from chess_engine_4.training.scaling_laws import (
    fit_loss_power_law,
    fit_power_law,
    fit_sigmoid_law,
    read_best_runs,
)

DEFAULT_OUTPUT = Path("website/src/generated/scaling-laws.json")
CURVE_POINT_COUNT = 61
FAMILIES = {
    "dense": {
        "name": "Dense",
        "description": "Stacked MLP trained on lc0 planes.",
        "best_runs": Path("experiments/best-runs-dense.toml"),
        "config": Path("configs/dense.py"),
        "training_ratio": 0.2,
    },
    "moe64a2": {
        "name": "MoE 64A2",
        "description": "Stacked MLP with alternating dense and 64-expert, 2-active layers.",
        "best_runs": Path("experiments/best-runs-moe64a2.toml"),
        "config": Path("configs/moe64a2.py"),
        "training_ratio": 0.05,
        "extrapolate": False,
    },
}


def export_scaling_data() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_scaling_data(args.output)
    print(f"wrote {args.output}")


def write_scaling_data(output: Path) -> None:
    payload = {
        "families": {
            family_id: build_family_payload(family_id, metadata)
            for family_id, metadata in FAMILIES.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(f"{output.suffix}.tmp")
    temporary_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_output.replace(output)


def build_family_payload(family_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    path = metadata["best_runs"]
    results = read_best_runs(path)
    stale_results = [result for result in read_best_runs(path, include_stale=True) if result.stale]
    training_ratio = float(metadata["training_ratio"])
    if any(result.training_ratio != training_ratio for result in results):
        raise ValueError(f"{path}: every active run must use training_ratio={training_ratio:g}.")
    with path.open("rb") as handle:
        raw_runs = tomllib.load(handle)["runs"]

    result_flops = {result.budget: result.flops for result in results}
    loss_law = fit_loss_power_law((result_flops[r.budget], r.loss) for r in results)
    policy_law = fit_sigmoid_law((result_flops[r.budget], r.policy_top1) for r in results)
    params_law = fit_power_law((result_flops[r.budget], r.params) for r in results)
    samples_law = fit_power_law((result_flops[r.budget], r.samples_seen) for r in results)
    batch_size_law = fit_power_law((result_flops[r.budget], r.batch_size) for r in results)
    lr_law = fit_power_law((result_flops[r.budget], r.lr) for r in results)
    observed = [
        observed_point(result, result_flops[result.budget], raw_runs, family_id)
        for result in results
    ]
    stale_observed = [
        observed_point(result, result.flops, raw_runs, family_id) for result in stale_results
    ]
    target_width = max(result.d_model for result in results) * 2
    target_config = None
    if metadata.get("extrapolate", True):
        target_config = load_training_config(
            metadata["config"],
            d_model=target_width,
            training_ratio=training_ratio,
        )

    extrapolated = []
    target_flops = max(result_flops.values())
    if target_config is not None:
        target_flops_per_sample = measure_training_flops_per_sample(
            target_config.model,
            batch_size=target_config.run.batch_size,
        )
        target_flops = (
            target_flops_per_sample * target_config.run.batch_size * target_config.run.steps
        )
        extrapolated.append(
            extrapolated_recipe_point(
                target_config,
                target_flops,
                loss_law=loss_law,
                policy_law=policy_law,
            )
        )

    min_log_flops = math.log10(min(result_flops.values()))
    max_log_flops = math.log10(target_flops)
    curve_flops = [
        10 ** (min_log_flops + (max_log_flops - min_log_flops) * index / (CURVE_POINT_COUNT - 1))
        for index in range(CURVE_POINT_COUNT)
    ]
    curves = {
        "loss": curve(curve_flops, loss_law.predict),
        "policyTop1": curve(curve_flops, policy_law.predict),
        "params": curve(curve_flops, params_law.predict),
        "samples": curve(curve_flops, samples_law.predict),
        "samplesPerParam": curve(
            curve_flops,
            lambda flops: samples_law.predict(flops) / params_law.predict(flops),
        ),
        "lr": curve(curve_flops, lr_law.predict),
        "steps": curve(
            curve_flops,
            lambda flops: samples_law.predict(flops) / batch_size_law.predict(flops),
        ),
        "batchSize": curve(curve_flops, batch_size_law.predict),
    }
    return {
        "id": family_id,
        "name": metadata["name"],
        "description": metadata["description"],
        "trainingRatio": training_ratio,
        "observed": observed,
        "staleObserved": stale_observed,
        "extrapolated": extrapolated,
        "curves": curves,
    }


def observed_point(
    result: Any,
    flops: float,
    raw_runs: dict[str, Any],
    family_id: str,
) -> dict[str, Any]:
    raw_run = raw_runs[result.budget]
    throughput_data = _throughput_data(family_id)
    throughput = throughput_data["models"][f"d{result.d_model}"]
    gpu = throughput.get("gpu", throughput_data["sweep"]["gpu"])
    runtime_sec = result.samples_seen / result.batch_size * float(
        throughput["measured_wall_ms_per_step"]
    ) / 1000.0
    return {
        "name": _recipe_name(result.d_model, result.training_ratio),
        "sourceExperiment": str(raw_run["source_experiment"]),
        "modelKind": result.model_kind,
        "runName": result.run_name,
        "wandbUrl": result.wandb_url,
        "physicalFlops": flops,
        "dModel": result.d_model,
        "trainingRatio": result.training_ratio,
        "depth": result.depth,
        "batchSize": result.batch_size,
        "steps": result.samples_seen / result.batch_size,
        "lr": result.lr,
        "params": result.params,
        "samplesSeen": result.samples_seen,
        "samplesPerParam": result.samples_seen / result.params,
        "loss": result.loss,
        "policyTop1": result.policy_top1,
        "gpu": gpu,
        "runtimeSec": runtime_sec,
        "stale": result.stale,
    }


@cache
def _throughput_data(family_id: str) -> dict[str, Any]:
    path = Path(f"experiments/throughput-{family_id}.toml")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _recipe_name(d_model: int, training_ratio: float) -> str:
    return f"d{d_model}"


def extrapolated_recipe_point(
    config: Any,
    flops: float,
    *,
    loss_law: Any,
    policy_law: Any,
) -> dict[str, float | str]:
    params = model_parameter_count(config.model)
    samples = config.run.batch_size * config.run.steps
    return {
        "name": f"d{config.model.d_model}",
        "physicalFlops": flops,
        "params": params,
        "samplesSeen": samples,
        "samplesPerParam": samples / params,
        "loss": loss_law.predict(flops),
        "policyTop1": policy_law.predict(flops),
        "lr": config.optimizer.lr,
        "steps": config.run.steps,
        "batchSize": config.run.batch_size,
    }


def curve(flops_values: list[float], predict: Any) -> list[dict[str, float]]:
    return [
        {
            "physicalFlops": flops,
            "value": predict(flops),
        }
        for flops in flops_values
    ]


if __name__ == "__main__":
    export_scaling_data()
