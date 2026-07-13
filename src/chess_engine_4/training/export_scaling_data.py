"""Export canonical scaling-law data for the static website."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path
from typing import Any

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
        "version": 2,
        "families": {
            family_id: build_family_payload(family_id, metadata)
            for family_id, metadata in FAMILIES.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_family_payload(family_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    path = metadata["best_runs"]
    results = read_best_runs(path)
    stale_results = [result for result in read_best_runs(path, include_stale=True) if result.stale]
    with path.open("rb") as handle:
        raw_runs = tomllib.load(handle)["runs"]

    result_flops = {result.budget: physical_flops(result) for result in results}
    stale_flops = {result.budget: physical_flops(result) for result in stale_results}
    loss_law = fit_loss_power_law((result_flops[r.budget], r.loss) for r in results)
    policy_law = fit_sigmoid_law((result_flops[r.budget], r.policy_top1) for r in results)
    params_law = fit_power_law((result_flops[r.budget], r.params) for r in results)
    samples_law = fit_power_law((result_flops[r.budget], r.samples_seen) for r in results)
    batch_size_law = fit_power_law((result_flops[r.budget], r.batch_size) for r in results)
    lr_law = fit_power_law((result_flops[r.budget], r.lr) for r in results)

    observed = [observed_point(result, result_flops[result.budget], raw_runs) for result in results]
    stale_observed = [
        observed_point(result, stale_flops[result.budget], raw_runs) for result in stale_results
    ]

    frontier_exponent = round(math.log10(max(result_flops.values())))
    target_flops = [10.0 ** (frontier_exponent + offset) for offset in (1, 2)]
    extrapolated = [
        extrapolated_point(
            flops,
            loss_law=loss_law,
            policy_law=policy_law,
            params_law=params_law,
            samples_law=samples_law,
            batch_size_law=batch_size_law,
            lr_law=lr_law,
        )
        for flops in target_flops
    ]

    min_log_flops = math.log10(min(result_flops.values()))
    max_display_flops = max([target_flops[-1], *stale_flops.values()])
    max_log_flops = math.log10(max_display_flops)
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
        "observed": observed,
        "staleObserved": stale_observed,
        "extrapolated": extrapolated,
        "curves": curves,
    }


def observed_point(result: Any, flops: float, raw_runs: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": f"d{result.d_model}",
        "sourceExperiment": str(raw_runs[result.budget]["source_experiment"]),
        "modelKind": result.model_kind,
        "runName": result.run_name,
        "wandbUrl": result.wandb_url,
        "physicalFlops": flops,
        "dModel": result.d_model,
        "depth": result.depth,
        "batchSize": result.batch_size,
        "steps": result.samples_seen / result.batch_size,
        "lr": result.lr,
        "params": result.params,
        "samplesSeen": result.samples_seen,
        "samplesPerParam": result.samples_seen / result.params,
        "loss": result.loss,
        "policyTop1": result.policy_top1,
        "runtimeSec": float(raw_runs[result.budget]["runtime_sec"]),
    }


def extrapolated_point(
    flops: float,
    *,
    loss_law: Any,
    policy_law: Any,
    params_law: Any,
    samples_law: Any,
    batch_size_law: Any,
    lr_law: Any,
) -> dict[str, float | str]:
    params = params_law.predict(flops)
    samples = samples_law.predict(flops)
    return {
        "name": f"1e{round(math.log10(flops))} FLOPs",
        "physicalFlops": flops,
        "params": params,
        "samplesSeen": samples,
        "samplesPerParam": samples / params,
        "loss": loss_law.predict(flops),
        "policyTop1": policy_law.predict(flops),
        "lr": lr_law.predict(flops),
        "steps": samples / batch_size_law.predict(flops),
        "batchSize": batch_size_law.predict(flops),
    }


def curve(flops_values: list[float], predict: Any) -> list[dict[str, float]]:
    return [
        {
            "physicalFlops": flops,
            "value": predict(flops),
        }
        for flops in flops_values
    ]


def physical_flops(result: Any) -> float:
    steps = result.samples_seen / result.batch_size
    return result.compute / steps


if __name__ == "__main__":
    export_scaling_data()
