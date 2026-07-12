"""Export canonical scaling-law data for the static website."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path
from typing import Any

from chess_engine_4.training.scaling_laws import (
    fit_linear_law,
    fit_power_law,
    fit_scaling_laws,
    read_best_runs,
)

DEFAULT_OUTPUT = Path("website/src/generated/scaling-laws.json")
CURVE_POINT_COUNT = 61
FAMILIES = {
    "dense": {
        "name": "Dense",
        "description": "Single-token dense SwiGLU network trained on lc0 planes.",
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
        "version": 1,
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
    with path.open("rb") as handle:
        raw_runs = tomllib.load(handle)["runs"]

    laws = fit_scaling_laws(results)
    physical_flops_law = fit_power_law(
        (result.compute, physical_flops(result.compute, result.samples_seen, result.batch_size))
        for result in results
    )
    policy_law = fit_linear_law((result.compute, result.policy_top1) for result in results)

    observed = [
        {
            "budget": result.budget,
            "sourceExperiment": str(raw_runs[result.budget]["source_experiment"]),
            "modelKind": result.model_kind,
            "runName": result.run_name,
            "wandbUrl": result.wandb_url,
            "compute": result.compute,
            "physicalFlops": physical_flops(
                result.compute,
                result.samples_seen,
                result.batch_size,
            ),
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
        for result in results
    ]

    frontier_exponent = round(math.log10(max(result.compute for result in results)))
    target_computes = [10.0 ** (frontier_exponent + offset) for offset in (1, 2)]
    extrapolated = [
        extrapolated_point(
            compute,
            laws=laws,
            physical_flops_law=physical_flops_law,
            policy_law=policy_law,
        )
        for compute in target_computes
    ]

    min_log_compute = math.log10(min(result.compute for result in results))
    max_log_compute = math.log10(target_computes[-1])
    curve_computes = [
        10
        ** (min_log_compute + (max_log_compute - min_log_compute) * index / (CURVE_POINT_COUNT - 1))
        for index in range(CURVE_POINT_COUNT)
    ]
    curves = {
        "loss": curve(
            curve_computes,
            physical_flops_law,
            laws.loss.predict,
        ),
        "policyTop1": curve(curve_computes, physical_flops_law, policy_law.predict),
        "params": curve(curve_computes, physical_flops_law, laws.params.predict),
        "samples": curve(curve_computes, physical_flops_law, laws.samples.predict),
        "samplesPerParam": curve(
            curve_computes,
            physical_flops_law,
            lambda compute: laws.samples.predict(compute) / laws.params.predict(compute),
        ),
        "lr": curve(curve_computes, physical_flops_law, laws.lr.predict),
        "steps": curve(
            curve_computes,
            physical_flops_law,
            lambda compute: laws.samples.predict(compute) / laws.batch_size.predict(compute),
        ),
        "batchSize": curve(curve_computes, physical_flops_law, laws.batch_size.predict),
    }
    return {
        "id": family_id,
        "name": metadata["name"],
        "description": metadata["description"],
        "observed": observed,
        "extrapolated": extrapolated,
        "curves": curves,
    }


def extrapolated_point(
    compute: float,
    *,
    laws: Any,
    physical_flops_law: Any,
    policy_law: Any,
) -> dict[str, float | str]:
    params = laws.params.predict(compute)
    samples = laws.samples.predict(compute)
    return {
        "budget": f"1e{round(math.log10(compute))}",
        "compute": compute,
        "physicalFlops": physical_flops_law.predict(compute),
        "params": params,
        "samplesSeen": samples,
        "samplesPerParam": samples / params,
        "loss": laws.loss.predict(compute),
        "policyTop1": policy_law.predict(compute),
        "lr": laws.lr.predict(compute),
        "steps": samples / laws.batch_size.predict(compute),
        "batchSize": laws.batch_size.predict(compute),
    }


def curve(computes: list[float], physical_flops_law: Any, predict: Any) -> list[dict[str, float]]:
    return [
        {
            "compute": compute,
            "physicalFlops": physical_flops_law.predict(compute),
            "value": predict(compute),
        }
        for compute in computes
    ]


def physical_flops(compute: float, samples_seen: int, batch_size: int) -> float:
    steps = samples_seen / batch_size
    return compute / steps


if __name__ == "__main__":
    export_scaling_data()
