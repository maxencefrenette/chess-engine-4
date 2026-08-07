"""Benchmark every engine in a tournament config with lc0 backendbench."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import modal

from chess_engine_4.evaluation.tournament import load_tournament_config
from chess_engine_4.modal_eval import LC0_REMOTE_PATH, app, backendbench_function

_RESULT_ROW = re.compile(
    r"^\s*(?P<batch_size>\d+),\s*"
    r"(?P<nodes_per_sec>[\d.]+),\s*"
    r"(?P<mean_ms>[\d.]+),",
    re.MULTILINE,
)


def benchmark_tournament_modal() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark every engine in an lc0 tournament config."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    if args.batches <= 0:
        parser.error("--batches must be positive.")

    tournament, engines = load_tournament_config(args.config)
    payloads = [
        {
            "name": engine.name,
            "weights": engine.weights,
            "backend": engine.backend,
            "batch_size": tournament.policy_mode_size,
            "batches": args.batches,
            "lc0_path": str(LC0_REMOTE_PATH),
        }
        for engine in engines
    ]
    function = backendbench_function(
        tournament.gpu,
        max_containers=tournament.max_concurrency,
    )
    with modal.enable_output(), app.run():
        raw_results = list(function.map(payloads))

    results = {
        "gpu": tournament.gpu,
        "batch_size": tournament.policy_mode_size,
        "batches": args.batches,
        "engines": [parse_backendbench_result(result) for result in raw_results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def parse_backendbench_result(result: dict[str, Any]) -> dict[str, Any]:
    match = _RESULT_ROW.search(str(result["output"]))
    if match is None:
        raise ValueError(f"Could not parse backendbench output:\n{result['output']}")
    return {
        "name": str(result["name"]),
        "weights": str(result["weights"]),
        "backend": str(result["backend"]),
        "nodes_per_sec": float(match.group("nodes_per_sec")),
        "mean_ms": float(match.group("mean_ms")),
    }
