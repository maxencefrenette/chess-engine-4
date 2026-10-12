"""Benchmark tournament engines through Modal."""

import argparse
import json
from pathlib import Path

import modal

from chess_engine_4.backends.modal.eval import app, backendbench_function, lc0_path_for_gpu
from chess_engine_4.evaluation.backendbench import parse_backendbench_result
from chess_engine_4.evaluation.tournament import load_tournament_config


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
            "lc0_path": str(lc0_path_for_gpu(tournament.gpu)),
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
