"""Combine the main and adaptive UHO tournament evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from chess_engine_4.evaluation.tournament import (
    _CI_Z,
    _REPORT_PRIOR_STD,
    Engine,
    MatchResult,
    _cluster_robust_covariance,
    _elo_derivatives,
    _evidence_cluster_count,
    fit_elos,
)

HERE = Path(__file__).parent
MAIN = HERE / "tournament-uho-extended-results.json"
EXTRA = HERE / "tournament-uho-extra-results.json"
OUTPUT = HERE / "tournament-uho-combined-results.json"
RANDOM_SOURCES = (
    HERE / "tournament-random-rerun-results.json",
    HERE / "tournament-random-rerun-half-quarter-results.json",
    HERE / "tournament-random-rerun-full-quarter-results.json",
)
RANDOM_OUTPUT = HERE / "tournament-random-rerun-combined-results.json"
PROBE_RUNTIME_SEC = 33.08899232
GPU_CPU_RATE = 0.000842 + 2 * 0.0000131


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-rerun", action="store_true")
    if parser.parse_args().random_rerun:
        analyze_random_rerun()
        return

    main_report = json.loads(MAIN.read_text())
    extra_report = json.loads(EXTRA.read_text())
    engines = [Engine(**row) for row in main_report["engines"]]
    matches = [_match(row) for row in main_report["matches"]]
    matches.append(_match(extra_report["matches"][0], wave=3))
    ratings, covariance = _fit_with_covariance(engines, matches)

    contrasts = _contrasts(
        engines,
        ratings,
        covariance,
        (
            ("retention-0.25", "retention-0.5"),
            ("retention-0.5", "retention-1.0"),
            ("retention-0.25", "retention-1.0"),
        ),
    )

    tournament_runtime = float(main_report["total_gpu_runtime_sec"]) + float(
        extra_report["total_gpu_runtime_sec"]
    )
    report = {
        "sources": [MAIN.name, EXTRA.name],
        "opening_book": main_report["opening_book"],
        "opening_book_sha256": main_report["opening_book_sha256"],
        "opening_seed": 1,
        "opening_ranges": [
            {"offset": 32, "pairs": 736, "matchups": 3},
            {"offset": 768, "pairs": 568, "matchups": 1},
        ],
        "ratings": ratings,
        "pairwise_contrasts": contrasts,
        "clusters": _evidence_cluster_count(matches),
        "total_games": sum(
            match.player1_wins + match.player2_wins + match.draws for match in matches
        ),
        "tournament_runtime_sec": tournament_runtime,
        "probe_runtime_sec": PROBE_RUNTIME_SEC,
        "gpu_cpu_rate_usd_per_sec": GPU_CPU_RATE,
        "tournament_cost_usd": tournament_runtime * GPU_CPU_RATE,
        "total_including_probe_cost_usd":
            (tournament_runtime + PROBE_RUNTIME_SEC) * GPU_CPU_RATE,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def analyze_random_rerun() -> None:
    reports = [json.loads(path.read_text()) for path in RANDOM_SOURCES]
    engines = [Engine(**row) for row in reports[0]["engines"]]
    matches = [_match(row) for report in reports for row in report["matches"]]
    ratings, covariance = _fit_with_covariance(engines, matches)
    runtime = sum(float(report["total_gpu_runtime_sec"]) for report in reports)
    report = {
        "sources": [path.name for path in RANDOM_SOURCES],
        "opening_seeds": [report["opening_seed"] for report in reports],
        "ratings": ratings,
        "pairwise_contrasts": _contrasts(
            engines,
            ratings,
            covariance,
            (
                ("random-retention-0.25", "random-retention-0.5"),
                ("random-retention-0.5", "random-retention-1.0"),
                ("random-retention-0.25", "random-retention-1.0"),
            ),
        ),
        "clusters": _evidence_cluster_count(matches),
        "total_games": sum(
            match.player1_wins + match.player2_wins + match.draws for match in matches
        ),
        "tournament_runtime_sec": runtime,
        "gpu_cpu_rate_usd_per_sec": GPU_CPU_RATE,
        "tournament_cost_usd": runtime * GPU_CPU_RATE,
    }
    RANDOM_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def _contrasts(
    engines: list[Engine],
    ratings: list[dict[str, object]],
    covariance: np.ndarray,
    pairs: tuple[tuple[str, str], ...],
) -> list[dict[str, object]]:
    indices = {engine.name: index for index, engine in enumerate(engines)}
    rating_by_name = {str(row["name"]): float(row["elo"]) for row in ratings}
    contrasts = []
    for left, right in pairs:
        direction = np.zeros(len(engines))
        direction[indices[left]] = 1.0
        direction[indices[right]] = -1.0
        difference = rating_by_name[left] - rating_by_name[right]
        half_width = _CI_Z * math.sqrt(float(direction @ covariance @ direction))
        contrasts.append(
            {
                "left": left,
                "right": right,
                "elo_difference": difference,
                "elo_95ci": half_width,
                "elo_95ci_low": difference - half_width,
                "elo_95ci_high": difference + half_width,
            }
        )
    return contrasts


def _match(row: dict[str, object], wave: int | None = None) -> MatchResult:
    payload = dict(row)
    if wave is not None:
        payload["wave"] = wave
    for key in ("pentanomial", "pair_scores", "opening_ids", "ce4_batch_stats"):
        if payload.get(key) is not None:
            payload[key] = tuple(payload[key])
    return MatchResult(**payload)  # type: ignore[arg-type]


def _fit_with_covariance(
    engines: list[Engine], matches: list[MatchResult]
) -> tuple[list[dict[str, object]], np.ndarray]:
    indices = {engine.name: index for index, engine in enumerate(engines)}
    transform = np.vstack(
        (np.eye(len(engines) - 1), -np.ones((1, len(engines) - 1)))
    )
    parameters = np.zeros(len(engines) - 1)
    precision = 1.0 / _REPORT_PRIOR_STD**2
    for _ in range(100):
        ratings = transform @ parameters
        gradient, hessian = _elo_derivatives(ratings, indices, matches)
        gradient += precision * ratings
        hessian += np.eye(len(engines)) * precision
        step = np.linalg.solve(transform.T @ hessian @ transform, transform.T @ gradient)
        parameters -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    ratings = transform @ parameters
    _, hessian = _elo_derivatives(ratings, indices, matches)
    hessian += np.eye(len(engines)) * precision
    covariance = _cluster_robust_covariance(
        ratings, indices, matches, hessian, transform
    )
    fitted = fit_elos(engines, matches)
    for row in fitted:
        row["ci_method"] = "pentanomial cluster-robust sandwich"
    return fitted, covariance


if __name__ == "__main__":
    main()
