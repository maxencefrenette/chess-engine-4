"""Parallel Modal round-robin orchestration and Elo fitting."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_engine_4.modal_eval import (
    DEFAULT_LC0_REMOTE_PATH,
    LC0_COMMIT,
    POLICY_OPENING_BOOK_PATH,
    app,
    selfplay_eval_function,
)

_RESULT_BLOCK = re.compile(
    r'\[White "(?P<white>[^"]+)"\]\s*'
    r'\[Black "(?P<black>[^"]+)"\]\s*'
    r'\[Results "(?P<white_wins>\d+) (?P<black_wins>\d+) (?P<draws>\d+)"\]'
)
_ELO_SCALE = math.log(10.0) / 400.0


@dataclass(frozen=True, slots=True)
class Engine:
    name: str
    weights: str
    backend: str
    training_flops: float | None = None


@dataclass(frozen=True, slots=True)
class Tournament:
    name: str
    gpu: str
    games_per_matchup: int
    max_concurrency: int
    policy_mode_size: int
    parallelism: int = 1
    visits: int | None = None


@dataclass(frozen=True, slots=True)
class MatchResult:
    player1: str
    player2: str
    player1_wins: int
    player2_wins: int
    draws: int
    runtime_sec: float


def eval_roundrobin_modal() -> None:
    parser = argparse.ArgumentParser(description="Run a parallel lc0 round robin on Modal.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tournament, engines = load_roundrobin_config(args.config)
    payloads = build_match_payloads(tournament, engines)
    function = selfplay_eval_function(
        tournament.gpu,
        max_containers=tournament.max_concurrency,
    )
    with app.run():
        raw_results = list(function.map(payloads))
    matches = [
        parse_match_result(payload, result)
        for payload, result in zip(payloads, raw_results, strict=True)
    ]
    ratings = fit_elos(engines, matches)
    report = {
        "tournament": asdict(tournament),
        "lc0_commit": LC0_COMMIT,
        "opening_book": str(POLICY_OPENING_BOOK_PATH),
        "engines": [asdict(engine) for engine in engines],
        "matches": [asdict(match) for match in matches],
        "ratings": ratings,
        "flops_scaling_law": fit_flops_scaling_law(engines, ratings),
        "total_games": sum(
            match.player1_wins + match.player2_wins + match.draws for match in matches
        ),
        "total_gpu_runtime_sec": sum(match.runtime_sec for match in matches),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def load_roundrobin_config(path: Path) -> tuple[Tournament, list[Engine]]:
    with path.open("rb") as config_file:
        config = tomllib.load(config_file)
    tournament = Tournament(**config["tournament"])
    engines = [Engine(**engine) for engine in config["engines"]]
    if len(engines) < 2:
        raise ValueError("A round robin requires at least two engines.")
    if len({engine.name for engine in engines}) != len(engines):
        raise ValueError("Engine names must be unique.")
    if tournament.games_per_matchup <= 0 or tournament.games_per_matchup % 2:
        raise ValueError("games_per_matchup must be a positive even number.")
    return tournament, engines


def build_match_payloads(tournament: Tournament, engines: list[Engine]) -> list[dict[str, Any]]:
    return [
        {
            "run_name": f"{tournament.name}-{player1.name}-vs-{player2.name}",
            "games": tournament.games_per_matchup,
            "policy_mode_size": tournament.policy_mode_size,
            "visits": tournament.visits,
            "parallelism": tournament.parallelism,
            "gpu": tournament.gpu,
            "player1": asdict(player1),
            "player2": asdict(player2),
            "lc0_path": str(DEFAULT_LC0_REMOTE_PATH),
        }
        for player1, player2 in itertools.combinations(engines, 2)
    ]


def parse_match_result(payload: dict[str, Any], result: dict[str, Any]) -> MatchResult:
    weights_to_name = {
        payload["player1"]["weights"]: payload["player1"]["name"],
        payload["player2"]["weights"]: payload["player2"]["name"],
    }
    scores = {name: 0 for name in weights_to_name.values()}
    draws = 0
    games = 0
    for block in _RESULT_BLOCK.finditer(result["results"]):
        white = weights_to_name[block.group("white")]
        black = weights_to_name[block.group("black")]
        white_wins = int(block.group("white_wins"))
        black_wins = int(block.group("black_wins"))
        block_draws = int(block.group("draws"))
        scores[white] += white_wins
        scores[black] += black_wins
        draws += block_draws
        games += white_wins + black_wins + block_draws
    if games != payload["games"]:
        raise ValueError(f"Expected {payload['games']} games, parsed {games}.")
    return MatchResult(
        player1=payload["player1"]["name"],
        player2=payload["player2"]["name"],
        player1_wins=scores[payload["player1"]["name"]],
        player2_wins=scores[payload["player2"]["name"]],
        draws=draws,
        runtime_sec=float(result["runtime_sec"]),
    )


def fit_elos(engines: list[Engine], matches: list[MatchResult]) -> list[dict[str, float | str]]:
    indices = {engine.name: index for index, engine in enumerate(engines)}
    ratings = np.zeros(len(engines), dtype=np.float64)
    for _ in range(100):
        gradient, hessian = _elo_derivatives(ratings, indices, matches)
        step = np.linalg.solve(
            hessian[:-1, :-1] + np.eye(len(engines) - 1) * 1e-9,
            gradient[:-1],
        )
        ratings[:-1] -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    ratings -= ratings.mean()
    _, hessian = _elo_derivatives(ratings, indices, matches)
    covariance = np.zeros_like(hessian)
    covariance[:-1, :-1] = np.linalg.inv(hessian[:-1, :-1] + np.eye(len(engines) - 1) * 1e-9)
    center = np.eye(len(engines)) - np.ones_like(hessian) / len(engines)
    centered_covariance = center @ covariance @ center
    rows = [
        {
            "name": engine.name,
            "elo": float(ratings[index]),
            "elo_95ci": float(1.96 * math.sqrt(max(centered_covariance[index, index], 0.0))),
        }
        for index, engine in enumerate(engines)
    ]
    return sorted(rows, key=lambda row: float(row["elo"]), reverse=True)


def _elo_derivatives(
    ratings: np.ndarray,
    indices: dict[str, int],
    matches: list[MatchResult],
) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.zeros_like(ratings)
    hessian = np.zeros((len(ratings), len(ratings)), dtype=np.float64)
    for match in matches:
        left = indices[match.player1]
        right = indices[match.player2]
        games = match.player1_wins + match.player2_wins + match.draws
        score = match.player1_wins + 0.5 * match.draws
        probability = 1.0 / (1.0 + math.exp(-_ELO_SCALE * (ratings[left] - ratings[right])))
        error = games * probability - score
        gradient[left] += _ELO_SCALE * error
        gradient[right] -= _ELO_SCALE * error
        curvature = games * _ELO_SCALE**2 * probability * (1.0 - probability)
        hessian[left, left] += curvature
        hessian[right, right] += curvature
        hessian[left, right] -= curvature
        hessian[right, left] -= curvature
    return gradient, hessian


def fit_flops_scaling_law(
    engines: list[Engine], ratings: list[dict[str, float | str]]
) -> dict[str, float] | None:
    rating_by_name = {str(row["name"]): float(row["elo"]) for row in ratings}
    points = [
        (engine.training_flops, rating_by_name[engine.name])
        for engine in engines
        if engine.training_flops is not None
    ]
    if len(points) < 2:
        return None
    flops, elo = (np.asarray(values, dtype=np.float64) for values in zip(*points, strict=True))
    slope, intercept = np.polyfit(np.log10(flops), elo, 1)
    return {"elo_per_decade": float(slope), "intercept": float(intercept)}
