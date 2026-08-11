"""Adaptive Modal tournament orchestration and Elo fitting."""

from __future__ import annotations

import argparse
import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_engine_4.modal_eval import (
    OPENING_BOOK_PATH,
    app,
    lc0_path_for_gpu,
    selfplay_eval_function,
)

_RESULT_BLOCK = re.compile(
    r'\[White "(?P<white>[^"]+)"\]\s*'
    r'\[Black "(?P<black>[^"]+)"\]\s*'
    r'\[Results "(?P<white_wins>\d+) (?P<black_wins>\d+) (?P<draws>\d+)"\]'
)
_GAME_READY = re.compile(
    r"^gameready\b.*?\bgameid (?P<game_id>\d+)\b.*?"
    r"\bplayer1 (?P<player1>white|black)\b.*?"
    r"\bresult (?P<result>whitewon|blackwon|draw)\b",
    re.MULTILINE,
)
_ELO_SCALE = math.log(10.0) / 400.0
_SCHEDULE_PRIOR_STD = 400.0
_REPORT_PRIOR_STD = 10_000.0
_CI_LEVEL = 0.95
_CI_Z = 1.959963984540054


@dataclass(frozen=True, slots=True)
class Engine:
    name: str
    weights: str
    backend: str
    training_flops: float | None = None
    seed_elo: float = 0.0


@dataclass(frozen=True, slots=True)
class Tournament:
    name: str
    gpu: str
    games_per_matchup: int
    waves: int
    max_concurrency: int
    policy_mode_size: int
    parallelism: int = 1
    visits: int | None = None
    opening_book: str = str(OPENING_BOOK_PATH)
    opening_book_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class MatchResult:
    wave: int
    player1: str
    player2: str
    player1_wins: int
    player2_wins: int
    draws: int
    runtime_sec: float
    # Counts of mirrored opening-pair scores for player1: 0, 0.5, 1, 1.5, 2.
    # None identifies legacy or genuinely unpaired evidence.
    pentanomial: tuple[int, int, int, int, int] | None = None
    # Ordered half-point pair scores (0..4) by lc0 opening-pair game id.
    pair_scores: tuple[int, ...] | None = None
    # Stable identities for pair scores, including book, shuffle seed, and offset.
    opening_ids: tuple[str, ...] | None = None
    ce4_batch_stats: tuple[dict[str, Any], ...] | None = None


def eval_tournament_modal() -> None:
    parser = argparse.ArgumentParser(description="Run an adaptive lc0 tournament on Modal.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tournament, engines = load_tournament_config(args.config)
    matches = _load_completed_matches(args.output, tournament, engines)
    completed_waves = max((match.wave for match in matches), default=-1) + 1
    if completed_waves < tournament.waves:
        function = selfplay_eval_function(
            tournament.gpu,
            max_containers=tournament.max_concurrency,
        )
        with app.run():
            for wave in range(completed_waves, tournament.waves):
                pairings = select_pairings(
                    engines, matches, wave, tournament.games_per_matchup
                )
                payloads = build_match_payloads(tournament, pairings, wave)
                raw_results = list(function.map(payloads))
                matches.extend(
                    parse_match_result(payload, result)
                    for payload, result in zip(payloads, raw_results, strict=True)
                )
                report = build_report(tournament, engines, matches)
                _write_report(args.output, report)
                print(
                    f"completed wave {wave + 1}/{tournament.waves}: "
                    + ", ".join(f"{left.name} vs {right.name}" for left, right in pairings)
                )

    report = build_report(tournament, engines, matches)
    _write_report(args.output, report)
    print(json.dumps(report, indent=2))


def load_tournament_config(path: Path) -> tuple[Tournament, list[Engine]]:
    with path.open("rb") as config_file:
        config = tomllib.load(config_file)
    tournament = Tournament(**config["tournament"])
    engines = [Engine(**engine) for engine in config["engines"]]
    if len(engines) < 2:
        raise ValueError("A tournament requires at least two engines.")
    if len({engine.name for engine in engines}) != len(engines):
        raise ValueError("Engine names must be unique.")
    for engine in engines:
        if engine.backend not in {"ce4", "cudnn-fp16"}:
            raise ValueError(f"Engine {engine.name!r} has unsupported backend {engine.backend!r}.")
        if engine.backend == "ce4" and Path(engine.weights).suffix != ".safetensors":
            raise ValueError(f"Engine {engine.name!r} must use Safetensors with the ce4 backend.")
    if tournament.games_per_matchup <= 0 or tournament.games_per_matchup % 2:
        raise ValueError("games_per_matchup must be a positive even number.")
    if tournament.waves < 1:
        raise ValueError("waves must be positive.")
    return tournament, engines


def select_pairings(
    engines: list[Engine],
    matches: list[MatchResult],
    wave: int,
    games_per_matchup: int,
) -> list[tuple[Engine, Engine]]:
    if wave == 0:
        indices = [(index, index + 1) for index in range(0, len(engines) - 1, 2)]
    elif wave == 1:
        indices = [(index, index + 1) for index in range(1, len(engines) - 1, 2)]
        if len(engines) % 2 == 0:
            indices.append((len(engines) - 1, 0))
    else:
        ratings, covariance = _schedule_posterior(engines, matches)
        scores = {
            (left, right): _pair_information_gain(
                left,
                right,
                games_per_matchup,
                ratings,
                covariance,
            )
            for left in range(len(engines))
            for right in range(left + 1, len(engines))
        }
        indices = _maximum_weight_matching(tuple(range(len(engines))), scores)[1]
    return [(engines[left], engines[right]) for left, right in indices]


def build_match_payloads(
    tournament: Tournament,
    pairings: list[tuple[Engine, Engine]],
    wave: int,
) -> list[dict[str, Any]]:
    return [
        {
            "run_name": (
                f"{tournament.name}-w{wave + 1}-{player1.name}-vs-{player2.name}"
            ),
            "wave": wave,
            "games": tournament.games_per_matchup,
            "policy_mode_size": tournament.policy_mode_size,
            "visits": tournament.visits,
            "parallelism": tournament.parallelism,
            "opening_book": tournament.opening_book,
            "opening_book_sha256": tournament.opening_book_sha256,
            "gpu": tournament.gpu,
            "player1": asdict(player1),
            "player2": asdict(player2),
            "lc0_path": str(lc0_path_for_gpu(tournament.gpu)),
        }
        for player1, player2 in pairings
    ]


def parse_match_result(payload: dict[str, Any], result: dict[str, Any]) -> MatchResult:
    weights_to_name = {
        payload["player1"]["weights"]: payload["player1"]["name"],
        payload["player2"]["weights"]: payload["player2"]["name"],
    }
    scores = dict.fromkeys(weights_to_name.values(), 0)
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
    pair_scores = _parse_lc0_pair_scores(result.get("lc0_output", ""), games)
    pentanomial = _pentanomial(pair_scores) if pair_scores is not None else None
    opening_ids = (
        tuple(
            f"{payload.get('opening_book', OPENING_BOOK_PATH)}|"
            f"index={index}"
            for index in range(len(pair_scores))
        )
        if pair_scores is not None
        else None
    )
    if pentanomial is not None:
        paired_score = sum(index * count / 2 for index, count in enumerate(pentanomial))
        aggregate_score = scores[payload["player1"]["name"]] + 0.5 * draws
        if paired_score != aggregate_score:
            raise ValueError(
                "Lc0 per-game evidence disagrees with aggregate tournament results: "
                f"{paired_score} != {aggregate_score}."
            )
    return MatchResult(
        wave=int(payload["wave"]),
        player1=payload["player1"]["name"],
        player2=payload["player2"]["name"],
        player1_wins=scores[payload["player1"]["name"]],
        player2_wins=scores[payload["player2"]["name"]],
        draws=draws,
        runtime_sec=float(result["runtime_sec"]),
        pentanomial=pentanomial,
        pair_scores=pair_scores,
        opening_ids=opening_ids,
        ce4_batch_stats=tuple(result.get("ce4_batch_stats", ())) or None,
    )


def fit_elos(engines: list[Engine], matches: list[MatchResult]) -> list[dict[str, Any]]:
    if not _comparison_graph_connected(engines, matches):
        raise ValueError("Cannot fit Elo ratings for a disconnected comparison graph.")
    indices = {engine.name: index for index, engine in enumerate(engines)}
    transform = np.vstack(
        (np.eye(len(engines) - 1), -np.ones((1, len(engines) - 1)))
    )
    parameters = np.zeros(len(engines) - 1, dtype=np.float64)
    prior_precision = 1.0 / _REPORT_PRIOR_STD**2
    for _ in range(100):
        ratings = transform @ parameters
        gradient, hessian = _elo_derivatives(ratings, indices, matches)
        gradient += prior_precision * ratings
        hessian += np.eye(len(engines)) * prior_precision
        step = np.linalg.solve(transform.T @ hessian @ transform, transform.T @ gradient)
        parameters -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    ratings = transform @ parameters
    _, hessian = _elo_derivatives(ratings, indices, matches)
    hessian += np.eye(len(engines)) * prior_precision
    centered_covariance = _cluster_robust_covariance(
        ratings, indices, matches, hessian, transform
    )
    paired_matches = sum(
        match.pair_scores is not None or match.pentanomial is not None for match in matches
    )
    if paired_matches == len(matches):
        method = "paired-opening cluster-robust sandwich"
    elif paired_matches:
        method = "mixed paired-opening and unpaired cluster-robust sandwich"
    else:
        method = "unpaired game-level robust sandwich"
    rows = [
        {
            "name": engine.name,
            "elo": float(ratings[index]),
            "elo_95ci": float(
                _CI_Z * math.sqrt(max(centered_covariance[index, index], 0.0))
            ),
            "elo_95ci_low": float(
                ratings[index]
                - _CI_Z * math.sqrt(max(centered_covariance[index, index], 0.0))
            ),
            "elo_95ci_high": float(
                ratings[index]
                + _CI_Z * math.sqrt(max(centered_covariance[index, index], 0.0))
            ),
            "ci_method": method,
        }
        for index, engine in enumerate(engines)
    ]
    return sorted(rows, key=lambda row: float(row["elo"]), reverse=True)


def _parse_lc0_pair_scores(output: str, expected_games: int) -> tuple[int, ...] | None:
    """Recover mirrored-pair scores from lc0's per-game completion records."""
    if not output:
        return None
    scores: dict[int, float] = {}
    for match in _GAME_READY.finditer(output):
        game_id = int(match.group("game_id"))
        if game_id in scores:
            raise ValueError(f"Duplicate lc0 gameid {game_id} in tournament output.")
        result = match.group("result")
        if result == "draw":
            score = 0.5
        else:
            player1_won = (result == "whitewon") == (match.group("player1") == "white")
            score = 1.0 if player1_won else 0.0
        scores[game_id] = score
    if not scores:
        return None
    expected_ids = set(range(expected_games))
    if set(scores) != expected_ids:
        raise ValueError(
            "Incomplete lc0 per-game evidence: expected gameids "
            f"0..{expected_games - 1}, got {sorted(scores)}."
        )
    pair_scores = []
    for game_id in range(0, expected_games, 2):
        pair_score = scores[game_id] + scores[game_id + 1]
        pair_scores.append(int(pair_score * 2))
    return tuple(pair_scores)


def _pentanomial(pair_scores: tuple[int, ...]) -> tuple[int, int, int, int, int]:
    counts = [0, 0, 0, 0, 0]
    for score in pair_scores:
        counts[score] += 1
    return tuple(counts)  # type: ignore[return-value]


def _cluster_robust_covariance(
    ratings: np.ndarray,
    indices: dict[str, int],
    matches: list[MatchResult],
    hessian: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Return CR1 covariance, clustering mirrored games by opening pair."""
    parameter_count = len(ratings) - 1
    bread = np.linalg.inv(transform.T @ hessian @ transform)
    meat = np.zeros_like(bread)
    clusters = 0
    indexed_opening_gradients: dict[str, np.ndarray] = {}
    for match in matches:
        left = indices[match.player1]
        right = indices[match.player2]
        direction = np.zeros(len(ratings), dtype=np.float64)
        direction[left] = _ELO_SCALE
        direction[right] = -_ELO_SCALE
        reduced_direction = transform.T @ direction
        probability = _win_probability(ratings[left] - ratings[right])
        if match.pair_scores is not None:
            if len(match.pair_scores) * 2 != _match_games(match):
                raise ValueError(
                    f"Invalid pair scores for {match.player1} vs {match.player2}."
                )
            if any(score < 0 or score > 4 for score in match.pair_scores):
                raise ValueError(
                    f"Pair scores outside 0..4 for {match.player1} vs {match.player2}."
                )
            if match.opening_ids is not None and len(match.opening_ids) != len(
                match.pair_scores
            ):
                raise ValueError(
                    f"Opening identities disagree with pair scores for "
                    f"{match.player1} vs {match.player2}."
                )
            if (
                match.pentanomial is not None
                and _pentanomial(match.pair_scores) != match.pentanomial
            ):
                raise ValueError(
                    f"Pair scores disagree with pentanomial counts for "
                    f"{match.player1} vs {match.player2}."
                )
            for opening_index, score_index in enumerate(match.pair_scores):
                opening_id = (
                    match.opening_ids[opening_index]
                    if match.opening_ids is not None
                    else f"legacy-index={opening_index}"
                )
                residual = 2.0 * probability - 0.5 * score_index
                gradient = residual * reduced_direction
                indexed_opening_gradients[opening_id] = (
                    indexed_opening_gradients.get(
                        opening_id, np.zeros(parameter_count, dtype=np.float64)
                    )
                    + gradient
                )
        elif match.pentanomial is not None:
            if sum(match.pentanomial) * 2 != _match_games(match):
                raise ValueError(
                    f"Invalid pentanomial counts for {match.player1} vs {match.player2}."
                )
            for score_index, count in enumerate(match.pentanomial):
                residual = 2.0 * probability - 0.5 * score_index
                meat += count * residual**2 * np.outer(reduced_direction, reduced_direction)
                clusters += count
        else:
            outcomes = ((1.0, match.player1_wins), (0.5, match.draws), (0.0, match.player2_wins))
            for outcome, count in outcomes:
                residual = probability - outcome
                meat += count * residual**2 * np.outer(reduced_direction, reduced_direction)
                clusters += count
    for gradient in indexed_opening_gradients.values():
        meat += np.outer(gradient, gradient)
        clusters += 1
    if clusters > parameter_count:
        meat *= clusters / (clusters - parameter_count)
    reduced_covariance = bread @ meat @ bread
    return transform @ reduced_covariance @ transform.T


def _match_games(match: MatchResult) -> int:
    return match.player1_wins + match.player2_wins + match.draws


def _evidence_cluster_count(matches: list[MatchResult]) -> int:
    indexed_openings: set[str] = set()
    independent_clusters = 0
    for match in matches:
        if match.pair_scores is not None:
            if match.opening_ids is not None:
                indexed_openings.update(match.opening_ids)
            else:
                indexed_openings.update(
                    f"legacy-index={index}" for index in range(len(match.pair_scores))
                )
        elif match.pentanomial is not None:
            independent_clusters += sum(match.pentanomial)
        else:
            independent_clusters += _match_games(match)
    return len(indexed_openings) + independent_clusters


def _schedule_posterior(
    engines: list[Engine], matches: list[MatchResult]
) -> tuple[np.ndarray, np.ndarray]:
    indices = {engine.name: index for index, engine in enumerate(engines)}
    prior = np.asarray([engine.seed_elo for engine in engines], dtype=np.float64)
    prior -= prior.mean()
    ratings = prior.copy()
    precision = 1.0 / _SCHEDULE_PRIOR_STD**2
    for _ in range(100):
        gradient, hessian = _elo_derivatives(ratings, indices, matches)
        gradient += precision * (ratings - prior)
        hessian += np.eye(len(engines)) * precision
        step = np.linalg.solve(hessian, gradient)
        ratings -= step
        ratings -= ratings.mean()
        if np.max(np.abs(step)) < 1e-8:
            break
    _, hessian = _elo_derivatives(ratings, indices, matches)
    hessian += np.eye(len(engines)) * precision
    covariance = np.linalg.inv(hessian)
    center = np.eye(len(engines)) - np.ones_like(hessian) / len(engines)
    return ratings, center @ covariance @ center


def _pair_information_gain(
    left: int,
    right: int,
    games: int,
    ratings: np.ndarray,
    covariance: np.ndarray,
) -> float:
    direction = np.zeros(len(ratings), dtype=np.float64)
    direction[left] = 1.0
    direction[right] = -1.0
    probability = _win_probability(ratings[left] - ratings[right])
    information = games * _ELO_SCALE**2 * probability * (1.0 - probability)
    projected = covariance @ direction
    return float(
        information * np.dot(projected, projected)
        / (1.0 + information * np.dot(direction, projected))
    )


def _maximum_weight_matching(
    remaining: tuple[int, ...], scores: dict[tuple[int, int], float]
) -> tuple[float, list[tuple[int, int]]]:
    if len(remaining) < 2:
        return 0.0, []
    first = remaining[0]
    best_score = -math.inf
    best_pairs: list[tuple[int, int]] = []
    if len(remaining) % 2:
        best_score, best_pairs = _maximum_weight_matching(remaining[1:], scores)
    for offset, second in enumerate(remaining[1:], start=1):
        rest = remaining[1:offset] + remaining[offset + 1 :]
        tail_score, tail_pairs = _maximum_weight_matching(rest, scores)
        pair = (min(first, second), max(first, second))
        score = scores[pair] + tail_score
        if score > best_score:
            best_score = score
            best_pairs = [pair, *tail_pairs]
    return best_score, best_pairs


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
        probability = _win_probability(ratings[left] - ratings[right])
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
    engines: list[Engine], ratings: list[dict[str, Any]]
) -> dict[str, float] | None:
    rating_by_name = {str(row["name"]): float(row["elo"]) for row in ratings}
    points = [
        (engine.training_flops, rating_by_name[engine.name])
        for engine in engines
        if engine.training_flops is not None
    ]
    if len(points) < 2 or len({flops for flops, _ in points}) < 2:
        return None
    flops, elo = (np.asarray(values, dtype=np.float64) for values in zip(*points, strict=True))
    slope, intercept = np.polyfit(np.log10(flops), elo, 1)
    return {"elo_per_decade": float(slope), "intercept": float(intercept)}


def build_report(
    tournament: Tournament,
    engines: list[Engine],
    matches: list[MatchResult],
) -> dict[str, Any]:
    ratings = fit_elos(engines, matches) if _comparison_graph_connected(engines, matches) else []
    paired_matches = sum(
        match.pair_scores is not None or match.pentanomial is not None for match in matches
    )
    return {
        "tournament": asdict(tournament),
        "opening_book": tournament.opening_book,
        "opening_book_sha256": tournament.opening_book_sha256,
        "engines": [asdict(engine) for engine in engines],
        "completed_waves": max((match.wave for match in matches), default=-1) + 1,
        "matches": [asdict(match) for match in matches],
        "ratings": ratings,
        "rating_model": {
            "point_estimate": "Bradley-Terry score quasi-MLE with draws worth 0.5",
            "regularization": f"centered Normal(0, {_REPORT_PRIOR_STD:g} Elo) stabilization prior",
            "confidence_level": _CI_LEVEL,
            "interval": "two-sided Wald interval",
            "covariance": (
                "CR1 sandwich; mirrored games and the same indexed opening reused across "
                "matchups are clustered when ordered pair scores are retained; aggregate "
                "pentanomial pairs are clustered within matchup; otherwise games are unpaired"
            ),
            "paired_matches": paired_matches,
            "unpaired_matches": len(matches) - paired_matches,
            "clusters": _evidence_cluster_count(matches),
            "free_rating_parameters": len(engines) - 1,
        },
        "flops_scaling_law": fit_flops_scaling_law(engines, ratings) if ratings else None,
        "total_games": sum(
            _match_games(match) for match in matches
        ),
        "total_gpu_runtime_sec": sum(match.runtime_sec for match in matches),
    }


def _comparison_graph_connected(engines: list[Engine], matches: list[MatchResult]) -> bool:
    adjacency = {engine.name: set() for engine in engines}
    for match in matches:
        adjacency[match.player1].add(match.player2)
        adjacency[match.player2].add(match.player1)
    visited: set[str] = set()
    pending = [engines[0].name]
    while pending:
        name = pending.pop()
        if name not in visited:
            visited.add(name)
            pending.extend(adjacency[name] - visited)
    return len(visited) == len(engines)


def _win_probability(rating_delta: float) -> float:
    logit = max(-50.0, min(50.0, _ELO_SCALE * rating_delta))
    return 1.0 / (1.0 + math.exp(-logit))


def _load_completed_matches(
    output: Path,
    tournament: Tournament,
    engines: list[Engine],
) -> list[MatchResult]:
    if not output.exists():
        return []
    payload = json.loads(output.read_text())
    if payload["tournament"] != asdict(tournament):
        raise ValueError("Existing tournament output does not match the tournament config.")
    if payload["engines"] != [asdict(engine) for engine in engines]:
        raise ValueError("Existing tournament output does not match the configured engines.")
    return [
        MatchResult(
            **{
                **match,
                "pentanomial": (
                    tuple(match["pentanomial"])
                    if match.get("pentanomial") is not None
                    else None
                ),
                "pair_scores": (
                    tuple(match["pair_scores"])
                    if match.get("pair_scores") is not None
                    else None
                ),
                "opening_ids": (
                    tuple(match["opening_ids"])
                    if match.get("opening_ids") is not None
                    else None
                ),
                "ce4_batch_stats": (
                    tuple(match["ce4_batch_stats"])
                    if match.get("ce4_batch_stats") is not None
                    else None
                ),
            }
        )
        for match in payload["matches"]
    ]


def _write_report(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
