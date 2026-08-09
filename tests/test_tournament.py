import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from chess_engine_4.evaluation.tournament import (
    Engine,
    MatchResult,
    Tournament,
    _load_completed_matches,
    build_match_payloads,
    fit_elos,
    load_tournament_config,
    parse_match_result,
    select_pairings,
)


def test_policy_config_builds_connected_opening_waves() -> None:
    tournament, engines = load_tournament_config(Path("configs/eval/policy-elo.toml"))

    first = select_pairings(engines, [], 0, tournament.games_per_matchup)
    second = select_pairings(engines, [], 1, tournament.games_per_matchup)
    payloads = build_match_payloads(tournament, first, 0)

    assert len(engines) == 14
    assert len(first) == len(second) == 7
    assert all(payload["visits"] is None for payload in payloads)
    assert all(payload["policy_mode_size"] == 256 for payload in payloads)
    assert all(
        payload["opening_book"].endswith("UHO_Lichess_4852_v1-random-65536.pgn")
        for payload in payloads
    )
    assert 0 <= tournament.opening_seed <= 2_147_483_647
    assert all(payload["opening_seed"] == tournament.opening_seed for payload in payloads)
    assert all("opening_offset" not in payload for payload in payloads)


def test_adaptive_wave_pairs_each_engine_once() -> None:
    engines = [Engine(f"e{index}", f"/{index}", "cuda") for index in range(6)]
    matches = [
        MatchResult(0, "e0", "e1", 20, 8, 4, 1.0),
        MatchResult(0, "e2", "e3", 18, 10, 4, 1.0),
        MatchResult(0, "e4", "e5", 16, 12, 4, 1.0),
        MatchResult(1, "e1", "e2", 17, 11, 4, 1.0),
        MatchResult(1, "e3", "e4", 17, 11, 4, 1.0),
        MatchResult(1, "e5", "e0", 8, 20, 4, 1.0),
    ]

    pairings = select_pairings(engines, matches, 2, 32)

    assert len(pairings) == 3
    assert len({engine.name for pair in pairings for engine in pair}) == 6


def test_parse_match_result_combines_both_colors() -> None:
    payload = {
        "wave": 0,
        "games": 8,
        "player1": {"name": "alpha", "weights": "/alpha.pb.gz"},
        "player2": {"name": "beta", "weights": "/beta.pb.gz"},
    }
    result = {
        "runtime_sec": 1.5,
        "results": """
[White "/alpha.pb.gz"]
[Black "/beta.pb.gz"]
[Results "3 0 1"]

[White "/beta.pb.gz"]
[Black "/alpha.pb.gz"]
[Results "1 2 1"]
""",
    }

    match = parse_match_result(payload, result)

    assert match.player1_wins == 5
    assert match.player2_wins == 1
    assert match.draws == 2
    assert match.pentanomial is None


def test_parse_match_result_retains_lc0_opening_pairs() -> None:
    payload = {
        "wave": 0,
        "games": 4,
        "opening_book": "/uho.pgn",
        "opening_seed": 7,
        "player1": {"name": "alpha", "weights": "/alpha"},
        "player2": {"name": "beta", "weights": "/beta"},
    }
    result = {
        "runtime_sec": 1.0,
        "results": """
[White "/alpha"]
[Black "/beta"]
[Results "1 0 1"]
[White "/beta"]
[Black "/alpha"]
[Results "0 1 1"]
""",
        "lc0_output": """
gameready gameid 3 play_start_ply 4 player1 black result draw
gameready gameid 0 play_start_ply 4 player1 white result whitewon
gameready gameid 2 play_start_ply 4 player1 white result draw
gameready gameid 1 play_start_ply 4 player1 black result blackwon
""",
    }

    match = parse_match_result(payload, result)

    assert match.pentanomial == (0, 0, 1, 0, 1)
    assert match.pair_scores == (4, 2)
    assert match.opening_ids == (
        "/uho.pgn|seed=7|index=0",
        "/uho.pgn|seed=7|index=1",
    )


def test_fit_elos_orders_engines_by_score() -> None:
    engines = [Engine("strong", "/strong", "cuda"), Engine("weak", "/weak", "cuda")]
    matches = [MatchResult(0, "strong", "weak", 30, 5, 5, 1.0)]

    ratings = fit_elos(engines, matches)

    assert ratings[0]["name"] == "strong"
    assert ratings[0]["elo"] > ratings[1]["elo"]
    assert math.isfinite(ratings[0]["elo_95ci"])


def test_fit_elos_uses_paired_evidence_for_balanced_results() -> None:
    engines = [Engine("alpha", "/alpha", "cuda"), Engine("beta", "/beta", "cuda")]
    pair_scores = (0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4)
    paired = MatchResult(
        0, "alpha", "beta", 10, 10, 20, 1.0, (2, 3, 10, 3, 2), pair_scores
    )

    rating = fit_elos(engines, [paired])[0]

    assert rating["elo"] == pytest.approx(0.0)
    assert rating["elo_95ci"] > 0.0
    assert rating["ci_method"] == "paired-opening cluster-robust sandwich"


def test_paired_draw_heavy_evidence_does_not_claim_independent_games() -> None:
    engines = [Engine("alpha", "/alpha", "cuda"), Engine("beta", "/beta", "cuda")]
    paired = MatchResult(0, "alpha", "beta", 2, 2, 36, 1.0, (0, 0, 20, 0, 0))
    unpaired = MatchResult(0, "alpha", "beta", 2, 2, 36, 1.0)

    paired_rating = fit_elos(engines, [paired])[0]
    unpaired_rating = fit_elos(engines, [unpaired])[0]

    assert paired_rating["elo_95ci"] == pytest.approx(0.0)
    assert unpaired_rating["elo_95ci"] > 0.0
    assert unpaired_rating["ci_method"] == "unpaired game-level robust sandwich"


def test_same_indexed_openings_are_clustered_across_matchups() -> None:
    engines = [Engine("alpha", "/alpha", "cuda"), Engine("beta", "/beta", "cuda")]
    pair_scores = (0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4)
    pentanomial = (2, 3, 10, 3, 2)
    raw_matches = [
        MatchResult(wave, "alpha", "beta", 10, 10, 20, 1.0, pentanomial, pair_scores)
        for wave in range(2)
    ]
    aggregate_matches = [
        MatchResult(wave, "alpha", "beta", 10, 10, 20, 1.0, pentanomial)
        for wave in range(2)
    ]

    raw_width = fit_elos(engines, raw_matches)[0]["elo_95ci"]
    aggregate_width = fit_elos(engines, aggregate_matches)[0]["elo_95ci"]

    assert raw_width > aggregate_width


def test_fit_elos_handles_decisive_and_degenerate_results() -> None:
    engines = [Engine("alpha", "/alpha", "cuda"), Engine("beta", "/beta", "cuda")]
    decisive = MatchResult(0, "alpha", "beta", 30, 5, 5, 1.0, (0, 2, 2, 5, 11))
    swept = MatchResult(0, "alpha", "beta", 40, 0, 0, 1.0, (0, 0, 0, 0, 20))

    decisive_rating = fit_elos(engines, [decisive])[0]
    swept_rating = fit_elos(engines, [swept])[0]

    assert decisive_rating["elo"] > 0.0
    assert math.isfinite(decisive_rating["elo_95ci"])
    assert swept_rating["elo"] > decisive_rating["elo"]
    assert math.isfinite(swept_rating["elo"])


def test_fit_elos_rejects_disconnected_comparison_graph() -> None:
    engines = [
        Engine("alpha", "/alpha", "cuda"),
        Engine("beta", "/beta", "cuda"),
        Engine("isolated", "/isolated", "cuda"),
    ]

    with pytest.raises(ValueError, match="disconnected"):
        fit_elos(engines, [MatchResult(0, "alpha", "beta", 1, 1, 2, 1.0)])


def test_resume_loads_legacy_and_paired_match_evidence(tmp_path: Path) -> None:
    tournament = Tournament("test", "L4", 4, 2, 1, 1)
    engines = [
        Engine("alpha", "/alpha.safetensors", "ce4"),
        Engine("beta", "/beta.safetensors", "ce4"),
    ]
    output = tmp_path / "results.json"
    match = {
        "wave": 0,
        "player1": "alpha",
        "player2": "beta",
        "player1_wins": 1,
        "player2_wins": 1,
        "draws": 2,
        "runtime_sec": 1.0,
    }
    payload = {
        "tournament": asdict(tournament),
        "engines": [asdict(engine) for engine in engines],
        "matches": [match],
    }
    output.write_text(json.dumps(payload))

    assert _load_completed_matches(output, tournament, engines)[0].pentanomial is None

    payload["matches"][0]["pentanomial"] = [0, 0, 2, 0, 0]
    payload["matches"][0]["pair_scores"] = [2, 2]
    output.write_text(json.dumps(payload))

    assert _load_completed_matches(output, tournament, engines)[0].pentanomial == (
        0,
        0,
        2,
        0,
        0,
    )
    assert _load_completed_matches(output, tournament, engines)[0].pair_scores == (2, 2)
