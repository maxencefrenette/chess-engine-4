from pathlib import Path

from chess_engine_4.evaluation.tournament import (
    Engine,
    MatchResult,
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

    assert len(engines) == 8
    assert len(first) == len(second) == 4
    assert all(payload["visits"] is None for payload in payloads)
    assert all(payload["policy_mode_size"] == 32 for payload in payloads)


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


def test_fit_elos_orders_engines_by_score() -> None:
    engines = [Engine("strong", "/strong", "cuda"), Engine("weak", "/weak", "cuda")]
    matches = [MatchResult(0, "strong", "weak", 30, 5, 5, 1.0)]

    ratings = fit_elos(engines, matches)

    assert ratings[0]["name"] == "strong"
    assert ratings[0]["elo"] > ratings[1]["elo"]
