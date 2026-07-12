from pathlib import Path

from chess_engine_4.evaluation.round_robin import (
    Engine,
    MatchResult,
    build_match_payloads,
    fit_elos,
    load_roundrobin_config,
    parse_match_result,
)


def test_policy_config_expands_to_every_matchup() -> None:
    tournament, engines = load_roundrobin_config(Path("configs/eval/policy-elo.toml"))

    payloads = build_match_payloads(tournament, engines)

    assert len(engines) == 8
    assert len(payloads) == 28
    assert all(payload["visits"] is None for payload in payloads)
    assert all(payload["policy_mode_size"] == 32 for payload in payloads)


def test_parse_match_result_combines_both_colors() -> None:
    payload = {
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
    matches = [MatchResult("strong", "weak", 30, 5, 5, 1.0)]

    ratings = fit_elos(engines, matches)

    assert ratings[0]["name"] == "strong"
    assert ratings[0]["elo"] > ratings[1]["elo"]
