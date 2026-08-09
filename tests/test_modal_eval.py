from pathlib import Path

from chess_engine_4.modal_eval import (
    OPENING_BOOK_PATH,
    _fastchess_command,
    _parse_fastchess_pair_scores,
    _parse_fastchess_pentanomial,
    _selfplay_command,
)


def test_fastchess_command_uses_randomized_paired_openings() -> None:
    payload = {
        "lc0_path": "/artifacts/bin/lc0",
        "games": 2,
        "rounds": 10,
        "concurrency": 1,
        "startup_ms": 120_000,
        "ping_ms": 120_000,
        "opening_seed": 12345,
        "candidate_name": "candidate",
        "candidate_weights": "/artifacts/models/candidate.safetensors",
        "candidate_backend": "ce4",
        "candidate_nodes": 100,
        "baseline_name": "baseline",
        "baseline_weights": "/artifacts/leela/baseline.pb.gz",
        "baseline_backend": "cudnn-fp16",
        "baseline_nodes": 100,
        "nodes": None,
        "tc": "1.0+0.01",
    }

    command = _fastchess_command(payload, Path("/artifacts/evals/test/games.pgn"))

    assert "-repeat" in command
    assert command[command.index("-srand") + 1] == "12345"
    openings = command.index("-openings")
    assert command[openings + 1 : openings + 4] == [
        f"file={OPENING_BOOK_PATH}",
        "format=pgn",
        "order=random",
    ]


def test_selfplay_command_configures_deterministic_low_visit_search() -> None:
    payload = {
        "lc0_path": "/artifacts/bin/lc0",
        "games": 64,
        "policy_mode_size": 64,
        "visits": 16,
        "parallelism": 32,
        "opening_book": "/uho.pgn",
        "opening_seed": 17,
        "player1": {"weights": "/dense.safetensors", "backend": "ce4"},
        "player2": {"weights": "/t74.pb.gz", "backend": "cudnn-fp16"},
    }

    command = _selfplay_command(payload, Path("/results.pgn"))

    assert "--visits=16" in command
    assert "--parallelism=32" in command
    assert "--no-share-trees" in command
    assert "--temperature=0.0" in command
    assert "--noise-epsilon=0.0" in command
    assert "--openings-pgn=/uho.pgn" in command
    assert "--openings-mode=shuffled" in command
    assert "--openings-seed=17" in command
    assert not any(argument.startswith("--openings-offset=") for argument in command)
    assert "--player1.backend=ce4" in command
    assert "--player1.backend-opts=max_batch=256,batch_wait_us=200" in command
    assert "--player2.backend-opts=child(backend=cudnn-fp16,max_batch=256,threads=1)" in command


def test_fastchess_pgn_parser_retains_pentanomial_pairs() -> None:
    pgn = """
[Round "1"]
[White "candidate"]
[Black "baseline"]
[Result "1-0"]

[Round "1"]
[White "baseline"]
[Black "candidate"]
[Result "1/2-1/2"]

[Round "2"]
[White "candidate"]
[Black "baseline"]
[Result "0-1"]

[Round "2"]
[White "baseline"]
[Black "candidate"]
[Result "1-0"]
"""

    assert _parse_fastchess_pentanomial(pgn, "candidate") == (1, 0, 0, 1, 0)
    assert _parse_fastchess_pair_scores(pgn, "candidate") == (3, 0)
