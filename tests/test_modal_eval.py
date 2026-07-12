from pathlib import Path

from chess_engine_4.modal_eval import OPENING_BOOK_PATH, _fastchess_command, _selfplay_command


def test_fastchess_command_uses_fixed_paired_openings() -> None:
    payload = {
        "lc0_path": "/artifacts/bin/lc0",
        "games": 2,
        "rounds": 10,
        "concurrency": 1,
        "startup_ms": 120_000,
        "ping_ms": 120_000,
        "candidate_name": "candidate",
        "candidate_weights": "/artifacts/leela/candidate.pb.gz",
        "candidate_backend": "onnx-trt",
        "candidate_nodes": 100,
        "baseline_name": "baseline",
        "baseline_weights": "/artifacts/leela/baseline.pb.gz",
        "baseline_backend": "onnx-trt",
        "baseline_nodes": 100,
        "nodes": None,
        "tc": "1.0+0.01",
    }

    command = _fastchess_command(payload, Path("/artifacts/evals/test/games.pgn"))

    assert "-repeat" in command
    assert command[command.index("-srand") + 1] == "1"
    openings = command.index("-openings")
    assert command[openings + 1 : openings + 4] == [
        f"file={OPENING_BOOK_PATH}",
        "format=epd",
        "order=random",
    ]


def test_selfplay_command_configures_deterministic_low_visit_search() -> None:
    payload = {
        "lc0_path": "/artifacts/bin/lc0",
        "games": 64,
        "policy_mode_size": 64,
        "visits": 16,
        "parallelism": 32,
        "player1": {"weights": "/dense.pb.gz", "backend": "onnx-trt"},
        "player2": {"weights": "/t74.pb.gz", "backend": "cuda"},
    }

    command = _selfplay_command(payload, Path("/results.pgn"))

    assert "--visits=16" in command
    assert "--parallelism=32" in command
    assert "--no-share-trees" in command
    assert "--temperature=0.0" in command
    assert "--noise-epsilon=0.0" in command
    assert "--player1.backend-opts=child(backend=onnx-trt,max_batch=256,threads=1)" in command
    assert "--player2.backend-opts=child(backend=cuda,max_batch=256,threads=1)" in command
