"""Modal evaluation entrypoint using lc0 and fastchess."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "chess-engine-4-eval"
ARTIFACT_VOLUME_NAME = "chess-engine-4-artifacts"
REMOTE_ARTIFACT_PATH = "/artifacts"
REMOTE_LEELA_PATH = Path(REMOTE_ARTIFACT_PATH) / "leela"
REMOTE_EVAL_PATH = Path(REMOTE_ARTIFACT_PATH) / "evals"
OPENING_BOOK_PATH = Path(REMOTE_ARTIFACT_PATH) / "books" / "noob_2moves.epd"
POLICY_OPENING_BOOK_PATH = Path(REMOTE_ARTIFACT_PATH) / "books" / "noob_2moves.pgn"
OPENING_BOOK_SEED = 1
BT4_URL = "https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1740.pb.gz"
BT4_REMOTE_PATH = REMOTE_LEELA_PATH / "BT4-1740.pb.gz"

ORT_VERSION = "1.23.2"
LC0_COMMIT = "d8ce48258c39d331c119f8c8729374ceb3df8409"
FASTCHESS_VERSION = "1.8.0-alpha"
FASTCHESS_URL = (
    f"https://github.com/Disservin/fastchess/releases/download/v{FASTCHESS_VERSION}/"
    "fastchess-linux-x86-64.tar"
)
DEFAULT_LC0_REMOTE_PATH = Path(REMOTE_ARTIFACT_PATH) / "bin" / "lc0"
RUNTIME_LIBRARY_PATH = (
    "/opt/onnxruntime/lib:"
    "/usr/local/lib/python3.12/site-packages/tensorrt_libs:"
    "/usr/local/cuda/lib64"
)

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install(
        "ca-certificates",
        "curl",
        "libgomp1",
        "libopenblas0-pthread",
        "libprotobuf32t64",
        "unzip",
        "zlib1g",
    )
    .pip_install(
        "numpy>=2.2",
        "python-dotenv>=1.2.2",
        "tensorrt-cu12==10.13.3.9",
    )
    .run_commands(
        "find /usr/local/lib/python3.12/site-packages -type d -name '*libs' "
        "> /etc/ld.so.conf.d/python-wheel-libs.conf && ldconfig"
    )
    .run_commands(
        f"curl -L https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/"
        f"onnxruntime-linux-x64-gpu-{ORT_VERSION}.tgz | tar -xz -C /opt",
        f"mv /opt/onnxruntime-linux-x64-gpu-{ORT_VERSION} /opt/onnxruntime",
        f"curl -L {FASTCHESS_URL} | tar -x -C /opt",
        "install -m 755 /opt/fastchess-linux-x86-64/fastchess /usr/local/bin/fastchess",
    )
    .env({"LD_LIBRARY_PATH": RUNTIME_LIBRARY_PATH})
)

lc0_builder_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install(
        "ca-certificates",
        "curl",
        "g++",
        "git",
        "libopenblas-dev",
        "libprotobuf-dev",
        "meson",
        "ninja-build",
        "pkg-config",
        "protobuf-compiler",
        "zlib1g-dev",
    )
    .run_commands(
        f"curl -L https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/"
        f"onnxruntime-linux-x64-gpu-{ORT_VERSION}.tgz | tar -xz -C /opt",
        f"mv /opt/onnxruntime-linux-x64-gpu-{ORT_VERSION} /opt/onnxruntime",
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/LeelaChessZero/lc0.git /opt/lc0",
        f"cd /opt/lc0 && git checkout {LC0_COMMIT} && "
        "git submodule update --init --recursive --depth 1",
        "cd /opt/lc0 && ./build.sh release -Dgtest=false -Donnx=true "
        "-Donnx_libdir=/opt/onnxruntime/lib -Donnx_include=/opt/onnxruntime/include "
        "-Dnative_arch=false -Dnative_cuda=false -Ddefault_backend=cuda",
        "install -m 755 /opt/lc0/build/release/lc0 /usr/local/bin/lc0",
    )
    .env({"LD_LIBRARY_PATH": RUNTIME_LIBRARY_PATH})
)


def prepare_lc0_modal() -> None:
    parser = argparse.ArgumentParser(
        description="Build lc0 once on Modal and cache the Linux binary in the artifacts Volume."
    )
    parser.add_argument("--output", default=str(DEFAULT_LC0_REMOTE_PATH))
    args = parser.parse_args()

    with app.run():
        result = _prepare_lc0_remote.remote(args.output)
    print(f"lc0_path={result['lc0_path']}")
    print(result["version"])


def eval_modal() -> None:
    parser = argparse.ArgumentParser(description="Run an lc0-vs-lc0 fastchess match on Modal.")
    parser.add_argument("candidate_weights", type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--tc", default="1.0+0.01")
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--candidate-nodes", type=int, default=None)
    parser.add_argument("--baseline-nodes", type=int, default=None)
    parser.add_argument("--startup-ms", type=int, default=120_000)
    parser.add_argument("--ping-ms", type=int, default=120_000)
    parser.add_argument("--candidate-backend", default="onnx-trt")
    parser.add_argument("--baseline-backend", default="cuda")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--baseline-name", default="BT4-1740")
    parser.add_argument(
        "--baseline-weights",
        default=str(BT4_REMOTE_PATH),
        help="Remote Modal Volume path to the baseline lc0 weights file.",
    )
    parser.add_argument(
        "--baseline-url",
        default=BT4_URL,
        help="Optional URL to download the baseline weights if missing.",
    )
    parser.add_argument(
        "--lc0-path",
        default=str(DEFAULT_LC0_REMOTE_PATH),
        help="Remote Modal Volume path to the lc0 binary built by prepare-lc0-modal.",
    )
    args = parser.parse_args()

    if not args.candidate_weights.exists():
        raise FileNotFoundError(args.candidate_weights)

    run_name = args.name or args.candidate_weights.stem
    remote_candidate = REMOTE_LEELA_PATH / args.candidate_weights.name
    _upload_candidate(args.candidate_weights, remote_candidate)

    payload = {
        "run_name": run_name,
        "candidate_weights": str(remote_candidate),
        "games": args.games,
        "rounds": args.rounds,
        "concurrency": args.concurrency,
        "tc": args.tc,
        "nodes": args.nodes,
        "candidate_nodes": args.candidate_nodes,
        "baseline_nodes": args.baseline_nodes,
        "startup_ms": args.startup_ms,
        "ping_ms": args.ping_ms,
        "candidate_backend": args.candidate_backend,
        "baseline_backend": args.baseline_backend,
        "candidate_name": args.candidate_name,
        "baseline_name": args.baseline_name,
        "baseline_weights": args.baseline_weights,
        "baseline_url": args.baseline_url,
        "lc0_path": args.lc0_path,
    }

    with app.run():
        result = _eval.remote(payload)
    print(result["stdout"])
    print(f"pgn_path={result['pgn_path']}")
    print(f"log_path={result['log_path']}")


def eval_selfplay_modal() -> None:
    parser = argparse.ArgumentParser(description="Run a batched lc0 self-play match on Modal.")
    parser.add_argument("player1_weights")
    parser.add_argument("player2_weights")
    parser.add_argument("--name", required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--policy-mode-size", type=int, default=256)
    parser.add_argument("--visits", type=int, default=None)
    parser.add_argument("--parallelism", type=int, default=32)
    parser.add_argument("--gpu", choices=("T4", "L4", "B200"), default="T4")
    parser.add_argument("--player1-backend", default="onnx-trt")
    parser.add_argument("--player2-backend", default="cuda")
    parser.add_argument("--lc0-path", default=str(DEFAULT_LC0_REMOTE_PATH))
    args = parser.parse_args()
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number.")
    payload = {
        "run_name": args.name,
        "games": args.games,
        "policy_mode_size": args.policy_mode_size,
        "visits": args.visits,
        "parallelism": args.parallelism,
        "gpu": args.gpu,
        "player1": {
            "weights": args.player1_weights,
            "backend": args.player1_backend,
        },
        "player2": {
            "weights": args.player2_weights,
            "backend": args.player2_backend,
        },
        "lc0_path": args.lc0_path,
    }
    eval_function = selfplay_eval_function(args.gpu)
    with app.run():
        result = eval_function.remote(payload)
    summary = {key: value for key, value in result.items() if key != "lc0_output"}
    print(json.dumps(summary, indent=2))


def _upload_candidate(local_path: Path, remote_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "put",
            "--force",
            ARTIFACT_VOLUME_NAME,
            str(local_path),
            _volume_relative_path(remote_path),
        ],
        check=True,
    )


def _volume_relative_path(mounted_path: Path) -> str:
    mounted = str(mounted_path)
    prefix = REMOTE_ARTIFACT_PATH.rstrip("/") + "/"
    if mounted == REMOTE_ARTIFACT_PATH:
        return "/"
    if mounted.startswith(prefix):
        return mounted.removeprefix(prefix)
    return mounted


def _run_eval_remote(payload: dict[str, Any]) -> dict[str, str]:
    _require_lc0(payload)
    if not OPENING_BOOK_PATH.exists():
        raise FileNotFoundError(f"Missing evaluation opening book: {OPENING_BOOK_PATH}")
    REMOTE_LEELA_PATH.mkdir(parents=True, exist_ok=True)
    REMOTE_EVAL_PATH.mkdir(parents=True, exist_ok=True)
    baseline_weights = Path(payload["baseline_weights"])
    if not baseline_weights.exists():
        baseline_url = payload.get("baseline_url")
        if not baseline_url:
            raise FileNotFoundError(baseline_weights)
        _download_file(baseline_url, baseline_weights)
        artifact_volume.commit()

    run_dir = REMOTE_EVAL_PATH / str(payload["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = run_dir / "games.pgn"
    log_path = run_dir / "fastchess.log"
    command = _fastchess_command(payload, pgn_path)
    command.extend(["-log", f"file={log_path}", "level=info", "engine=true", "append=false"])
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "LD_LIBRARY_PATH": RUNTIME_LIBRARY_PATH},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fastchess failed with exit code "
            f"{completed.returncode}\ncommand={' '.join(command)}\n{completed.stdout}"
        )
    artifact_volume.commit()
    return {"stdout": completed.stdout, "pgn_path": str(pgn_path), "log_path": str(log_path)}


def _run_selfplay_eval_remote(payload: dict[str, Any]) -> dict[str, Any]:
    _require_lc0(payload)
    if not POLICY_OPENING_BOOK_PATH.exists():
        raise FileNotFoundError(f"Missing policy opening book: {POLICY_OPENING_BOOK_PATH}")
    for player in (payload["player1"], payload["player2"]):
        if not Path(player["weights"]).exists():
            raise FileNotFoundError(player["weights"])
    run_dir = REMOTE_EVAL_PATH / str(payload["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "selfplay-results.pgn"
    results_path.unlink(missing_ok=True)
    command = _selfplay_command(payload, results_path)
    started_at = time.monotonic()
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "LD_LIBRARY_PATH": RUNTIME_LIBRARY_PATH},
    )
    runtime_sec = time.monotonic() - started_at
    if completed.returncode != 0:
        raise RuntimeError(
            "lc0 selfplay failed with exit code "
            f"{completed.returncode}\ncommand={' '.join(command)}\n{completed.stdout}"
        )
    if not results_path.exists():
        raise RuntimeError(
            "lc0 selfplay completed without writing tournament results.\n"
            f"command={' '.join(command)}\n{completed.stdout}"
        )
    result = {
        "gpu": payload["gpu"],
        "games": payload["games"],
        "runtime_sec": runtime_sec,
        "games_per_sec": payload["games"] / runtime_sec,
        "results_path": str(results_path),
        "results": results_path.read_text(),
        "lc0_output": completed.stdout,
    }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    artifact_volume.commit()
    return result


def _selfplay_command(payload: dict[str, Any], results_path: Path) -> list[str]:
    command = [
        payload["lc0_path"],
        "selfplay",
        f"--games={payload['games']}",
        f"--openings-pgn={POLICY_OPENING_BOOK_PATH}",
        "--mirror-openings",
        "--openings-mode=sequential",
        f"--tournament-results-file={results_path}",
        f"--player1.weights={payload['player1']['weights']}",
        f"--player2.weights={payload['player2']['weights']}",
    ]
    if payload["visits"] is None:
        command.extend(
            [
                "--parallelism=1",
                f"--policy-mode-size={payload['policy_mode_size']}",
                f"--player1.backend={payload['player1']['backend']}",
                f"--player2.backend={payload['player2']['backend']}",
            ]
        )
    else:
        command.extend(
            [
                f"--parallelism={payload['parallelism']}",
                f"--visits={payload['visits']}",
                "--no-share-trees",
                "--temperature=0.0",
                "--noise-epsilon=0.0",
                "--player1.backend=multiplexing",
                "--player1.backend-opts=child(backend="
                f"{payload['player1']['backend']},max_batch=256,threads=1)",
                "--player2.backend=multiplexing",
                "--player2.backend-opts=child(backend="
                f"{payload['player2']['backend']},max_batch=256,threads=1)",
            ]
        )
    return command


def _require_lc0(payload: dict[str, Any]) -> None:
    lc0_path = Path(payload["lc0_path"])
    if lc0_path.exists():
        return
    raise FileNotFoundError(
        f"Missing lc0 binary at {lc0_path}. Run `uv run prepare-lc0-modal` first."
    )


def _download_file(url: str, path: Path) -> None:
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "chess-engine-4/0.1"})
    with urllib.request.urlopen(request) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output)


@app.function(
    image=lc0_builder_image,
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=3 * 60 * 60,
)
def _prepare_lc0_remote(output: str) -> dict[str, str]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2("/usr/local/bin/lc0", output_path)
    output_path.chmod(0o755)
    artifact_volume.commit()

    completed = subprocess.run(
        [str(output_path), "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "LD_LIBRARY_PATH": RUNTIME_LIBRARY_PATH},
    )
    return {"lc0_path": str(output_path), "version": completed.stdout.strip()}


def _fastchess_command(payload: dict[str, Any], pgn_path: Path) -> list[str]:
    lc0_path = payload["lc0_path"]
    command = [
        "fastchess",
        "-repeat",
        "-games",
        str(payload["games"]),
        "-rounds",
        str(payload["rounds"]),
        "-concurrency",
        str(payload["concurrency"]),
        "-srand",
        str(OPENING_BOOK_SEED),
        "-openings",
        f"file={OPENING_BOOK_PATH}",
        "format=epd",
        "order=random",
        "-report",
        "penta=true",
        "-startup-ms",
        str(payload["startup_ms"]),
        "-ping-ms",
        str(payload["ping_ms"]),
        "-pgnout",
        f"file={pgn_path}",
        "nodes=true",
        "nps=true",
        "timeleft=true",
        "latency=true",
        "-each",
        "proto=uci",
        "option.Threads=1",
        "-engine",
        f"name={payload['candidate_name']}",
        f"cmd={lc0_path}",
        "dir=/tmp",
        _engine_limit_flag(payload, "candidate"),
        f"option.WeightsFile={payload['candidate_weights']}",
        f"option.Backend={payload['candidate_backend']}",
        "-engine",
        f"name={payload['baseline_name']}",
        f"cmd={lc0_path}",
        "dir=/tmp",
        _engine_limit_flag(payload, "baseline"),
        f"option.WeightsFile={payload['baseline_weights']}",
        f"option.Backend={payload['baseline_backend']}",
    ]
    return [item for item in command if item]


def _engine_limit_flag(payload: dict[str, Any], engine: str) -> str:
    engine_nodes = payload.get(f"{engine}_nodes")
    if engine_nodes:
        return f"nodes={engine_nodes}"
    if payload.get("nodes"):
        return f"nodes={payload['nodes']}"
    return f"tc={payload['tc']}"


@app.function(
    image=image,
    gpu="B200",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


def selfplay_eval_function(gpu: str, *, max_containers: int | None = None) -> modal.Function:
    options: dict[str, Any] = {
        "image": image,
        "gpu": gpu,
        "volumes": {REMOTE_ARTIFACT_PATH: artifact_volume},
        "timeout": 24 * 60 * 60,
        "name": f"selfplay_eval_{gpu.lower()}",
    }
    if max_containers is not None:
        options["max_containers"] = max_containers
    return app.function(
        **options,
    )(_run_selfplay_eval_remote)
