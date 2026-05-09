"""Modal evaluation entrypoint using lc0 and fastchess."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

APP_NAME = "chess-engine-4-eval"
ARTIFACT_VOLUME_NAME = "chess-engine-4-artifacts"
REMOTE_ARTIFACT_PATH = "/artifacts"
REMOTE_LEELA_PATH = Path(REMOTE_ARTIFACT_PATH) / "leela"
REMOTE_EVAL_PATH = Path(REMOTE_ARTIFACT_PATH) / "evals"
BT4_URL = "https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1740.pb.gz"
BT4_REMOTE_PATH = REMOTE_LEELA_PATH / "BT4-1740.pb.gz"

GPU_CHOICES = {
    "any": "any",
    "t4": "T4",
    "l4": "L4",
    "a10g": "A10G",
    "a100-40gb": "A100-40GB",
    "a100-80gb": "A100-80GB",
    "l40s": "L40S",
    "h100": "H100",
    "h200": "H200",
    "b200": "B200",
}

ORT_VERSION = "1.23.2"
LC0_VERSION = "0.32.1"
FASTCHESS_VERSION = "1.8.0-alpha"
FASTCHESS_URL = (
    f"https://github.com/Disservin/fastchess/releases/download/v{FASTCHESS_VERSION}/"
    "fastchess-linux-x86-64.tar"
)
DEFAULT_LC0_REMOTE_PATH = Path(REMOTE_ARTIFACT_PATH) / "bin" / "lc0"
RUNTIME_LIBRARY_PATH = "/opt/onnxruntime/lib:/usr/local/cuda/lib64"

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
        f"git clone --depth 1 --branch v{LC0_VERSION} --recurse-submodules "
        "https://github.com/LeelaChessZero/lc0.git /opt/lc0",
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
    parser.add_argument("--gpu", default="l4", choices=sorted(GPU_CHOICES))
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
    parser.add_argument("--candidate-backend", default="onnx-cuda")
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

    train_function = _remote_function_for_gpu(args.gpu)
    with app.run():
        result = train_function.remote(payload)
    print(result["stdout"])
    print(f"pgn_path={result['pgn_path']}")
    print(f"log_path={result['log_path']}")


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
    gpu="any",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval_any(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


@app.function(
    image=image,
    gpu="T4",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval_t4(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


@app.function(
    image=image,
    gpu="L4",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval_l4(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


@app.function(
    image=image,
    gpu="A10G",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval_a10g(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval_a100_40gb(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval_a100_80gb(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


@app.function(
    image=image,
    gpu="L40S",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval_l40s(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


@app.function(
    image=image,
    gpu="H100",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval_h100(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


@app.function(
    image=image,
    gpu="H200",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval_h200(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


@app.function(
    image=image,
    gpu="B200",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=24 * 60 * 60,
)
def _eval_b200(payload: dict[str, Any]) -> dict[str, str]:
    return _run_eval_remote(payload)


def _remote_function_for_gpu(gpu: str) -> modal.Function:
    return {
        "any": _eval_any,
        "t4": _eval_t4,
        "l4": _eval_l4,
        "a10g": _eval_a10g,
        "a100-40gb": _eval_a100_40gb,
        "a100-80gb": _eval_a100_80gb,
        "l40s": _eval_l40s,
        "h100": _eval_h100,
        "h200": _eval_h200,
        "b200": _eval_b200,
    }[gpu]
