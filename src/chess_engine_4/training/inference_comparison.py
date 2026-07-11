"""Lightweight lc0 output collection and inference comparison."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MOVE_STATS_RE = re.compile(
    r"^info string [a-h][1-8][a-h][1-8][nbrq]?\s+"
    r"\(\s*(\d+)\s*\).*\(P:\s*([0-9.]+)%\)"
)
_ROOT_STATS_RE = re.compile(
    r"^info string node\s+\(\s*\d+\s*\).*"
    r"\(WL:\s*([-0-9.]+)\).*\(D:\s*([-0-9.]+)\)"
)


@dataclass(slots=True)
class UciPosition:
    initial_fen: str
    moves: tuple[str, ...]


@dataclass(slots=True)
class NetworkOutputs:
    policies: list[dict[int, float] | np.ndarray]
    q: np.ndarray
    d: np.ndarray


def evaluate_lc0(
    *,
    lc0_path: Path,
    weights_path: Path,
    positions: list[UciPosition],
    backend: str,
    env: dict[str, str],
) -> NetworkOutputs:
    command = [
        str(lc0_path),
        f"--weights={weights_path}",
        f"--backend={backend}",
        "--verbose-move-stats",
        "--minibatch-size=1",
        "--task-workers=0",
        "--threads=1",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("Failed to open lc0 pipes.")
    try:
        _send_lc0(process, "uci")
        _read_until(process, "uciok")
        policies: list[dict[int, float] | np.ndarray] = []
        q_values: list[float] = []
        d_values: list[float] = []
        for position in positions:
            _send_lc0(process, "ucinewgame")
            position_command = f"position fen {position.initial_fen}"
            if position.moves:
                position_command += " moves " + " ".join(position.moves)
            _send_lc0(process, position_command)
            _send_lc0(process, "go nodes 1")
            policy, q, d = _parse_lc0_evaluation(_read_until(process, "bestmove "))
            policies.append(policy)
            q_values.append(q)
            d_values.append(d)
        return NetworkOutputs(
            policies=policies,
            q=np.asarray(q_values, dtype=np.float32),
            d=np.asarray(d_values, dtype=np.float32),
        )
    finally:
        if process.poll() is None:
            _send_lc0(process, "quit")
            process.wait(timeout=30)


def compare_outputs(native: NetworkOutputs, exported: NetworkOutputs) -> dict[str, float | int]:
    if len(native.policies) != len(exported.policies):
        raise ValueError("Native and exported output counts differ.")
    policy_absolute_errors: list[float] = []
    policy_kls: list[float] = []
    top1_matches = 0
    for native_logits, exported_policy in zip(native.policies, exported.policies, strict=True):
        if not isinstance(exported_policy, dict):
            raise TypeError("Exported policies must be sparse dictionaries.")
        legal_indices = sorted(exported_policy)
        if not legal_indices:
            raise ValueError("lc0 returned no legal policy moves.")
        logits = np.asarray(native_logits, dtype=np.float64)[legal_indices]
        logits -= logits.max()
        native_policy = np.exp(logits)
        native_policy /= native_policy.sum()
        onnx_policy = np.asarray([exported_policy[index] for index in legal_indices])
        onnx_policy /= onnx_policy.sum()
        policy_absolute_errors.extend(np.abs(native_policy - onnx_policy).tolist())
        policy_kls.append(
            float(np.sum(native_policy * np.log(native_policy / np.clip(onnx_policy, 1e-12, None))))
        )
        native_top1 = legal_indices[int(native_policy.argmax())]
        exported_top1 = legal_indices[int(onnx_policy.argmax())]
        top1_matches += int(native_top1 == exported_top1)

    q_error = exported.q - native.q
    d_error = exported.d - native.d
    native_wdl = _wdl_from_q_d(native.q, native.d)
    exported_wdl = _wdl_from_q_d(exported.q, exported.d)
    return {
        "positions": len(native.policies),
        "policy_top1_agreement": top1_matches / len(native.policies),
        "policy_mae": float(np.mean(policy_absolute_errors)),
        "policy_max_abs_error": float(np.max(policy_absolute_errors)),
        "policy_kl_native_to_export": float(np.mean(policy_kls)),
        "q_mae": float(np.mean(np.abs(q_error))),
        "q_rmse": float(np.sqrt(np.mean(np.square(q_error)))),
        "q_max_abs_error": float(np.max(np.abs(q_error))),
        "draw_mae": float(np.mean(np.abs(d_error))),
        "draw_rmse": float(np.sqrt(np.mean(np.square(d_error)))),
        "draw_max_abs_error": float(np.max(np.abs(d_error))),
        "wdl_mae": float(np.mean(np.abs(exported_wdl - native_wdl))),
        "wdl_max_abs_error": float(np.max(np.abs(exported_wdl - native_wdl))),
    }


def _send_lc0(process: subprocess.Popen[str], command: str) -> None:
    assert process.stdin is not None
    process.stdin.write(command + "\n")
    process.stdin.flush()


def _read_until(process: subprocess.Popen[str], marker: str) -> list[str]:
    assert process.stdout is not None
    lines = []
    for line in process.stdout:
        stripped = line.strip()
        lines.append(stripped)
        if marker in stripped:
            return lines
    raise RuntimeError(
        f"lc0 exited before emitting {marker!r}; returncode={process.poll()}, tail={lines[-20:]}"
    )


def _parse_lc0_evaluation(lines: list[str]) -> tuple[dict[int, float], float, float]:
    policy: dict[int, float] = {}
    q = None
    d = None
    for line in lines:
        move_match = _MOVE_STATS_RE.match(line)
        if move_match:
            policy[int(move_match.group(1))] = float(move_match.group(2)) / 100.0
        root_match = _ROOT_STATS_RE.match(line)
        if root_match:
            q = float(root_match.group(1))
            d = float(root_match.group(2))
    if not policy or q is None or d is None:
        raise ValueError(f"Could not parse lc0 one-node output: {lines[-30:]}")
    return policy, q, d


def _wdl_from_q_d(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.stack(((1.0 - d + q) / 2.0, d, (1.0 - d - q) / 2.0), axis=-1)
