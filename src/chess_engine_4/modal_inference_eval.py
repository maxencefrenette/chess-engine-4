"""Modal command for measuring native-training versus lc0 inference drift."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import modal

from chess_engine_4.modal_eval import (
    DEFAULT_LC0_REMOTE_PATH,
    RUNTIME_LIBRARY_PATH,
)
from chess_engine_4.modal_eval import (
    image as lc0_image,
)
from chess_engine_4.modal_train import (
    REMOTE_ARTIFACT_PATH,
    REMOTE_DATA_PATH,
    artifact_volume,
    data_volume,
)
from chess_engine_4.modal_train import (
    image as training_image,
)

APP_NAME = "chess-engine-4-inference-eval"
REMOTE_EVAL_PATH = Path(REMOTE_ARTIFACT_PATH) / "evals" / "inference-mismatch"

app = modal.App(APP_NAME)


def eval_inference_modal() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a checkpoint's native TE outputs with lc0 ONNX exports."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weights", type=Path, nargs="+")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", default="onnx-trt")
    parser.add_argument("--name", default=None)
    parser.add_argument("--lc0-path", type=Path, default=DEFAULT_LC0_REMOTE_PATH)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    run_name = args.name or args.checkpoint.stem
    payload = {
        "checkpoint": str(_mounted_artifact_path(args.checkpoint)),
        "weights": [str(_mounted_artifact_path(path)) for path in args.weights],
        "samples": args.samples,
        "seed": args.seed,
        "backend": args.backend,
        "run_name": run_name,
        "lc0_path": str(_mounted_artifact_path(args.lc0_path)),
    }
    with modal.enable_output(), app.run():
        intermediates = _evaluate_native.remote(payload)
        result = _evaluate_exports.remote({**payload, **intermediates})
    print(json.dumps(result, indent=2, sort_keys=True))


def _run_native_evaluation(payload: dict[str, Any]) -> dict[str, str]:
    import numpy as np

    from chess_engine_4.data.leela import resolve_data_paths
    from chess_engine_4.training.inference_mismatch import (
        evaluate_native_checkpoint,
        sample_training_positions,
    )

    checkpoint_path = Path(payload["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    positions = sample_training_positions(
        resolve_data_paths(REMOTE_DATA_PATH),
        count=int(payload["samples"]),
        seed=int(payload["seed"]),
    )
    native_mxfp8 = evaluate_native_checkpoint(
        checkpoint_path,
        positions,
        precision_override="mxfp8",
    )
    native_bf16 = evaluate_native_checkpoint(
        checkpoint_path,
        positions,
        precision_override="bf16",
    )

    prefix = REMOTE_EVAL_PATH / str(payload["run_name"])
    positions_path = prefix.with_name(prefix.name + "-positions.json")
    native_path = prefix.with_name(prefix.name + "-native.npz")
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    positions_path.write_text(
        json.dumps(
            [
                {"initial_fen": position.initial_fen, "moves": position.moves}
                for position in positions
            ]
        )
    )
    np.savez(
        native_path,
        mxfp8_policies=np.stack(native_mxfp8.policies),
        mxfp8_q=native_mxfp8.q,
        mxfp8_d=native_mxfp8.d,
        bf16_policies=np.stack(native_bf16.policies),
        bf16_q=native_bf16.q,
        bf16_d=native_bf16.d,
    )
    artifact_volume.commit()
    return {"positions_path": str(positions_path), "native_path": str(native_path)}


def _run_export_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from chess_engine_4.training.inference_comparison import (
        NetworkOutputs,
        UciPosition,
        compare_outputs,
        compare_raw_outputs,
        evaluate_lc0,
    )

    weights_paths = [Path(path) for path in payload["weights"]]
    lc0_path = Path(payload["lc0_path"])
    positions_path = Path(payload["positions_path"])
    native_path = Path(payload["native_path"])
    for required_path in [lc0_path, positions_path, native_path, *weights_paths]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    positions = [
        UciPosition(initial_fen=item["initial_fen"], moves=tuple(item["moves"]))
        for item in json.loads(positions_path.read_text())
    ]
    with np.load(native_path) as data:
        native_mxfp8 = NetworkOutputs(
            policies=list(data["mxfp8_policies"]),
            q=data["mxfp8_q"],
            d=data["mxfp8_d"],
        )
        native_bf16 = NetworkOutputs(
            policies=list(data["bf16_policies"]),
            q=data["bf16_q"],
            d=data["bf16_d"],
        )

    comparisons = {}
    for weights_path in weights_paths:
        lc0 = evaluate_lc0(
            lc0_path=lc0_path,
            weights_path=weights_path,
            positions=positions,
            backend=str(payload["backend"]),
            env={**os.environ, "LD_LIBRARY_PATH": RUNTIME_LIBRARY_PATH},
        )
        comparisons[weights_path.name] = {
            "native_mxfp8_vs_native_bf16": compare_raw_outputs(
                native_mxfp8,
                native_bf16,
            ),
            "native_mxfp8_vs_lc0": compare_outputs(native_mxfp8, lc0),
            "native_bf16_vs_lc0": compare_outputs(native_bf16, lc0),
        }

    result = {
        "checkpoint": str(payload["checkpoint"]),
        "samples": len(positions),
        "seed": int(payload["seed"]),
        "backend": str(payload["backend"]),
        "exports": comparisons,
    }
    output_path = REMOTE_EVAL_PATH / f"{payload['run_name']}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    positions_path.unlink()
    native_path.unlink()
    artifact_volume.commit()
    result["result_path"] = str(output_path)
    return result


def _mounted_artifact_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(REMOTE_ARTIFACT_PATH) / path


@app.function(
    image=training_image,
    gpu="B200",
    cpu=8,
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=60 * 60,
)
def _evaluate_native(payload: dict[str, Any]) -> dict[str, str]:
    return _run_native_evaluation(payload)


@app.function(
    image=lc0_image,
    gpu="B200",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=3 * 60 * 60,
)
def _evaluate_exports(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_export_evaluation(payload)
