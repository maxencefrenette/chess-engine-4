"""Modal validation command for comparing native precision recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import modal

from chess_engine_4.modal_train import (
    REMOTE_ARTIFACT_PATH,
    REMOTE_DATA_PATH,
    REMOTE_PARQUET_DATA_PATH,
    artifact_volume,
    data_volume,
    image,
)

app = modal.App("chess-engine-4-precision-eval")


def eval_precision_modal() -> None:
    parser = argparse.ArgumentParser(description="Compare checkpoint validation by precision.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--samples", type=int, default=131_072)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        parser.error("--samples and --batch-size must be positive")
    if args.batch_size % 32:
        parser.error("--batch-size must be divisible by 32 for MXFP8")

    payload = {
        "checkpoint": str(_mounted_artifact_path(args.checkpoint)),
        "data_glob": str(Path(REMOTE_PARQUET_DATA_PATH) / args.data_glob),
        "samples": args.samples,
        "batch_size": args.batch_size,
    }
    with modal.enable_output(), app.run():
        result = _evaluate_precision.remote(payload)
    print(json.dumps(result, indent=2, sort_keys=True))


def _run_precision_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    import math

    import numpy as np
    import torch

    from chess_engine_4.data.leela import LeelaParquetDataset, resolve_data_paths
    from chess_engine_4.model import build_model, model_config_from_dict
    from chess_engine_4.model.transformer_engine import autocast_context
    from chess_engine_4.training.losses import LossWeights, lczero_loss
    from chess_engine_4.training.packed_input import PlaneInputExpander

    checkpoint_path = Path(payload["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    paths = resolve_data_paths(payload["data_glob"])
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    raw_config = checkpoint["config"]
    model_config = model_config_from_dict(raw_config["model"])
    loss_weights = LossWeights(**raw_config.get("loss", {}))
    models = {}
    for precision in ("mxfp8", "bf16"):
        model = build_model(model_config).cuda().eval()
        state = dict(checkpoint["model_state_dict"])
        for name, tensor in model.state_dict().items():
            if name.endswith(".router_qb_beta") and name not in state:
                state[name] = torch.zeros_like(tensor)
        model.load_state_dict(state)
        models[precision] = model
    expander = PlaneInputExpander().cuda().eval()

    metric_batches: dict[str, dict[str, list[float]]] = {
        precision: {
            "loss": [],
            "loss_policy": [],
            "loss_value": [],
            "loss_moves_left": [],
            "policy_top1": [],
        }
        for precision in models
    }
    samples_seen = 0
    dataset = LeelaParquetDataset(
        paths,
        batch_size=int(payload["batch_size"]),
        threads=min(8, len(paths)),
        prefetch_per_thread=2,
    )
    with torch.no_grad():
        for packed, scalars, policy_indices, policy_probs, values in dataset:
            remaining = int(payload["samples"]) - samples_seen
            if remaining <= 0:
                break
            if remaining < len(packed):
                packed = packed[:remaining]
                scalars = scalars[:remaining]
                policy_indices = policy_indices[:remaining]
                policy_probs = policy_probs[:remaining]
                values = values[:remaining]
            packed = packed.pin_memory().cuda(non_blocking=True)
            scalars = scalars.pin_memory().to(
                device="cuda",
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            policy_indices = policy_indices.pin_memory().cuda(non_blocking=True)
            policy_probs = policy_probs.pin_memory().cuda(non_blocking=True)
            values = values.pin_memory().cuda(non_blocking=True)
            planes = expander(packed, scalars)

            for precision, model in models.items():
                with autocast_context(precision):
                    output = model(planes.clone())
                    loss = lczero_loss(
                        output,
                        (policy_indices, policy_probs),
                        values,
                        weights=loss_weights,
                    )
                valid = policy_indices >= 0
                gathered = output.policy_logits.gather(
                    dim=-1,
                    index=policy_indices.clamp_min(0).long(),
                ).masked_fill(~valid, torch.finfo(output.policy_logits.dtype).min)
                top1 = (gathered.argmax(dim=-1) == policy_probs.argmax(dim=-1)).float().mean()
                batch_metrics = metric_batches[precision]
                batch_metrics["loss"].append(loss.task.item())
                batch_metrics["loss_policy"].append(loss.policy.item())
                batch_metrics["loss_value"].append(loss.value.item())
                batch_metrics["loss_moves_left"].append(loss.moves_left.item())
                batch_metrics["policy_top1"].append(top1.item())
            samples_seen += len(packed)
            if samples_seen >= int(payload["samples"]):
                break

    results = {}
    for precision, metrics in metric_batches.items():
        results[precision] = {}
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            results[precision][metric] = float(array.mean())
            results[precision][f"{metric}_se"] = float(
                array.std(ddof=1) / math.sqrt(len(array)) if len(array) > 1 else 0.0
            )
    paired_differences = {}
    for metric in metric_batches["bf16"]:
        difference = np.asarray(metric_batches["bf16"][metric]) - np.asarray(
            metric_batches["mxfp8"][metric]
        )
        paired_differences[metric] = float(difference.mean())
        paired_differences[f"{metric}_se"] = float(
            difference.std(ddof=1) / math.sqrt(len(difference)) if len(difference) > 1 else 0.0
        )
    results["bf16_minus_mxfp8"] = paired_differences
    return {
        "checkpoint": str(checkpoint_path),
        "data_glob": str(payload["data_glob"]),
        "data_files": len(paths),
        "samples": samples_seen,
        "batch_size": int(payload["batch_size"]),
        "results": results,
    }


def _mounted_artifact_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(REMOTE_ARTIFACT_PATH) / path


@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    volumes={REMOTE_DATA_PATH: data_volume, REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=60 * 60,
)
def _evaluate_precision(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_precision_evaluation(payload)
