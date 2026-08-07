"""Export training checkpoints into the stable chess-engine-4 model format."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal
import torch
from safetensors.torch import save_file

from chess_engine_4.modal_train import (
    ARTIFACT_VOLUME_NAME,
    REMOTE_ARTIFACT_PATH,
    artifact_volume,
    image,
)

FORMAT_NAME = "chess-engine-4"
FORMAT_VERSION = "1"
REMOTE_MODEL_PATH = Path(REMOTE_ARTIFACT_PATH) / "models"

app = modal.App("chess-engine-4-export-model")


def export_model() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint as Safetensors.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--remote-only",
        action="store_true",
        help="Keep a remotely exported model in the Modal artifacts volume.",
    )
    args = parser.parse_args()

    output = args.output or Path("artifacts/models") / f"{args.checkpoint.stem}.safetensors"
    if args.checkpoint.exists():
        export_checkpoint(args.checkpoint, output)
    else:
        remote_checkpoint = _mounted_artifact_path(args.checkpoint)
        remote_output = REMOTE_MODEL_PATH / output.name
        with app.run():
            result = _export_checkpoint_remote.remote(
                str(remote_checkpoint),
                str(remote_output),
            )
        if args.remote_only:
            print(f"wrote {result}")
            return
        _download_artifact(Path(result), output)
    print(f"wrote {output}")


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tensors, metadata = exported_model(checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path, metadata=metadata)


def exported_model(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint does not contain a config.model mapping.")
    model = config["model"]
    kind = model.get("kind", "dense")
    if kind == "dense":
        return exported_dense_model(checkpoint)
    if kind == "moe64a2":
        return exported_moe_model(checkpoint)
    raise ValueError(f"Safetensors export does not support architecture {kind!r}.")


def exported_dense_model(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    model, state = _model_and_state(checkpoint, expected_kind="dense")
    if model.get("activation", "swiglu") != "swiglu":
        raise ValueError("Safetensors export currently supports only SwiGLU dense checkpoints.")
    depth = int(model["depth"])
    d_model = int(model["d_model"])
    policy_size = int(model.get("policy_size", 1858))
    tensors = {
        "input.weight": _bf16_tensor(state, "input.weight"),
        "input.bias": _bf16_tensor(state, "input.bias"),
        "final_norm.weight": _bf16_tensor(state, "norm.weight"),
        "policy.weight": _bf16_tensor(state, "policy_head.weight"),
        "policy.bias": _bf16_tensor(state, "policy_head.bias"),
        "wdl.weight": _bf16_tensor(state, "wdl_head.weight"),
        "wdl.bias": _bf16_tensor(state, "wdl_head.bias"),
        "moves_left.weight": _bf16_tensor(state, "moves_left_head.weight"),
        "moves_left.bias": _bf16_tensor(state, "moves_left_head.bias"),
    }
    for layer in range(depth):
        source = f"blocks.{layer}.layer"
        target = f"blocks.{layer}"
        tensors[f"{target}.norm.weight"] = _bf16_tensor(state, f"{source}.layer_norm_weight")
        tensors[f"{target}.gate_up.weight"] = _bf16_tensor(state, f"{source}.fc1_weight")
        tensors[f"{target}.down.weight"] = _bf16_tensor(state, f"{source}.fc2_weight")

    metadata = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "architecture": "dense",
        "d_model": str(d_model),
        "depth": str(depth),
        "expansion_ratio": str(float(model.get("expansion_ratio", 4.0))),
        "activation": "swiglu",
        "history_length": str(int(model.get("history_length", 8))),
        "input_planes": str(int(model.get("input_planes", 112))),
        "board_size": str(int(model.get("board_size", 8))),
        "policy_size": str(policy_size),
        "policy_storage_size": str(tensors["policy.weight"].shape[0]),
        "wdl_size": "3",
        "moves_left_size": "1",
        "rms_norm_eps": str(float(model.get("rms_norm_eps", 1e-6))),
        "input_format": "lc0-classical-112",
        "input_normalization": "history-select-rule50-div99-v1",
        "source_run": str(checkpoint.get("run_name", "")),
        "source_step": str(checkpoint.get("step", "")),
    }
    return tensors, metadata


def exported_moe_model(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    model, state = _model_and_state(checkpoint, expected_kind="moe64a2")
    if model.get("activation", "swiglu") != "swiglu":
        raise ValueError("Safetensors export currently supports only SwiGLU MoE checkpoints.")

    depth = int(model["depth"])
    d_model = int(model["d_model"])
    hidden_dim = int(d_model * float(model.get("expansion_ratio", 2.0)))
    policy_size = int(model.get("policy_size", 1858))
    tensors = {
        "input.weight": _bf16_tensor(state, "input.weight"),
        "input.bias": _bf16_tensor(state, "input.bias"),
        "final_norm.weight": _bf16_tensor(state, "norm.weight"),
        "policy.weight": _bf16_tensor(state, "policy_head.weight"),
        "policy.bias": _bf16_tensor(state, "policy_head.bias"),
        "wdl.weight": _bf16_tensor(state, "wdl_head.weight"),
        "wdl.bias": _bf16_tensor(state, "wdl_head.bias"),
        "moves_left.weight": _bf16_tensor(state, "moves_left_head.weight"),
        "moves_left.bias": _bf16_tensor(state, "moves_left_head.bias"),
    }
    for layer in range(depth):
        target = f"blocks.{layer}"
        if layer % 2:
            source = f"blocks.{layer}.layer"
            tensors[f"{target}.norm.weight"] = _bf16_tensor(
                state, f"{source}.layer_norm_weight"
            )
            tensors[f"{target}.gate_up.weight"] = _bf16_tensor(
                state, f"{source}.fc1_weight"
            )
            tensors[f"{target}.down.weight"] = _bf16_tensor(
                state, f"{source}.fc2_weight"
            )
            continue

        tensors[f"{target}.norm.weight"] = _bf16_tensor(
            state, f"blocks.{layer}.norm.weight"
        )
        tensors[f"{target}.router.weight"] = _bf16_tensor(
            state, f"blocks.{layer}.router.weight"
        )[:64]
        custom_gate_up = f"blocks.{layer}.experts.gate_up_weight"
        if custom_gate_up in state:
            tensors[f"{target}.experts.gate_up.weight"] = _bf16_tensor(
                state, custom_gate_up
            )
            tensors[f"{target}.experts.down.weight"] = _bf16_tensor(
                state, f"blocks.{layer}.experts.down_weight"
            )
        else:
            tensors[f"{target}.experts.gate_up.weight"] = torch.stack(
                [
                    _bf16_tensor(state, f"blocks.{layer}.experts.0.weight{expert}")
                    for expert in range(64)
                ]
            )
            tensors[f"{target}.experts.down.weight"] = torch.stack(
                [
                    _bf16_tensor(state, f"blocks.{layer}.experts.2.weight{expert}")
                    for expert in range(64)
                ]
            )

        if tensors[f"{target}.experts.gate_up.weight"].shape != (
            64,
            2 * hidden_dim,
            d_model,
        ):
            raise ValueError(f"MoE layer {layer} has an unexpected gate/up shape.")
        if tensors[f"{target}.experts.down.weight"].shape != (64, d_model, hidden_dim):
            raise ValueError(f"MoE layer {layer} has an unexpected down shape.")

    metadata = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "architecture": "moe64a2",
        "d_model": str(d_model),
        "depth": str(depth),
        "expansion_ratio": str(float(model.get("expansion_ratio", 2.0))),
        "num_experts": "64",
        "num_active_experts": "2",
        "layer_pattern": "alternating-moe-dense",
        "activation": "swiglu",
        "history_length": str(int(model.get("history_length", 8))),
        "input_planes": str(int(model.get("input_planes", 112))),
        "board_size": str(int(model.get("board_size", 8))),
        "policy_size": str(policy_size),
        "policy_storage_size": str(tensors["policy.weight"].shape[0]),
        "wdl_size": "3",
        "moves_left_size": "1",
        "rms_norm_eps": str(float(model.get("rms_norm_eps", 1e-6))),
        "input_format": "lc0-classical-112",
        "input_normalization": "history-select-rule50-div99-v1",
        "source_run": str(checkpoint.get("run_name", "")),
        "source_step": str(checkpoint.get("step", "")),
    }
    return tensors, metadata


def _model_and_state(
    checkpoint: dict[str, Any], *, expected_kind: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint does not contain a config.model mapping.")
    model = config["model"]
    if model.get("kind", "dense") != expected_kind:
        raise ValueError(f"Checkpoint is not a {expected_kind} model.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model_state_dict mapping.")
    return model, state


def _bf16_tensor(state: dict[str, Any], name: str) -> torch.Tensor:
    tensor = state.get(name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Checkpoint is missing tensor {name!r}.")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Tensor {name!r} must be BF16, got {tensor.dtype}.")
    return tensor.detach().cpu().contiguous()


def _mounted_artifact_path(path: Path) -> Path:
    if str(path).startswith(f"{REMOTE_ARTIFACT_PATH}/"):
        return path
    return Path(REMOTE_ARTIFACT_PATH) / str(path).lstrip("/")


def _volume_path(path: Path) -> str:
    return "/" + str(path).removeprefix(REMOTE_ARTIFACT_PATH).lstrip("/")


def _download_artifact(remote_path: Path, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            "--force",
            ARTIFACT_VOLUME_NAME,
            _volume_path(remote_path),
            str(local_path),
        ],
        check=True,
    )


@app.function(
    image=image,
    cpu=2,
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=60 * 60,
)
def _export_checkpoint_remote(checkpoint_path: str, output_path: str) -> str:
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    export_checkpoint(checkpoint, output)
    artifact_volume.commit()
    return str(output)


if __name__ == "__main__":
    export_model()
