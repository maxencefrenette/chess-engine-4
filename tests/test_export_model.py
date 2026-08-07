from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from chess_engine_4.training.export_model import export_checkpoint, exported_dense_model


def _checkpoint() -> dict:
    d_model = 32
    hidden_dim = 128
    policy_storage_size = 1888
    state = {
        "input.weight": torch.zeros(d_model, 7168, dtype=torch.bfloat16),
        "input.bias": torch.zeros(d_model, dtype=torch.bfloat16),
        "norm.weight": torch.ones(d_model, dtype=torch.bfloat16),
        "policy_head.weight": torch.zeros(policy_storage_size, d_model, dtype=torch.bfloat16),
        "policy_head.bias": torch.zeros(policy_storage_size, dtype=torch.bfloat16),
        "wdl_head.weight": torch.zeros(32, d_model, dtype=torch.bfloat16),
        "wdl_head.bias": torch.zeros(32, dtype=torch.bfloat16),
        "moves_left_head.weight": torch.zeros(32, d_model, dtype=torch.bfloat16),
        "moves_left_head.bias": torch.zeros(32, dtype=torch.bfloat16),
    }
    for layer in range(2):
        prefix = f"blocks.{layer}.layer"
        state[f"{prefix}.layer_norm_weight"] = torch.ones(d_model, dtype=torch.bfloat16)
        state[f"{prefix}.fc1_weight"] = torch.zeros(2 * hidden_dim, d_model, dtype=torch.bfloat16)
        state[f"{prefix}.fc2_weight"] = torch.zeros(d_model, hidden_dim, dtype=torch.bfloat16)
    return {
        "run_name": "test-run",
        "step": 42,
        "config": {
            "model": {
                "kind": "dense",
                "d_model": d_model,
                "depth": 2,
                "expansion_ratio": 4.0,
                "activation": "swiglu",
                "history_length": 8,
            }
        },
        "model_state_dict": state,
    }


def test_exported_dense_model_uses_stable_names() -> None:
    tensors, metadata = exported_dense_model(_checkpoint())

    assert metadata["format_version"] == "1"
    assert metadata["architecture"] == "dense"
    assert metadata["d_model"] == "32"
    assert metadata["source_run"] == "test-run"
    assert "blocks.1.gate_up.weight" in tensors
    assert not any("_extra_state" in name for name in tensors)
    assert all(tensor.dtype == torch.bfloat16 for tensor in tensors.values())


def test_export_checkpoint_writes_safetensors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "model.safetensors"
    torch.save(_checkpoint(), checkpoint)

    export_checkpoint(checkpoint, output)

    with safe_open(output, framework="pt") as handle:
        assert handle.metadata()["input_normalization"] == "history-select-rule50-div99-v1"
        assert handle.get_tensor("policy.weight").shape == (1888, 32)


def test_export_rejects_non_bf16_weights() -> None:
    checkpoint = _checkpoint()
    checkpoint["model_state_dict"]["input.weight"] = torch.zeros(32, 7168)

    with pytest.raises(ValueError, match="must be BF16"):
        exported_dense_model(checkpoint)
