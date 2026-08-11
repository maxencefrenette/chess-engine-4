from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from chess_engine_4.training.export_model import (
    export_checkpoint,
    exported_dense_model,
    exported_moe_model,
)


def _checkpoint(*, d_model: int = 64) -> dict:
    hidden_dim = 4 * d_model
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
    assert metadata["d_model"] == "64"
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
        assert handle.get_tensor("policy.weight").shape == (1888, 64)


def test_export_rejects_non_bf16_weights() -> None:
    checkpoint = _checkpoint()
    checkpoint["model_state_dict"]["input.weight"] = torch.zeros(64, 7168)

    with pytest.raises(ValueError, match="must be BF16"):
        exported_dense_model(checkpoint)


def test_export_rejects_legacy_non_lc0_geometry() -> None:
    checkpoint = _checkpoint()
    checkpoint["config"]["model"]["policy_size"] = 1860

    with pytest.raises(ValueError, match="policy_size must be 1858"):
        exported_dense_model(checkpoint)


def test_exported_moe_model_flattens_experts() -> None:
    d_model = 128
    hidden_dim = 256
    state = _checkpoint(d_model=d_model)["model_state_dict"]
    for name in list(state):
        if name.startswith("blocks."):
            del state[name]
    state.update(
        {
            "blocks.0.norm.weight": torch.ones(d_model, dtype=torch.bfloat16),
            "blocks.0.router.weight": torch.zeros(64, d_model, dtype=torch.bfloat16),
            "blocks.0.router_qb_beta": torch.linspace(-1, 1, 64),
            "blocks.1.layer.layer_norm_weight": torch.ones(
                d_model, dtype=torch.bfloat16
            ),
            "blocks.1.layer.fc1_weight": torch.zeros(
                8 * d_model, d_model, dtype=torch.bfloat16
            ),
            "blocks.1.layer.fc2_weight": torch.zeros(
                d_model, 4 * d_model, dtype=torch.bfloat16
            ),
        }
    )
    for expert in range(64):
        state[f"blocks.0.experts.0.weight{expert}"] = torch.zeros(
            2 * hidden_dim, d_model, dtype=torch.bfloat16
        )
        state[f"blocks.0.experts.2.weight{expert}"] = torch.zeros(
            d_model, hidden_dim, dtype=torch.bfloat16
        )
    checkpoint = {
        "run_name": "moe-test",
        "step": 7,
        "config": {
            "model": {
                "kind": "moe64a2",
                "d_model": d_model,
                "depth": 2,
                "expansion_ratio": 2.0,
                "activation": "swiglu",
                "history_length": 8,
            }
        },
        "model_state_dict": state,
    }

    tensors, metadata = exported_moe_model(checkpoint)

    assert metadata["architecture"] == "moe64a2"
    assert metadata["num_experts"] == "64"
    assert metadata["router_load_balancing"] == "quantile"
    torch.testing.assert_close(
        tensors["blocks.0.router_qb_beta"],
        torch.linspace(-1, 1, 64),
    )
    assert tensors["blocks.0.experts.gate_up.weight"].shape == (64, 512, 128)
    assert tensors["blocks.0.experts.down.weight"].shape == (64, 128, 256)
    assert tensors["blocks.1.gate_up.weight"].shape == (1024, 128)

    del state["blocks.0.router_qb_beta"]
    legacy_tensors, _ = exported_moe_model(checkpoint)
    torch.testing.assert_close(
        legacy_tensors["blocks.0.router_qb_beta"],
        torch.zeros(64),
    )


def test_export_rejects_quantile_moe_without_router_bias() -> None:
    checkpoint = {
        "config": {
            "model": {
                "kind": "moe64a2",
                "d_model": 64,
                "depth": 2,
                "expansion_ratio": 2.0,
                "activation": "swiglu",
                "history_length": 8,
                "router_load_balancing": "quantile",
            }
        },
        "model_state_dict": {},
    }

    with pytest.raises(ValueError, match="router_qb_beta"):
        exported_moe_model(checkpoint)
