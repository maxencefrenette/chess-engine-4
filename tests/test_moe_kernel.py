from __future__ import annotations

import pytest
import torch

from chess_engine_4.kernels.capabilities import moe_op_prefix, require_moe_kernel
from chess_engine_4.kernels.moe import _moe_op


@pytest.mark.parametrize(
    ("capability", "prefix", "variant"),
    [
        ((8, 0), "sm80_", "moe-sm80-bf16"),
        ((9, 0), "sm90_", "moe-sm90-bf16"),
        ((10, 0), "sm100_", "moe-sm100-bf16"),
        ((12, 0), "", "moe-sm120-bf16"),
    ],
)
def test_moe_kernel_dispatches_supported_architectures(
    capability: tuple[int, int],
    prefix: str,
    variant: str,
) -> None:
    assert moe_op_prefix(capability) == prefix
    assert (
        require_moe_kernel(
            capability=capability,
            precision="bf16",
            d_model=256,
            hidden_dim=512,
            rows=128,
        )
        == variant
    )


def test_moe_kernel_rejects_unsupported_gpu_architecture(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (11, 0))

    with pytest.raises(
        ValueError,
        match="support SM80, SM90, SM100, and SM120, got SM110",
    ):
        _moe_op(torch.empty(0), 128, "forward")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"precision": "mxfp8"}, "precision='bf16'"),
        ({"d_model": 1024, "hidden_dim": 2048}, "d_model in"),
        ({"hidden_dim": 1024}, "expansion_ratio=2"),
        ({"activation": "gelu"}, "activation='swiglu'"),
        ({"num_experts": 32}, "64 experts with 2 active experts"),
        ({"num_active_experts": 4}, "64 experts with 2 active experts"),
        ({"rows": 127}, "rows divisible by 16"),
    ],
)
def test_sm100_moe_kernel_rejects_unsupported_configuration(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "capability": (10, 0),
        "precision": "bf16",
        "d_model": 256,
        "hidden_dim": 512,
        "activation": "swiglu",
        "num_experts": 64,
        "num_active_experts": 2,
        "rows": 128,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        require_moe_kernel(**arguments)  # type: ignore[arg-type]


def test_moe_kernel_dispatches_sm90_prefix(monkeypatch) -> None:
    sentinel = object()
    module = type("Extension", (), {"sm90_moe_d128_forward": sentinel})()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr("chess_engine_4.kernels.moe.extension", lambda: module)

    assert _moe_op(torch.empty(0), 128, "forward") is sentinel
