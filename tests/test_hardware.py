from __future__ import annotations

import pytest

from chess_engine_4.hardware import gpu_spec, identify_training_gpu, modal_gpu_identifier


@pytest.mark.parametrize("gpu", ["H100", "H200"])
def test_hopper_catalog_identity_and_capability(gpu: str) -> None:
    spec = gpu_spec(gpu)

    assert spec.capability == (9, 0)
    assert spec.device_name == gpu
    assert spec.theoretical_tflops == {"bf16": 989.0}


def test_h100_modal_request_disables_automatic_h200_upgrade() -> None:
    assert modal_gpu_identifier("H100") == "H100!"
    assert modal_gpu_identifier("H200") == "H200"


def test_local_rtx_5070_identity() -> None:
    assert (
        identify_training_gpu(
            device_name="NVIDIA GeForce RTX 5070",
            capability=(12, 0),
        )
        == "RTX-5070"
    )
    with pytest.raises(ValueError, match="not available on Modal"):
        modal_gpu_identifier("RTX-5070")
