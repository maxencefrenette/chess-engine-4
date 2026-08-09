from __future__ import annotations

import pytest

from chess_engine_4.hardware import gpu_spec, modal_gpu_identifier


@pytest.mark.parametrize("gpu", ["H100", "H200"])
def test_hopper_catalog_identity_and_capability(gpu: str) -> None:
    spec = gpu_spec(gpu)

    assert spec.capability == (9, 0)
    assert spec.device_name == gpu
    assert spec.theoretical_tflops == {"bf16": 989.0}


def test_h100_modal_request_disables_automatic_h200_upgrade() -> None:
    assert modal_gpu_identifier("H100") == "H100!"
    assert modal_gpu_identifier("H200") == "H200"
