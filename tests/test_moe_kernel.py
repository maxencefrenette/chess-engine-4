from __future__ import annotations

import pytest
import torch

from chess_engine_4.kernels.moe import _moe_op


def test_moe_kernel_rejects_unsupported_gpu_architecture(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))

    with pytest.raises(ValueError, match="support SM80 and SM120, got SM90"):
        _moe_op(torch.empty(0), 128, "forward")
