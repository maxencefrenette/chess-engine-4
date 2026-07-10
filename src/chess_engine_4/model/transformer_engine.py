"""Shared Transformer Engine construction helpers."""

from __future__ import annotations

from functools import cache
from typing import Any


@cache
def te() -> Any:
    """Import TE lazily so CPU-only tooling can still read configs and export checkpoints."""

    try:
        import transformer_engine.pytorch as transformer_engine
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Transformer Engine models require a CUDA runtime with transformer-engine installed."
        ) from exc
    return transformer_engine
