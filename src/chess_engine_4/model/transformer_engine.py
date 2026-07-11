"""Shared Transformer Engine construction helpers."""

from __future__ import annotations

from functools import cache
from typing import Any

import torch


@cache
def te() -> Any:
    """Import Transformer Engine lazily for config and export-only local tooling."""

    try:
        import transformer_engine.pytorch as transformer_engine
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Transformer Engine models require a CUDA runtime with transformer-engine installed."
        ) from exc
    return transformer_engine


@cache
def mxfp8_recipe() -> Any:
    """Return the single Blackwell MXFP8 recipe used by every training model."""

    from transformer_engine.common.recipe import MXFP8BlockScaling

    return MXFP8BlockScaling()


@cache
def nvfp4_recipe() -> Any:
    """Return the Blackwell NVFP4 training recipe."""

    from transformer_engine.common.recipe import NVFP4BlockScaling

    return NVFP4BlockScaling()


def quantization_recipe(name: str) -> Any | None:
    if name == "bf16":
        return None
    if name == "mxfp8":
        return mxfp8_recipe()
    if name == "nvfp4":
        return nvfp4_recipe()
    raise ValueError(f"unknown quantization recipe: {name}")


def autocast_context(name: str) -> Any:
    recipe = quantization_recipe(name)
    if recipe is None:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False)
    return te().autocast(enabled=True, recipe=recipe)
