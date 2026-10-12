"""Cloud command wiring without submitting jobs or building remote images."""

import importlib
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("modal", reason="Install the modal extra to test cloud adapters")


def test_cloud_entrypoints_resolve_after_package_move() -> None:
    scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]
    for target in scripts.values():
        module, function = target.split(":")
        if module.startswith("chess_engine_4.backends.modal."):
            assert callable(getattr(importlib.import_module(module), function)), target
    adapter = importlib.import_module("chess_engine_4.backends.modal.export")
    assert callable(adapter.export_remote_checkpoint)
