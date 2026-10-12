"""Import boundaries for local commands, even when Modal is unavailable."""

import subprocess
import sys
import tomllib
from pathlib import Path


def test_modal_is_an_optional_dependency() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert not any(dep.startswith("modal") for dep in project["dependencies"])
    assert project["optional-dependencies"]["modal"] == ["modal>=1.4.2"]


def test_local_imports_and_help_without_modal() -> None:
    # A fresh interpreter prevents previously imported cloud modules from masking
    # accidental dependencies. No GPU initialization or cloud calls are allowed.
    code = '''
import importlib
import importlib.abc
import sys
import tomllib
from pathlib import Path

class NoModal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "modal" or fullname.startswith("modal."):
            raise ModuleNotFoundError("Modal intentionally unavailable", name="modal")

sys.meta_path.insert(0, NoModal())
for module in (
    "chess_engine_4.analysis.throughput_sweep",
    "chess_engine_4.evaluation.tournament",
    "chess_engine_4.evaluation.backendbench",
    "chess_engine_4.model.reference",
):
    importlib.import_module(module)

scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]
for command in (
    "train-local", "export-model", "inspect-data", "build-lc0", "build-kernels",
    "plan-budget", "plan-ladder", "throughput-sweep", "generate-website-data",
    "compare-run", "wandb-metrics", "ingest-iceberg", "lc0-to-parquet",
):
    module, function = scripts[command].split(":")
    sys.argv = [command, "--help"]
    try:
        getattr(importlib.import_module(module), function)()
    except SystemExit as error:
        assert error.code == 0, (command, error.code)
    else:
        raise AssertionError(f"{command} did not handle --help")

sys.argv = ["export-model", "missing.pt", "--remote"]
from chess_engine_4.release.export import export_model
try:
    export_model()
except SystemExit as error:
    assert error.code == 2
else:
    raise AssertionError("remote export should require the optional SDK")
assert "modal" not in sys.modules
assert not any(name.startswith("chess_engine_4.backends.modal.") for name in sys.modules)
'''
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Remote export requires Modal" in result.stderr
