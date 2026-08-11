from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from chess_engine_4.training.generate_website_data import (
    FAMILIES,
    build_family_payload,
    write_website_data,
)


def test_generate_website_data_includes_family_relative_targets(tmp_path: Path) -> None:
    output = tmp_path / "scaling-laws.json"
    write_website_data(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert [family["id"] for family in payload["families"]] == ["dense", "moe64a2"]
    dense = payload["families"][0]
    assert [point["name"] for point in dense["runs"]] == [
        "d32",
        "d64",
        "d128",
        "d256",
        "d512",
        "d768",
        "d1024",
        "d1280",
    ]
    assert dense["trainingRatio"] == 0.2
    assert all(point["status"] == "current" for point in dense["runs"])
    assert dense["extrapolated"] == []

    assert len(dense["curves"]["samplesPerParam"]) == 61
    assert len(dense["curves"]["lr"]) == 61
    assert len(dense["curves"]["steps"]) == 61
    assert len(dense["curves"]["batchSize"]) == 61
    assert all(point["physicalFlops"] > 0 for point in dense["runs"])
    assert all("compute" not in point for point in dense["runs"])
    assert all("compute" not in point for point in dense["curves"]["loss"])
    assert all(point["batchSize"] > 0 for point in dense["extrapolated"])
    assert [point["gpu"] for point in dense["runs"]] == [
        "RTX-PRO-6000",
        "RTX-PRO-6000",
        "RTX-PRO-6000",
        "RTX-PRO-6000",
        "B200",
        "B200",
        "B200",
        "B200",
    ]
    assert dense["runs"][0]["runtimeSec"] > 0
    assert dense["runs"][-1]["runtimeSec"] > 0
    assert all(
        removed not in point
        for point in dense["runs"]
        for removed in (
            "sourceExperiment",
            "modelKind",
            "dModel",
            "trainingRatio",
            "steps",
            "samplesPerParam",
            "stale",
        )
    )

    moe = payload["families"][1]
    assert [point["name"] for point in moe["runs"] if point["status"] == "current"] == [
        "d256",
        "d512",
    ]
    assert moe["trainingRatio"] == 0.05
    assert moe["extrapolated"] == []
    assert len(moe["curves"]["loss"]) == 61
    assert moe["curves"]["policyTop1"] == []
    assert all(
        len(moe["curves"][name]) == 61
        for name in ("params", "samples", "samplesPerParam", "lr", "steps", "batchSize")
    )


def test_generate_website_data_uses_one_stale_status(tmp_path: Path) -> None:
    source = Path("experiments/best-runs-dense.toml").read_text(encoding="utf-8")
    path = tmp_path / "best-runs-dense.toml"
    path.write_text(source.replace("[runs.d32]", "[runs.d32]\nstale = true", 1))

    family = build_family_payload("dense", replace(FAMILIES["dense"], best_runs=path))
    stale_run = next(run for run in family["runs"] if run["name"] == "d32")

    assert stale_run["status"] == "stale"
    assert "stale" not in stale_run
