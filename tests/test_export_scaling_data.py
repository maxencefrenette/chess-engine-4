from __future__ import annotations

import json
from pathlib import Path

from chess_engine_4.training.export_scaling_data import write_scaling_data


def test_export_scaling_data_includes_family_relative_targets(tmp_path: Path) -> None:
    output = tmp_path / "scaling-laws.json"
    write_scaling_data(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    dense = payload["families"]["dense"]
    assert set(payload["families"]) == {"dense", "moe64a2"}
    assert [point["name"] for point in dense["observed"]] == [
        "d32",
        "d64",
        "d128",
        "d256",
        "d512",
        "d1024",
    ]
    assert dense["trainingRatio"] == 0.2
    assert dense["staleObserved"] == []
    assert [point["name"] for point in dense["extrapolated"]] == ["d2048"]

    assert len(dense["curves"]["samplesPerParam"]) == 61
    assert len(dense["curves"]["lr"]) == 61
    assert len(dense["curves"]["steps"]) == 61
    assert len(dense["curves"]["batchSize"]) == 61
    assert all(point["physicalFlops"] > 0 for point in dense["observed"])
    assert all("compute" not in point for point in dense["observed"])
    assert all("compute" not in point for point in dense["curves"]["loss"])
    assert all(point["samplesPerParam"] > 0 for point in dense["extrapolated"])
    assert all(point["steps"] > 0 for point in dense["observed"])
    assert all(point["batchSize"] > 0 for point in dense["extrapolated"])
    assert [point["gpu"] for point in dense["observed"]] == [
        "RTX-PRO-6000",
        "RTX-PRO-6000",
        "RTX-PRO-6000",
        "RTX-PRO-6000",
        "B200",
        "B200",
    ]
    assert dense["observed"][0]["runtimeSec"] > 0
    assert dense["observed"][-1]["runtimeSec"] > 0

    moe = payload["families"]["moe64a2"]
    assert [point["name"] for point in moe["observed"]] == [
        "d128",
        "d256",
        "d512",
        "d1024",
    ]
    assert moe["trainingRatio"] == 0.05
    assert moe["extrapolated"] == []
