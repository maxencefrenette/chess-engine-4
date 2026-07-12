from __future__ import annotations

import json
from pathlib import Path

from chess_engine_4.training.export_scaling_data import write_scaling_data


def test_export_scaling_data_includes_dense_relative_targets(tmp_path: Path) -> None:
    output = tmp_path / "scaling-laws.json"
    write_scaling_data(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    dense = payload["families"]["dense"]
    assert set(payload["families"]) == {"dense"}
    assert [point["budget"] for point in dense["extrapolated"]] == ["1e24", "1e25"]

    assert len(dense["curves"]["samplesPerParam"]) == 61
    assert len(dense["curves"]["lr"]) == 61
    assert len(dense["curves"]["steps"]) == 61
    assert len(dense["curves"]["batchSize"]) == 61
    assert "lossUpper1sd" not in dense["observed"][0]
    assert all(point["physicalFlops"] > 0 for point in dense["observed"])
    assert all(point["samplesPerParam"] > 0 for point in dense["extrapolated"])
    assert all(point["steps"] > 0 for point in dense["observed"])
    assert all(point["batchSize"] > 0 for point in dense["extrapolated"])
