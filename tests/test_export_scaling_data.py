from __future__ import annotations

import json
from pathlib import Path

from chess_engine_4.training.export_scaling_data import write_scaling_data


def test_export_scaling_data_includes_two_family_relative_targets(tmp_path: Path) -> None:
    output = tmp_path / "scaling-laws.json"
    write_scaling_data(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    dense = payload["families"]["mlp"]
    moe = payload["families"]["mlp_moe16a2"]
    assert [point["budget"] for point in dense["extrapolated"]] == ["1e23", "1e24"]
    assert [point["budget"] for point in moe["extrapolated"]] == ["1e22", "1e23"]

    for family in (dense, moe):
        assert len(family["curves"]["samplesPerParam"]) == 61
        assert len(family["curves"]["lr"]) == 61
        assert "lossUpper1sd" not in family["observed"][0]
        assert all(point["physicalFlops"] > 0 for point in family["observed"])
        assert all(point["samplesPerParam"] > 0 for point in family["extrapolated"])
