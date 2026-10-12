"""Parse lc0 backendbench results independently of execution backend."""

from __future__ import annotations

import re
from typing import Any

_RESULT_ROW = re.compile(
    r"^\s*(?P<batch_size>\d+),\s*"
    r"(?P<nodes_per_sec>[\d.]+),\s*"
    r"(?P<mean_ms>[\d.]+),",
    re.MULTILINE,
)


def parse_backendbench_result(result: dict[str, Any]) -> dict[str, Any]:
    match = _RESULT_ROW.search(str(result["output"]))
    if match is None:
        raise ValueError(f"Could not parse backendbench output:\n{result['output']}")
    return {
        "name": str(result["name"]),
        "weights": str(result["weights"]),
        "backend": str(result["backend"]),
        "nodes_per_sec": float(match.group("nodes_per_sec")),
        "mean_ms": float(match.group("mean_ms")),
    }
