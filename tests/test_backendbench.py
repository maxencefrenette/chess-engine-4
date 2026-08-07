from chess_engine_4.evaluation.backendbench import parse_backendbench_result


def test_parse_backendbench_result() -> None:
    result = parse_backendbench_result(
        {
            "name": "dense-d128",
            "weights": "/models/dense-d128.safetensors",
            "backend": "ce4",
            "output": """size, mean nps, mean ms, sdev\n  32,    68869,  0.4646, 0.0030\n""",
        }
    )

    assert result == {
        "name": "dense-d128",
        "weights": "/models/dense-d128.safetensors",
        "backend": "ce4",
        "nodes_per_sec": 68869.0,
        "mean_ms": 0.4646,
    }
