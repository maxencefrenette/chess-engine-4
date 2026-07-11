from __future__ import annotations

import numpy as np

from chess_engine_4.training.inference_comparison import (
    NetworkOutputs,
    _parse_lc0_evaluation,
    compare_outputs,
)


def test_parse_lc0_evaluation_reads_policy_indices_and_root_value() -> None:
    policy, q, d = _parse_lc0_evaluation(
        [
            "info string e2e4  (322) N: 0 (+0) (P: 55.25%) (WL: -.-----)",
            "info string d2d4  (304) N: 0 (+0) (P: 44.75%) (WL: -.-----)",
            "info string node  (2) N: 1 (+0) (P: 100.0%) (WL: 0.12500) "
            "(D: 0.250) (M: 1.0) (Q: 0.12500)",
            "bestmove e2e4",
        ]
    )

    assert policy == {322: 0.5525, 304: 0.4475}
    assert q == 0.125
    assert d == 0.25


def test_compare_outputs_reports_policy_and_value_drift() -> None:
    native = NetworkOutputs(
        policies=[np.asarray([2.0, 0.0], dtype=np.float32)],
        q=np.asarray([0.2], dtype=np.float32),
        d=np.asarray([0.3], dtype=np.float32),
    )
    exported = NetworkOutputs(
        policies=[{0: 0.8, 1: 0.2}],
        q=np.asarray([0.1], dtype=np.float32),
        d=np.asarray([0.4], dtype=np.float32),
    )

    result = compare_outputs(native, exported)

    assert result["positions"] == 1
    assert result["policy_top1_agreement"] == 1.0
    assert np.isclose(result["q_mae"], 0.1)
    assert np.isclose(result["draw_mae"], 0.1)
