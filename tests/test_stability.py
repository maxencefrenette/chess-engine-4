from __future__ import annotations

import math

from chess_engine_4.training.stability import LossSpikeDetector


def test_loss_spike_detector_ignores_warmup_and_counts_late_spikes() -> None:
    warmup_detector = LossSpikeDetector()

    warmup_detector.update_many([3.0] * 199 + [4.0])
    assert warmup_detector.count == 0

    detector = LossSpikeDetector()

    detector.update_many([3.0] * 400 + [3.2])
    assert detector.count == 1


def test_loss_spike_detector_counts_non_finite_losses() -> None:
    detector = LossSpikeDetector()

    detector.update(math.nan)
    detector.update(math.inf)

    assert detector.count == 2
