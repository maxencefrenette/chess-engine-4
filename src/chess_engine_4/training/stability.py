"""Training stability metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar


@dataclass(slots=True)
class LossSpikeDetector:
    """Count unusually large one-sided loss excursions."""

    decay: ClassVar[float] = 0.99
    warmup_steps: ClassVar[int] = 200
    std_threshold: ClassVar[float] = 6.0
    minimum_excess: ClassVar[float] = 0.10

    steps: int = 0
    count: int = 0
    mean: float | None = None
    second_moment: float | None = None

    def update_many(self, losses: list[float]) -> None:
        for loss in losses:
            self.update(loss)

    def update(self, loss: float) -> None:
        self.steps += 1
        if not math.isfinite(loss):
            self.count += 1
            return

        if self.mean is None or self.second_moment is None:
            self.mean = loss
            self.second_moment = loss * loss
            return

        variance = max(self.second_moment - self.mean * self.mean, 0.0)
        threshold = self.mean + max(
            self.std_threshold * math.sqrt(variance),
            self.minimum_excess,
        )
        if self.steps > self.warmup_steps and loss > threshold:
            self.count += 1

        self.mean = self.decay * self.mean + (1.0 - self.decay) * loss
        self.second_moment = (
            self.decay * self.second_moment + (1.0 - self.decay) * loss * loss
        )
