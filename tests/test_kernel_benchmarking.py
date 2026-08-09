from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from chess_engine_4.kernels.benchmarking import (
    compare_gradients,
    named_tensor_metrics,
    tensor_metrics,
)


def test_output_metrics_reports_error_and_cosine_similarity() -> None:
    output = torch.tensor([1.0, 3.0])
    reference = torch.tensor([1.0, 2.0])

    metrics = tensor_metrics(output, reference, F)

    assert metrics["mean_absolute_error"] == 0.5
    assert metrics["max_absolute_error"] == 1.0
    assert metrics["cosine_similarity"] == pytest.approx(7 / (10**0.5 * 5**0.5))


def test_named_output_metrics_preserves_requested_names() -> None:
    custom = (torch.tensor([1.0]), torch.tensor([2.0]))
    reference = (torch.tensor([1.0]), torch.tensor([1.0]))

    metrics = named_tensor_metrics(("input", "weight"), custom, reference, F)

    assert list(metrics) == ["input", "weight"]
    assert metrics["input"]["mean_absolute_error"] == 0.0
    assert metrics["weight"]["mean_absolute_error"] == 1.0


def test_named_gradient_metrics_reports_finiteness_and_magnitudes() -> None:
    custom = (torch.tensor([1.0, 2.0]),)
    reference = (torch.tensor([1.0, 3.0]),)

    metrics = compare_gradients(("input",), custom, reference, F)["input"]

    assert metrics["custom_finite"] is True
    assert metrics["reference_finite"] is True
    assert metrics["custom_abs_max"] == 2.0
    assert metrics["reference_abs_max"] == 3.0
