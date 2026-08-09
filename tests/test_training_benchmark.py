from chess_engine_4.modal_train import _explicit_gpu_default_backend
from chess_engine_4.modal_training_benchmark import (
    _next_loader_batch,
    _summarize,
    _with_realized_cost,
)


def test_summarize_reports_distribution() -> None:
    summary = _summarize([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary == {
        "mean": 3.0,
        "median": 3.0,
        "p10": 1.4,
        "p90": 4.6,
        "stddev": 2.0**0.5,
    }


def test_realized_cost_includes_gpu_and_reserved_cpu() -> None:
    measurement = {
        "te": {"wall_ms": {"median": 10.0}},
        "custom": {"wall_ms": {"median": 8.0}},
    }

    result = _with_realized_cost(measurement, gpu="H100", cpu_cores=8)

    assert result["dollars_per_second"] == 0.001097 + 8 * 0.0000131
    assert result["te"]["dollars_per_step"] == result["dollars_per_second"] * 0.010
    assert result["custom"]["dollars_per_step"] == result["dollars_per_second"] * 0.008
    assert result["cost_efficiency_vs_te"] == 1.25


def test_loader_retry_only_handles_prefetch_timeout() -> None:
    iterator = iter((ValueError("timed out after 5s waiting"), "batch"))

    class FlakyIterator:
        def __next__(self):
            value = next(iterator)
            if isinstance(value, Exception):
                raise value
            return value

    assert _next_loader_batch(FlakyIterator()) == "batch"


def test_explicit_hopper_gpu_uses_only_measured_custom_default() -> None:
    assert _explicit_gpu_default_backend(gpu="H100", model_kind="moe64a2") == "custom"
    assert _explicit_gpu_default_backend(gpu="H100", model_kind="dense") == "te"
    assert _explicit_gpu_default_backend(gpu="H200", model_kind="moe64a2") == "te"
