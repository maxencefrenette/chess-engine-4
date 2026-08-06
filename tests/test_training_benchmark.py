from chess_engine_4.modal_training_benchmark import _summarize


def test_summarize_reports_distribution() -> None:
    summary = _summarize([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary == {
        "mean": 3.0,
        "median": 3.0,
        "p10": 1.4,
        "p90": 4.6,
        "stddev": 2.0**0.5,
    }
