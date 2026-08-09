import torch

from chess_engine_4.training import input_pipeline


def test_configured_transfer_paths(monkeypatch) -> None:
    copy_stream = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda *, device: copy_stream)
    device = torch.device("cpu")

    pageable = input_pipeline.TrainingBatchPipeline(kind="pageable", device=device)
    pinned = input_pipeline.TrainingBatchPipeline(kind="pinned", device=device)
    overlap = input_pipeline.TrainingBatchPipeline(kind="overlap", device=device)

    assert not pageable._pin_batch
    assert pageable._copy_stream is None
    assert pinned._pin_batch and pinned._copy_stream is None
    assert overlap._pin_batch
    assert overlap._copy_stream is copy_stream
