"""Host-to-device transfer for native training batches."""

from __future__ import annotations

from typing import assert_never

import torch

from chess_engine_4.model.config import InputPipeline
from chess_engine_4.training.losses import PolicyTarget
from chess_engine_4.training.packed_input import PackedPlaneInput

type NativeBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class TrainingBatchPipeline:
    """Stage and transfer batches using the configured host-to-device path."""

    def __init__(self, *, kind: InputPipeline, device: torch.device) -> None:
        self._pin_batch = kind != "pageable"
        self._device = device
        match kind:
            case "pageable" | "pinned":
                self._copy_stream = None
            case "overlap":
                self._copy_stream = torch.cuda.Stream(device=device)
            case _:
                assert_never(kind)

    def stage(self, batch: NativeBatch) -> NativeBatch:
        return pin_batch(batch) if self._pin_batch else batch

    def transfer(
        self,
        batch: NativeBatch,
        *,
        copy_start: torch.cuda.Event | None = None,
        copy_end: torch.cuda.Event | None = None,
    ) -> tuple[PackedPlaneInput, PolicyTarget, torch.Tensor]:
        result = _enqueue_batch_to_device(
            batch,
            device=self._device,
            copy_stream=self._copy_stream,
            copy_start=copy_start,
            copy_end=copy_end,
        )
        if self._copy_stream is not None:
            current_stream = torch.cuda.current_stream(self._device)
            current_stream.wait_stream(self._copy_stream)
            _record_batch_stream(*result, current_stream)
        return result


def pin_batch(batch: NativeBatch) -> NativeBatch:
    return (
        batch[0].pin_memory(),
        batch[1].pin_memory(),
        batch[2].pin_memory(),
        batch[3].pin_memory(),
        batch[4].pin_memory(),
    )


def _copy_batch_to_device(
    batch: NativeBatch,
    *,
    device: torch.device,
) -> tuple[PackedPlaneInput, PolicyTarget, torch.Tensor]:
    packed_planes, plane_scalars, policy_indices, policy_probs, value = batch
    planes = (
        packed_planes.to(device=device, non_blocking=True),
        plane_scalars.to(device=device, dtype=torch.bfloat16, non_blocking=True),
    )
    return (
        planes,
        (
            policy_indices.to(device=device, non_blocking=True),
            policy_probs.to(device=device, non_blocking=True),
        ),
        value.to(device, non_blocking=True),
    )


def _enqueue_batch_to_device(
    batch: NativeBatch,
    *,
    device: torch.device,
    copy_stream: torch.cuda.Stream | None,
    copy_start: torch.cuda.Event | None,
    copy_end: torch.cuda.Event | None,
) -> tuple[PackedPlaneInput, PolicyTarget, torch.Tensor]:
    if copy_stream is None:
        if copy_start is not None:
            copy_start.record()
        result = _copy_batch_to_device(batch, device=device)
        if copy_end is not None:
            copy_end.record()
        return result

    with torch.cuda.stream(copy_stream):
        if copy_start is not None:
            copy_start.record()
        result = _copy_batch_to_device(batch, device=device)
        if copy_end is not None:
            copy_end.record()
    return result


def _record_batch_stream(
    planes: PackedPlaneInput,
    policy: PolicyTarget,
    value: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    for tensor in (*planes, *policy, value):
        tensor.record_stream(stream)
