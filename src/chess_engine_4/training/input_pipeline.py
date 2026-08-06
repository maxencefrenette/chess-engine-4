"""Pinned host staging for native training batches."""

from __future__ import annotations

from dataclasses import dataclass
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

@dataclass(frozen=True, slots=True)
class StagedBatch:
    tensors: NativeBatch
    slot: int | None


class TrainingBatchPipeline:
    """Stage and transfer batches using the configured host-to-device path."""

    def __init__(self, *, kind: InputPipeline, device: torch.device) -> None:
        self._pin_batch = kind != "pageable"
        self._device = device
        match kind:
            case "pageable" | "pinned":
                self._stager = None
                self._copy_stream = None
            case "staging":
                self._stager = PinnedBatchStager()
                self._copy_stream = None
            case "overlap":
                self._stager = None
                self._copy_stream = torch.cuda.Stream(device=device)
            case _:
                assert_never(kind)

    def stage(self, batch: NativeBatch) -> StagedBatch:
        if self._stager is not None:
            return self._stager.stage(batch)
        tensors = pin_batch(batch) if self._pin_batch else batch
        return StagedBatch(tensors, None)

    def transfer(
        self,
        batch: StagedBatch,
        *,
        copy_start: torch.cuda.Event | None = None,
        copy_end: torch.cuda.Event | None = None,
    ) -> tuple[PackedPlaneInput, PolicyTarget, torch.Tensor]:
        result = _enqueue_batch_to_device(
            batch.tensors,
            device=self._device,
            copy_stream=self._copy_stream,
            copy_start=copy_start,
            copy_end=copy_end,
        )
        transfer_stream = (
            self._copy_stream
            if self._copy_stream is not None
            else torch.cuda.current_stream(self._device)
        )
        if batch.slot is not None and self._stager is not None:
            self._stager.record_h2d(batch.slot, transfer_stream)
        if self._copy_stream is not None:
            current_stream = torch.cuda.current_stream(self._device)
            current_stream.wait_stream(self._copy_stream)
            _record_batch_stream(*result, current_stream)
        return result


class PinnedBatchStager:
    """Copy batches into two reusable pinned slots without racing async H2D copies."""

    def __init__(self) -> None:
        self._slots: list[NativeBatch] = []
        self._copy_done: list[torch.cuda.Event | None] = [None, None]
        self._next_slot = 0

    def stage(self, batch: NativeBatch) -> StagedBatch:
        if not self._slots:
            self._slots = [_allocate_pinned_like(batch), _allocate_pinned_like(batch)]
        slot = self._next_slot
        self._next_slot = (slot + 1) % len(self._slots)
        copy_done = self._copy_done[slot]
        if copy_done is not None:
            copy_done.synchronize()
        destination = self._slots[slot]
        for target, source in zip(destination, batch, strict=True):
            target.copy_(source)
        return StagedBatch(destination, slot)

    def record_h2d(self, slot: int, stream: torch.cuda.Stream) -> None:
        event = self._copy_done[slot]
        if event is None:
            event = torch.cuda.Event()
            self._copy_done[slot] = event
        event.record(stream)


def _allocate_pinned_like(batch: NativeBatch) -> NativeBatch:
    return (
        torch.empty_like(batch[0], pin_memory=True),
        torch.empty_like(batch[1], pin_memory=True),
        torch.empty_like(batch[2], pin_memory=True),
        torch.empty_like(batch[3], pin_memory=True),
        torch.empty_like(batch[4], pin_memory=True),
    )


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
