"""Supported training hardware and dated Modal cost metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type TrainingGpu = Literal["A100", "B200", "RTX-PRO-6000"]


@dataclass(frozen=True, slots=True)
class GpuSpec:
    capability: tuple[int, int]
    device_name: str
    theoretical_tflops: dict[str, float]
    dollars_per_second: float


GPU_SPECS: dict[TrainingGpu, GpuSpec] = {
    "A100": GpuSpec((8, 0), "A100", {"bf16": 312.0}, 0.000583),
    "B200": GpuSpec(
        (10, 0),
        "B200",
        {"bf16": 2250.0, "mxfp8": 4500.0, "nvfp4": 9000.0},
        0.001736,
    ),
    "RTX-PRO-6000": GpuSpec((12, 0), "RTX PRO 6000", {"bf16": 503.8}, 0.000842),
}

TRAINING_GPUS: tuple[TrainingGpu, ...] = tuple(GPU_SPECS)
CPU_DOLLARS_PER_CORE_SECOND = 0.0000131


def gpu_spec(gpu: str) -> GpuSpec:
    try:
        return GPU_SPECS[gpu]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(f"Unsupported training GPU {gpu!r}.") from error


def hardware_dollars_per_second(gpu: str, cpu_cores: int) -> float:
    if cpu_cores <= 0:
        raise ValueError("cpu_cores must be positive.")
    return gpu_spec(gpu).dollars_per_second + cpu_cores * CPU_DOLLARS_PER_CORE_SECOND
