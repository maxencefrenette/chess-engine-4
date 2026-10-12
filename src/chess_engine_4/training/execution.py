"""Execution settings, separate from model and optimization recipes."""

from dataclasses import dataclass

from chess_engine_4.hardware import TrainingGpu


@dataclass(frozen=True, slots=True)
class InfraConfig:
    # Keep the persisted `infra` field names so existing checkpoints still load.
    # Recipes leave the GPU unresolved; a local or cloud launcher selects it.
    gpu: TrainingGpu | None = None
    cpu_cores: int = 8  # Modal allocation; does not reserve local CPU cores.
    dataloader_threads: int = 8
    dataloader_prefetch_per_thread: int = 2

    def __post_init__(self) -> None:
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive.")
        if self.dataloader_threads <= 0 or self.dataloader_prefetch_per_thread <= 0:
            raise ValueError("Data loader threads and prefetch must be positive.")

    def require_gpu(self) -> TrainingGpu:
        if self.gpu is None:
            raise ValueError("GPU is unresolved; select local or Modal deployment before launch.")
        return self.gpu
