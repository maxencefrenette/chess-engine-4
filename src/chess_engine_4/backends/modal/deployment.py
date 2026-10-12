"""Historical cloud placement defaults; no Modal SDK or cloud calls required."""

from dataclasses import replace

from chess_engine_4.hardware import TrainingGpu
from chess_engine_4.training.config import TrainingConfig

_GPU_BY_FAMILY: dict[str, dict[int, TrainingGpu]] = {
    "dense": {
        64: "RTX-PRO-6000", 128: "RTX-PRO-6000", 256: "RTX-PRO-6000",
        512: "B200", 768: "B200", 1024: "B200", 1280: "B200",
    },
    "moe64a2": {
        128: "RTX-PRO-6000", 256: "A100", 384: "B200", 512: "B200",
        640: "B200", 768: "B200", 1024: "B200",
    },
}


def apply_modal_defaults(config: TrainingConfig) -> TrainingConfig:
    """Select the old cloud placement only when no explicit GPU was supplied."""
    if config.infra.gpu is not None:
        return config
    try:
        gpu = _GPU_BY_FAMILY[config.model.kind][config.model.d_model]
    except KeyError as error:
        raise ValueError("No default Modal placement for this model; specify --gpu.") from error
    return replace(config, infra=replace(config.infra, gpu=gpu))
