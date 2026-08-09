"""Single source of truth for training-kernel capability and dispatch."""

from __future__ import annotations

from dataclasses import dataclass

from chess_engine_4.model.config import KernelBackend, Precision

type ComputeCapability = tuple[int, int]

SUPPORTED_DENSE_WIDTHS = frozenset({32, 64, 128, 256, 512, 1024, 2048})
SUPPORTED_MOE_WIDTHS = frozenset({128, 256, 512})

_DENSE_OP_PREFIX_BY_CAPABILITY: dict[ComputeCapability, str] = {
    (8, 0): "sm80_",
    (10, 0): "",
    (12, 0): "sm120_",
}
_MOE_OP_PREFIX_BY_CAPABILITY: dict[ComputeCapability, str] = {
    (8, 0): "sm80_",
    (10, 0): "sm100_",
    (12, 0): "",
}


@dataclass(frozen=True, slots=True)
class KernelSelection:
    backend: KernelBackend
    variant: str


def resolve_kernel_backend(
    *,
    backend: KernelBackend,
    kind: str,
    capability: ComputeCapability,
    precision: Precision,
    d_model: int,
    hidden_dim: int,
    activation: str,
    batch_size: int,
    num_experts: int | None = None,
    num_active_experts: int | None = None,
) -> KernelSelection:
    """Resolve an explicitly selected backend or reject it before launch."""

    if backend == "te":
        _require_te_precision(capability, precision)
        return KernelSelection(backend="te", variant=f"te-{precision}")
    if backend != "custom":
        raise ValueError(f"unknown kernel backend: {backend!r}")
    if kind == "dense":
        variant = require_dense_kernel(
            capability=capability,
            precision=precision,
            d_model=d_model,
            hidden_dim=hidden_dim,
            rows=batch_size,
            activation=activation,
        )
    elif kind == "moe64a2":
        variant = require_moe_kernel(
            capability=capability,
            precision=precision,
            d_model=d_model,
            hidden_dim=hidden_dim,
            activation=activation,
            num_experts=num_experts,
            num_active_experts=num_active_experts,
        )
    else:
        raise ValueError(f"custom kernels are not implemented for model kind {kind!r}")
    return KernelSelection(backend="custom", variant=variant)


def require_dense_kernel(
    *,
    capability: ComputeCapability,
    precision: Precision,
    d_model: int,
    hidden_dim: int,
    rows: int,
    activation: str = "swiglu",
) -> str:
    require_dense_precision(capability, precision)
    require_dense_model_shape(
        d_model=d_model,
        hidden_dim=hidden_dim,
        activation=activation,
    )

    if capability == (8, 0):
        row_alignment = 16
    else:
        if precision == "mxfp8" and d_model % 256:
            raise ValueError("custom MXFP8 dense kernels require d_model divisible by 256")
        row_alignment = 128
    if rows % row_alignment:
        raise ValueError(
            f"custom dense kernels on {_format_sm(capability)} require rows divisible by "
            f"{row_alignment}, got {rows}"
        )
    return f"dense-{_format_sm(capability).lower()}-{precision}"


def require_dense_model_shape(*, d_model: int, hidden_dim: int, activation: str) -> None:
    if d_model not in SUPPORTED_DENSE_WIDTHS:
        raise ValueError(
            f"custom dense kernels require d_model in {sorted(SUPPORTED_DENSE_WIDTHS)}, "
            f"got {d_model}"
        )
    if hidden_dim != 4 * d_model:
        raise ValueError("custom dense kernels require expansion_ratio=4")
    if activation != "swiglu":
        raise ValueError("custom dense kernels require activation='swiglu'")


def require_moe_kernel(
    *,
    capability: ComputeCapability,
    precision: Precision,
    d_model: int,
    hidden_dim: int,
    activation: str = "swiglu",
    num_experts: int | None = 64,
    num_active_experts: int | None = 2,
    rows: int | None = None,
) -> str:
    if capability not in _MOE_OP_PREFIX_BY_CAPABILITY:
        raise ValueError(
            f"custom MoE kernels support SM80, SM100, and SM120, got "
            f"{_format_sm(capability)}"
        )
    if precision != "bf16":
        raise ValueError("custom MoE kernels require precision='bf16'")
    require_moe_model_shape(
        d_model=d_model,
        hidden_dim=hidden_dim,
        activation=activation,
        num_experts=num_experts,
        num_active_experts=num_active_experts,
    )
    if rows is not None and rows % 16:
        raise ValueError(f"custom MoE kernels require rows divisible by 16, got {rows}")
    return f"moe-{_format_sm(capability).lower()}-bf16"


def require_moe_model_shape(
    *,
    d_model: int,
    hidden_dim: int,
    activation: str,
    num_experts: int | None = 64,
    num_active_experts: int | None = 2,
) -> None:
    if d_model not in SUPPORTED_MOE_WIDTHS:
        raise ValueError(
            f"custom MoE kernels require d_model in {sorted(SUPPORTED_MOE_WIDTHS)}, "
            f"got {d_model}"
        )
    if hidden_dim != 2 * d_model:
        raise ValueError("custom MoE kernels require expansion_ratio=2")
    if activation != "swiglu":
        raise ValueError("custom MoE kernels require activation='swiglu'")
    if num_experts != 64 or num_active_experts != 2:
        raise ValueError("custom MoE kernels require 64 experts with 2 active experts")


def dense_op_prefix(capability: ComputeCapability) -> str:
    try:
        return _DENSE_OP_PREFIX_BY_CAPABILITY[capability]
    except KeyError as error:
        raise ValueError(
            f"custom dense kernels support SM80, SM100, and SM120, got "
            f"{_format_sm(capability)}"
        ) from error


def require_dense_precision(capability: ComputeCapability, precision: Precision) -> str:
    prefix = dense_op_prefix(capability)
    if capability == (8, 0) and precision != "bf16":
        raise ValueError("custom dense kernels on SM80 require precision='bf16'")
    if capability == (10, 0) and precision not in {"bf16", "mxfp8"}:
        raise ValueError(
            "custom dense kernels on SM100 require precision='bf16' or 'mxfp8'"
        )
    if capability == (12, 0) and precision != "bf16":
        raise ValueError("custom dense kernels on SM120 require precision='bf16'")
    return prefix


def moe_op_prefix(capability: ComputeCapability) -> str:
    try:
        return _MOE_OP_PREFIX_BY_CAPABILITY[capability]
    except KeyError as error:
        raise ValueError(
            f"custom MoE kernels support SM80, SM100, and SM120, got "
            f"{_format_sm(capability)}"
        ) from error


def _require_te_precision(capability: ComputeCapability, precision: Precision) -> None:
    if precision == "bf16" and capability in {(8, 0), (10, 0), (12, 0)}:
        return
    if precision == "mxfp8" and capability == (10, 0):
        return
    if precision == "nvfp4" and capability in {(10, 0), (12, 0)}:
        return
    raise ValueError(
        f"Transformer Engine precision={precision!r} is not supported on "
        f"{_format_sm(capability)}"
    )


def _format_sm(capability: ComputeCapability) -> str:
    return f"SM{capability[0]}{capability[1]}"
