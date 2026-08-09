"""Shared model configuration types."""

from typing import Literal

type InputPipeline = Literal["overlap", "pageable", "pinned"]
type KernelBackend = Literal["custom", "te"]
type Precision = Literal["bf16", "mxfp8", "nvfp4"]
