"""Shared model configuration types."""

from typing import Literal

type InputPipeline = Literal["overlap", "pageable", "pinned", "staging"]
type Precision = Literal["bf16", "mxfp8", "nvfp4"]
