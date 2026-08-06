"""Shared model configuration types."""

from typing import Literal

type InputPipeline = Literal["overlap", "pageable", "pinned", "staging"]
