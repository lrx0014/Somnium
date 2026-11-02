"""Pluggable inference and training backends."""

from __future__ import annotations

from somnium.backends.base import (
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    TrainingBackend,
    TrainingJobResult,
    TrainingJobSpec,
)

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "InferenceBackend",
    "TrainingBackend",
    "TrainingJobResult",
    "TrainingJobSpec",
]
