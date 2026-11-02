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
from somnium.backends.fake import FakeInferenceBackend
from somnium.config import InferenceConfig


def create_inference_backend(config: InferenceConfig) -> InferenceBackend:
    """Instantiate the configured `InferenceBackend` (`mlx` or `fake`)."""
    backend_kind = config.backend.strip().lower()
    if backend_kind == "fake":
        return FakeInferenceBackend(
            default_model_version=f"{config.base_model_id}@{config.base_model_revision}"
        )
    if backend_kind == "mlx":
        from somnium.backends.mlx.inference import MLXInferenceBackend

        return MLXInferenceBackend.from_config(config)
    raise ValueError(f"Unsupported inference backend '{config.backend}'.")


__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "InferenceBackend",
    "TrainingBackend",
    "TrainingJobResult",
    "TrainingJobSpec",
    "create_inference_backend",
]
