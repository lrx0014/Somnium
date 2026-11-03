"""Apple Silicon MLX-LM inference and LoRA training backends (development profile).

Only modules inside this package may import `mlx` or `mlx_lm`.
"""

from __future__ import annotations

from somnium.backends.mlx.inference import AdapterSwapHook, MLXInferenceBackend
from somnium.backends.mlx.runner import (
    MLXCompatibilityReport,
    MLXLoRAAssociativeBackend,
    render_qwen3_chat_prompt,
    verify_qwen3_mlx_compatibility,
)
from somnium.backends.mlx.trainer import MLXTrainingBackend, load_adapter_checkpoint

__all__ = [
    "AdapterSwapHook",
    "MLXCompatibilityReport",
    "MLXInferenceBackend",
    "MLXLoRAAssociativeBackend",
    "MLXTrainingBackend",
    "load_adapter_checkpoint",
    "render_qwen3_chat_prompt",
    "verify_qwen3_mlx_compatibility",
]
