"""Apple Silicon MLX-LM inference and LoRA training backends (development profile).

Only modules inside this package may import `mlx` or `mlx_lm`.
"""

from __future__ import annotations

from somnium.backends.mlx.runner import (
    MLXCompatibilityReport,
    MLXLoRAAssociativeBackend,
    render_qwen3_chat_prompt,
    verify_qwen3_mlx_compatibility,
)

__all__ = [
    "MLXCompatibilityReport",
    "MLXLoRAAssociativeBackend",
    "render_qwen3_chat_prompt",
    "verify_qwen3_mlx_compatibility",
]
