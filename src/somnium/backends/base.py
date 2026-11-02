"""Transport-agnostic backend Protocols for inference and training.

Application modules (Freud, Hippocore, Engram, Dream Engine) depend exclusively
on these narrow Protocol interfaces. Concrete ML libraries (`mlx`, `torch`,
`peft`, `vllm`, `transformers`) may only be imported inside their dedicated
subpackages (`somnium.backends.mlx`, `somnium.backends.cuda`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from somnium.domain import TenantId


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Backend-neutral text generation request."""

    tenant_id: TenantId
    prompt: str
    system_prompt: str | None = None
    max_tokens: int = 512
    temperature: float = 0.1
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Backend-neutral text generation response."""

    text: str
    model_version: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True, slots=True)
class TrainingJobSpec:
    """Backend-neutral specification for a LoRA training job."""

    job_id: str
    tenant_id: TenantId
    dataset_manifest_hash: str
    dataset_uri: str
    base_model_id: str
    base_model_revision: str
    parent_adapter_version: str | None = None
    training_mode: str = "incremental"
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 16


@dataclass(frozen=True, slots=True)
class TrainingJobResult:
    """Result emitted by a TrainingBackend upon completing a candidate adapter."""

    job_id: str
    tenant_id: TenantId
    candidate_version: str
    checkpoint_uri: str
    dataset_manifest_hash: str
    metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class InferenceBackend(Protocol):
    """Pluggable inference backend for Engram associative recall."""

    @property
    def backend_name(self) -> str:
        """Identifier of the backend implementation (e.g., 'fake', 'mlx', 'cuda')."""
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a completion using the tenant's active model/adapter."""
        ...

    def swap_adapter(
        self,
        tenant_id: TenantId,
        new_model_version: str,
        adapter_uri: str | None,
    ) -> str:
        """Atomically switch the active adapter for a tenant and return the previous version."""
        ...

    def active_version(self, tenant_id: TenantId) -> str:
        """Return the currently active model/adapter version identifier for a tenant."""
        ...


@runtime_checkable
class TrainingBackend(Protocol):
    """Pluggable training backend for Dream Engine LoRA fine-tuning."""

    @property
    def backend_name(self) -> str:
        """Identifier of the training backend (e.g., 'fake', 'mlx', 'cuda')."""
        ...

    def train_adapter(self, spec: TrainingJobSpec) -> TrainingJobResult:
        """Train a candidate LoRA adapter from a dataset manifest and emit checkpoint metadata."""
        ...
