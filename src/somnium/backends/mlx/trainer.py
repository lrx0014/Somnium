"""MLX TrainingBackend for LoRA adapter fine-tuning (M5.3).

Per `design.md` §8.2 and §8.3:
- Trains candidate LoRA adapters from dataset manifests using MLX AdamW on Metal GPU.
- Supports incremental adaptation from a parent checkpoint or full retrain from clean base.
- Packages self-contained, content-addressed checkpoints (weights + adapter configuration)
  into the `ArtifactStore`.
- Conforms to the `TrainingBackend` Protocol (`somnium.backends.base`).
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlx.core as mx

from somnium.backends.base import TrainingJobResult, TrainingJobSpec
from somnium.backends.mlx.runner import MLXLoRAAssociativeBackend
from somnium.eval.questions import TrainingExample
from somnium.interfaces import ArtifactStore


class MLXTrainingBackend:
    """MLX-based LoRA training backend on Apple Silicon (M5.3)."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        associative_backend: MLXLoRAAssociativeBackend | None = None,
    ) -> None:
        self._artifact_store = artifact_store
        self._associative_backend = (
            associative_backend if associative_backend is not None else MLXLoRAAssociativeBackend()
        )
        self.trained_jobs: list[TrainingJobSpec] = []

    @property
    def backend_name(self) -> str:
        """Identifier of the MLX training backend."""
        return "mlx"

    @property
    def associative_backend(self) -> MLXLoRAAssociativeBackend:
        """Underlying neural associative memory backend."""
        return self._associative_backend

    def _load_examples(self, spec: TrainingJobSpec) -> tuple[TrainingExample, ...]:
        """Read and parse training examples from dataset artifact or URI."""
        raw_bytes: bytes | None = None

        if self._artifact_store.exists("datasets", spec.dataset_manifest_hash):
            raw_bytes = self._artifact_store.get_bytes("datasets", spec.dataset_manifest_hash)
        elif spec.dataset_uri:
            parsed = urlparse(spec.dataset_uri)
            if parsed.scheme in ("file", ""):
                local_path = Path(parsed.path if parsed.path else spec.dataset_uri)
                if local_path.is_file():
                    raw_bytes = local_path.read_bytes()

        if raw_bytes is None:
            raise KeyError(
                f"Dataset not found for manifest hash '{spec.dataset_manifest_hash}' "
                f"at URI '{spec.dataset_uri}'."
            )

        lines = [line.strip() for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
        examples: list[TrainingExample] = []
        for line in lines:
            data = json.loads(line)
            examples.append(
                TrainingExample(
                    example_id=str(data["example_id"]),
                    category=str(data["category"]),
                    fact_group_id=str(data["fact_group_id"]),
                    tenant_id=str(data["tenant_id"]),
                    scope=str(data["scope"]),
                    temporal_slice=int(data.get("temporal_slice", 1)),
                    source_record_ids=tuple(data.get("source_record_ids", ())),
                    system_prompt=str(data["system_prompt"]),
                    user_prompt=str(data["user_prompt"]),
                    assistant_response=str(data["assistant_response"]),
                    split=str(data.get("split", "train")),
                    created_at=data.get("created_at"),
                )
            )

        return tuple(examples)

    def train_adapter(self, spec: TrainingJobSpec) -> TrainingJobResult:
        """Train a candidate LoRA adapter and package checkpoint into ArtifactStore."""
        examples = self._load_examples(spec)
        self.trained_jobs.append(spec)
        candidate_ver = spec.candidate_version or f"engram-{spec.tenant_id}-{spec.job_id}"

        # 1. Execute LoRA optimization via MLXLoRAAssociativeBackend
        metrics = self._associative_backend.train_on_examples(
            examples,
            version_name=candidate_ver,
            parent_version=spec.parent_adapter_version,
            epochs=spec.epochs,
            learning_rate=spec.learning_rate,
        )
        metrics.setdefault("train_loss", metrics.get("final_loss", 0.0))

        # 2. Extract trained weights and response catalog
        weights_dict, response_catalog = self._associative_backend._checkpoints[candidate_ver]

        # 3. Serialize adapter into a self-contained ZIP archive
        adapter_config = {
            "job_id": spec.job_id,
            "tenant_id": spec.tenant_id,
            "candidate_version": candidate_ver,
            "parent_adapter_version": spec.parent_adapter_version,
            "base_model_id": spec.base_model_id,
            "base_model_revision": spec.base_model_revision,
            "dataset_manifest_hash": spec.dataset_manifest_hash,
            "training_mode": spec.training_mode,
            "lora_rank": spec.lora_rank,
            "epochs": spec.epochs,
            "learning_rate": spec.learning_rate,
            "metrics": metrics,
            "response_catalog": response_catalog,
        }

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "adapter_config.json",
                json.dumps(adapter_config, indent=2, sort_keys=True),
            )
            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_sf:
                tmp_path = tmp_sf.name
            try:
                mx.save_safetensors(tmp_path, weights_dict)
                zf.write(tmp_path, arcname="adapter.safetensors")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        zip_bytes = zip_buf.getvalue()

        # 4. Atomically persist adapter package into ArtifactStore under 'adapters'
        checkpoint_hash = self._artifact_store.put_bytes("adapters", zip_bytes)
        checkpoint_uri = self._artifact_store.uri_for("adapters", checkpoint_hash)

        return TrainingJobResult(
            job_id=spec.job_id,
            tenant_id=spec.tenant_id,
            candidate_version=candidate_ver,
            checkpoint_uri=checkpoint_uri,
            dataset_manifest_hash=spec.dataset_manifest_hash,
            metrics=metrics,
        )


def load_adapter_checkpoint(
    artifact_store: ArtifactStore,
    checkpoint_hash: str,
) -> tuple[dict[str, Any], dict[str, mx.array]]:
    """Unpack adapter configuration and MLX weights from an ArtifactStore checkpoint."""
    raw_bytes = artifact_store.get_bytes("adapters", checkpoint_hash)
    zip_buf = io.BytesIO(raw_bytes)
    with zipfile.ZipFile(zip_buf, "r") as zf:
        cfg_bytes = zf.read("adapter_config.json")
        cfg = json.loads(cfg_bytes.decode("utf-8"))
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_sf:
            tmp_sf.write(zf.read("adapter.safetensors"))
            tmp_path = tmp_sf.name

    try:
        loaded = mx.load(tmp_path)
        if isinstance(loaded, dict):
            loaded_weights: dict[str, mx.array] = {
                str(k): v for k, v in loaded.items() if isinstance(v, mx.array)
            }
        elif isinstance(loaded, tuple) and len(loaded) > 0 and isinstance(loaded[0], dict):
            loaded_weights = {str(k): v for k, v in loaded[0].items() if isinstance(v, mx.array)}
        else:
            loaded_weights = {}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return cfg, loaded_weights


__all__ = ["MLXTrainingBackend", "load_adapter_checkpoint"]
