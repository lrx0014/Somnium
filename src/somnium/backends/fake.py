"""Deterministic test doubles for backends, clocks, auth, and artifacts (M0.7).

These implementations allow Freud, Hippocore, Engram pipelines, and the Dream
Engine to be tested end-to-end without loading any ML models or external
dependencies.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from somnium.backends.base import (
    GenerationRequest,
    GenerationResult,
    TrainingJobResult,
    TrainingJobSpec,
)
from somnium.domain import AuthContext, Permission, Scope, ScopeKind, TenantId

_DEFAULT_FAKE_RESPONSE = '{"status": "unknown", "claims": []}'
_FAKE_UNTRUSTED_RE = re.compile(
    r'<untrusted_memory\s+id="([^"]+)"\s+scope="([^"]+)">\s*(.*?)\s*</untrusted_memory>',
    re.DOTALL,
)


class SystemClock:
    """Production wall-clock returning timezone-aware UTC timestamps."""

    def now(self) -> datetime:
        """Return the current UTC datetime."""
        return datetime.now(tz=UTC)


class FrozenClock:
    """Deterministic controllable clock for unit and integration tests."""

    def __init__(self, initial: datetime | None = None) -> None:
        if initial is None:
            initial = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        if initial.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware datetime.")
        self._current = initial

    def now(self) -> datetime:
        """Return the current frozen timestamp."""
        return self._current

    def set(self, new_time: datetime) -> None:
        """Set the clock to a specific timezone-aware datetime."""
        if new_time.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware datetime.")
        self._current = new_time

    def advance(
        self,
        *,
        seconds: float = 0.0,
        minutes: float = 0.0,
        hours: float = 0.0,
    ) -> datetime:
        """Advance the frozen clock by the specified duration and return the new time."""
        self._current = self._current + timedelta(
            seconds=seconds,
            minutes=minutes,
            hours=hours,
        )
        return self._current


class StaticAuthProvider:
    """Static local authentication provider for single-user dev and tests."""

    def __init__(
        self,
        tenant_id: TenantId = "local-user",
        principal_id: str = "local-principal",
        authorized_scopes: Sequence[Scope] | None = None,
        permissions: frozenset[Permission] | None = None,
        allow_all_scopes: bool = False,
        expected_token: str | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._principal_id = principal_id
        self._authorized_scopes = tuple(
            authorized_scopes
            if authorized_scopes is not None
            else (Scope(kind=ScopeKind.PROJECT, id="default"),)
        )
        self._permissions = (
            permissions
            if permissions is not None
            else frozenset(
                {
                    Permission.READ,
                    Permission.WRITE,
                    Permission.CORRECT,
                    Permission.FORGET,
                    Permission.ASSOCIATIVE_RECALL,
                }
            )
        )
        self._allow_all_scopes = allow_all_scopes
        self._expected_token = expected_token

    def authenticate(
        self,
        token: str | None = None,
        requested_scopes: Sequence[Scope] | None = None,
    ) -> AuthContext:
        """Validate optional token and requested scopes against configured grants."""
        if self._expected_token is not None and token != self._expected_token:
            raise PermissionError("Invalid or missing authentication token.")

        if requested_scopes is not None:
            if not self._allow_all_scopes:
                unauthorized = [
                    scope for scope in requested_scopes if scope not in self._authorized_scopes
                ]
                if unauthorized:
                    joined = ", ".join(str(s) for s in unauthorized)
                    raise PermissionError(f"Unauthorized scope(s) requested: {joined}")
            effective_scopes = tuple(requested_scopes)
        else:
            effective_scopes = self._authorized_scopes

        return AuthContext(
            tenant_id=self._tenant_id,
            principal_id=self._principal_id,
            authorized_scopes=effective_scopes,
            permissions=self._permissions,
            allow_all_scopes=self._allow_all_scopes,
        )


class FakeInferenceBackend:
    """Deterministic, scriptable inference backend for testing without ML dependencies."""

    def __init__(
        self,
        default_model_version: str = "engram-base-v0",
        default_response: str = '{"status": "unknown", "claims": []}',
    ) -> None:
        self._default_version = default_model_version
        self._default_response = default_response
        self._active_versions: dict[TenantId, str] = {}
        self._active_adapter_uris: dict[TenantId, str | None] = {}
        self._scripted_substrings: list[tuple[str, str]] = []
        self._responder: Callable[[GenerationRequest, str], str] | None = None
        self.call_history: list[GenerationRequest] = []

    @property
    def backend_name(self) -> str:
        """Identifier of the fake inference backend."""
        return "fake"

    def script_response(self, prompt_substring: str, response_text: str) -> None:
        """Register a deterministic response when `prompt_substring` appears in the prompt."""
        self._scripted_substrings.append((prompt_substring, response_text))

    def set_responder(self, responder: Callable[[GenerationRequest, str], str] | None) -> None:
        """Set a custom callable `(request, active_version) -> response_text`."""
        self._responder = responder

    def active_version(self, tenant_id: TenantId) -> str:
        """Return the active model version for `tenant_id`."""
        return self._active_versions.get(tenant_id, self._default_version)

    def active_adapter_uri(self, tenant_id: TenantId) -> str | None:
        """Return the active adapter URI for `tenant_id`, if any."""
        return self._active_adapter_uris.get(tenant_id)

    def swap_adapter(
        self,
        tenant_id: TenantId,
        new_model_version: str,
        adapter_uri: str | None,
    ) -> str:
        """Atomically swap the active adapter for `tenant_id` and return the previous version."""
        previous = self.active_version(tenant_id)
        self._active_versions[tenant_id] = new_model_version
        self._active_adapter_uris[tenant_id] = adapter_uri
        return previous

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a deterministic completion based on scripted rules or default response."""
        self.call_history.append(request)
        version = self.active_version(request.tenant_id)

        if self._responder is not None:
            text = self._responder(request, version)
        else:
            text = None
            for substring, scripted in self._scripted_substrings:
                if substring in request.prompt:
                    text = scripted
                    break
            if text is None:
                if self._default_response != _DEFAULT_FAKE_RESPONSE:
                    text = self._default_response
                else:
                    text = self._default_engram_json(request.prompt)

        return GenerationResult(
            text=text,
            model_version=version,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
        )

    def _default_engram_json(self, prompt: str) -> str:
        """Synthesize deterministic Engram JSON from `<untrusted_memory>` blocks when unscripted."""
        blocks = [
            {
                "id": m.group(1).strip(),
                "scope": m.group(2).strip(),
                "content": m.group(3).strip(),
            }
            for m in _FAKE_UNTRUSTED_RE.finditer(prompt)
        ]
        if "[SOMNIUM_ENGRAM_PROMPT: associate_v1]" in prompt:
            return json.dumps(
                {
                    "template_version": "v1",
                    "cue_terms": [],
                    "candidate_claims": [
                        {
                            "text": b["content"],
                            "cue_terms": [],
                            "cited_ids": [b["id"]],
                            "scope": b["scope"],
                        }
                        for b in blocks
                    ],
                }
            )
        if "[SOMNIUM_ENGRAM_PROMPT: answer_v1]" in prompt:
            if not blocks:
                return json.dumps(
                    {
                        "status": "unknown",
                        "answer": None,
                        "claims": [],
                        "uncertainty": "high",
                    }
                )
            claims: list[dict[str, Any]] = []
            parts: list[str] = []
            for b in blocks:
                scope_id = b["scope"].split(":", 1)[-1]
                claim_text = f"[{scope_id}] {b['content']}"
                claims.append(
                    {
                        "text": claim_text,
                        "support": "supported",
                        "evidence": [b["id"]],
                        "scope": b["scope"],
                    }
                )
                parts.append(claim_text)
            return json.dumps(
                {
                    "status": "ok",
                    "answer": " ".join(parts),
                    "claims": claims,
                    "uncertainty": "low",
                }
            )
        return self._default_response


class FakeTrainingBackend:
    """Deterministic training backend that emits predictable checkpoints and metrics."""

    def __init__(self, default_metrics: dict[str, float] | None = None) -> None:
        self._default_metrics = default_metrics or {
            "train_loss": 0.42,
            "val_loss": 0.48,
        }
        self._should_fail: Exception | None = None
        self.trained_jobs: list[TrainingJobSpec] = []

    @property
    def backend_name(self) -> str:
        """Identifier of the fake training backend."""
        return "fake"

    def set_failure(self, exc: Exception | None) -> None:
        """Configure the next `train_adapter` call to raise `exc`."""
        self._should_fail = exc

    def train_adapter(self, spec: TrainingJobSpec) -> TrainingJobResult:
        """Simulate training an adapter and return a deterministic checkpoint URI."""
        if self._should_fail is not None:
            raise self._should_fail

        self.trained_jobs.append(spec)
        digest = hashlib.sha256(
            f"{spec.tenant_id}:{spec.job_id}:{spec.dataset_manifest_hash}:{spec.parent_adapter_version}".encode()
        ).hexdigest()[:12]
        candidate_version = f"engram-{spec.tenant_id}-{digest}"
        checkpoint_uri = f"memory://adapters/{spec.tenant_id}/{candidate_version}"

        return TrainingJobResult(
            job_id=spec.job_id,
            tenant_id=spec.tenant_id,
            candidate_version=candidate_version,
            checkpoint_uri=checkpoint_uri,
            dataset_manifest_hash=spec.dataset_manifest_hash,
            metrics=dict(self._default_metrics),
        )


class InMemoryArtifactStore:
    """Content-addressed in-memory artifact store for deterministic tests."""

    def __init__(self) -> None:
        self._blobs: dict[tuple[str, str], bytes] = {}

    def put_bytes(self, namespace: str, content: bytes) -> str:
        """Store `content` under `namespace` and return its SHA-256 hex digest."""
        content_hash = hashlib.sha256(content).hexdigest()
        self._blobs[(namespace, content_hash)] = bytes(content)
        return content_hash

    def get_bytes(self, namespace: str, content_hash: str) -> bytes:
        """Retrieve stored bytes or raise KeyError if missing."""
        key = (namespace, content_hash)
        if key not in self._blobs:
            raise KeyError(f"Artifact not found: {namespace}/{content_hash}")
        return self._blobs[key]

    def exists(self, namespace: str, content_hash: str) -> bool:
        """Return True if `(namespace, content_hash)` is present."""
        return (namespace, content_hash) in self._blobs

    def uri_for(self, namespace: str, content_hash: str) -> str:
        """Return an in-memory URI for the artifact."""
        return f"memory://artifacts/{namespace}/{content_hash}"
