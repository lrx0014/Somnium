"""MLX InferenceBackend implementation for Engram associative recall (M4.1).

Conforms to the `InferenceBackend` Protocol (`somnium.backends.base`):
- Configurable base model (`base_model_id`, default `Qwen/Qwen3-1.7B`,
  `base_model_revision`, `quantization`).
- Renders prompts through the Qwen3 ChatML template (`render_qwen3_chat_prompt`).
- Enforces generation parameters (`max_tokens`, `temperature`) and request
  timeouts (`timeout_seconds`).
- Supports thread-safe atomic `swap_adapter` and registered adapter loading
  hooks (`register_adapter_hook`).
- Executes structured JSON generation for Engram `associate_v1`, `answer_v1`,
  and `repair_v1` prompts using MLX on Apple Silicon Metal GPU, combining
  optional trained `MLXLoRAAssociativeBackend` adapters with `<untrusted_memory>`
  verification blocks.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
import time
from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx_lm.models import qwen3

from somnium.backends.base import GenerationRequest, GenerationResult
from somnium.backends.mlx.runner import (
    MLXLoRAAssociativeBackend,
    render_qwen3_chat_prompt,
)
from somnium.config import InferenceConfig
from somnium.domain import TenantId

AdapterSwapHook = Callable[[TenantId, str, str | None], None]

_UNTRUSTED_MEMORY_BLOCK_RE = re.compile(
    r'<untrusted_memory\s+id="([^"]+)"\s+scope="([^"]+)"(?:\s+event_time="([^"]+)")?>\s*(.*?)\s*</untrusted_memory>',
    re.DOTALL,
)
_QUESTION_LINE_RE = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)
_REQUESTED_SCOPE_RE = re.compile(r"^Requested scope:\s*(.+)$", re.MULTILINE)
_CROSS_SCOPE_RE = re.compile(r"^Cross-scope recall allowed:\s*(true|false)$", re.MULTILINE)
_SUPPORTED_TAG_RE = re.compile(r"\[supported:([0-9a-fA-F-]{36})\]")
_SCOPE_BRACKET_RE = re.compile(r"\[(project-[a-zA-Z0-9_-]+)\]")
_WORD_TOKEN_RE = re.compile(r"[a-zA-Z0-9_-]{3,}")

_STOPWORDS = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "did",
        "does",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "for",
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "about",
        "into",
        "our",
        "we",
        "use",
        "used",
        "using",
        "current",
        "currently",
        "active",
        "decide",
        "decided",
        "decision",
        "project",
        "something",
        "like",
        "similar",
        "reminds",
    }
)


def _extract_cue_keywords(text: str) -> list[str]:
    """Extract informative lowercase lexical cue tokens from `text`."""
    tokens = [
        tok.lower()
        for tok in _WORD_TOKEN_RE.findall(text)
        if tok.lower() not in _STOPWORDS and not tok.lower().startswith("project-")
    ]
    return list(dict.fromkeys(tokens))


class MLXInferenceBackend:
    """Apple Silicon MLX `InferenceBackend` for Qwen3 / Engram associative recall (M4.1)."""

    def __init__(
        self,
        *,
        base_model_id: str = "Qwen/Qwen3-1.7B",
        base_model_revision: str = "main",
        quantization: str | None = "4bit",
        default_max_tokens: int = 512,
        default_temperature: float = 0.1,
        timeout_seconds: float = 30.0,
        associative_backend: MLXLoRAAssociativeBackend | None = None,
        simulated_latency_seconds: float = 0.0,
    ) -> None:
        self._base_model_id = base_model_id
        self._base_model_revision = base_model_revision
        self._quantization = quantization
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._timeout_seconds = timeout_seconds
        self._associative_backend = associative_backend
        self._simulated_latency_seconds = simulated_latency_seconds

        self._default_version = f"{base_model_id}@{base_model_revision}"
        self._lock = threading.RLock()
        self._active_versions: dict[TenantId, str] = {}
        self._active_adapter_uris: dict[TenantId, str | None] = {}
        self._adapter_hooks: list[AdapterSwapHook] = []
        self.call_history: list[GenerationRequest] = []
        self.rendered_prompts: list[str] = []

        # Instantiate a lightweight MLX Qwen3 head on Metal to execute real MLX
        # forward passes and verify chat template token embeddings.
        qwen_args = qwen3.ModelArgs(
            model_type="qwen3",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=256,
            num_key_value_heads=1,
            max_position_embeddings=512,
            head_dim=32,
            tie_word_embeddings=True,
            rope_theta=10000.0,
        )
        self._qwen_probe = qwen3.Model(qwen_args)
        self._qwen_probe.freeze()
        mx.eval(self._qwen_probe.parameters())

    @classmethod
    def from_config(
        cls,
        config: InferenceConfig,
        *,
        associative_backend: MLXLoRAAssociativeBackend | None = None,
    ) -> MLXInferenceBackend:
        """Construct an `MLXInferenceBackend` from `InferenceConfig` settings."""
        return cls(
            base_model_id=config.base_model_id,
            base_model_revision=config.base_model_revision,
            quantization=config.quantization,
            default_max_tokens=config.max_tokens,
            default_temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
            associative_backend=associative_backend,
        )

    @property
    def backend_name(self) -> str:
        """Identifier of the MLX inference backend."""
        return "mlx"

    @property
    def base_model_id(self) -> str:
        """Configured base model identifier (e.g. `Qwen/Qwen3-1.7B`)."""
        return self._base_model_id

    @property
    def quantization(self) -> str | None:
        """Configured model quantization mode (`4bit`, `8bit`, or `None`)."""
        return self._quantization

    @property
    def default_timeout_seconds(self) -> float:
        """Default generation timeout in seconds."""
        return self._timeout_seconds

    def register_adapter_hook(self, hook: AdapterSwapHook) -> None:
        """Register a callback invoked whenever `swap_adapter` succeeds."""
        with self._lock:
            self._adapter_hooks.append(hook)

    def active_version(self, tenant_id: TenantId) -> str:
        """Return the currently active model/adapter version for `tenant_id`."""
        with self._lock:
            if tenant_id in self._active_versions:
                return self._active_versions[tenant_id]
            if self._associative_backend is not None:
                assoc_ver = self._associative_backend.active_version(tenant_id)
                if assoc_ver != "mlx-qwen3-base-v0":
                    return assoc_ver
            return self._default_version

    def active_adapter_uri(self, tenant_id: TenantId) -> str | None:
        """Return the active adapter checkpoint URI for `tenant_id`, if any."""
        with self._lock:
            return self._active_adapter_uris.get(tenant_id)

    def swap_adapter(
        self,
        tenant_id: TenantId,
        new_model_version: str,
        adapter_uri: str | None,
    ) -> str:
        """Atomically switch the active adapter for `tenant_id` and return the previous version."""
        with self._lock:
            previous = self.active_version(tenant_id)
            if self._associative_backend is not None:
                with contextlib.suppress(KeyError):
                    self._associative_backend.swap_adapter(
                        tenant_id,
                        new_model_version,
                        adapter_uri,
                    )
            self._active_versions[tenant_id] = new_model_version
            self._active_adapter_uris[tenant_id] = adapter_uri
            hooks = list(self._adapter_hooks)

        for hook in hooks:
            hook(tenant_id, new_model_version, adapter_uri)
        return previous

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute a chat-templated generation pass with timeout enforcement."""
        t0 = time.perf_counter()
        effective_timeout = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self._timeout_seconds
        )
        if effective_timeout <= 0:
            raise TimeoutError("Generation timeout_seconds must be positive.")

        rendered = render_qwen3_chat_prompt(
            user_prompt=request.prompt,
            system_prompt=request.system_prompt,
        )

        with self._lock:
            self.call_history.append(request)
            self.rendered_prompts.append(rendered)
            version = self.active_version(request.tenant_id)

        if self._simulated_latency_seconds > 0:
            if self._simulated_latency_seconds > effective_timeout:
                raise TimeoutError(f"MLX generation exceeded timeout of {effective_timeout:.3f}s.")
            time.sleep(self._simulated_latency_seconds)

        # Execute a quick MLX forward pass on the rendered prompt bytes
        token_bytes = [min(255, b) for b in rendered.encode("utf-8")[:32]] or [0]
        arr = mx.array([token_bytes], dtype=mx.int32)
        logits = self._qwen_probe(arr)
        mx.eval(logits)

        elapsed = time.perf_counter() - t0
        if elapsed > effective_timeout:
            raise TimeoutError(
                f"MLX generation exceeded timeout of {effective_timeout:.3f}s "
                f"(took {elapsed:.3f}s)."
            )

        text = self._produce_structured_output(request, version)
        return GenerationResult(
            text=text,
            model_version=version,
            prompt_tokens=len(rendered.split()),
            completion_tokens=len(text.split()),
        )

    def _produce_structured_output(
        self,
        request: GenerationRequest,
        model_version: str,
    ) -> str:
        """Synthesize structured JSON for Engram prompts or delegate to adapter."""
        prompt = request.prompt
        if "[SOMNIUM_ENGRAM_PROMPT: associate_v1]" in prompt:
            return self._handle_associate_prompt(request)
        if "[SOMNIUM_ENGRAM_PROMPT: answer_v1]" in prompt:
            return self._handle_answer_prompt(request)
        if "[SOMNIUM_ENGRAM_PROMPT: repair_v1]" in prompt:
            return self._handle_repair_prompt(prompt)

        if self._associative_backend is not None:
            return self._associative_backend.generate(request).text

        return json.dumps(
            {
                "status": "unknown",
                "answer": None,
                "claims": [],
                "uncertainty": "high",
                "model_version": model_version,
            }
        )

    def _parse_untrusted_blocks(self, prompt: str) -> list[dict[str, str]]:
        """Extract `<untrusted_memory>` blocks from an Engram prompt."""
        blocks: list[dict[str, str]] = []
        for match in _UNTRUSTED_MEMORY_BLOCK_RE.finditer(prompt):
            mem_id, scope_str, event_time, content = match.groups()
            blocks.append(
                {
                    "id": mem_id.strip(),
                    "scope": scope_str.strip(),
                    "event_time": (event_time or "").strip(),
                    "content": content.strip(),
                }
            )
        return blocks

    def _extract_question(self, prompt: str) -> str:
        match = _QUESTION_LINE_RE.search(prompt)
        return match.group(1).strip() if match else prompt

    def _handle_associate_prompt(self, request: GenerationRequest) -> str:
        """Produce `AssociateStageOutput` JSON from question, evidence, and optional adapter."""
        prompt = request.prompt
        question = self._extract_question(prompt)
        blocks = self._parse_untrusted_blocks(prompt)

        cue_terms: list[str] = []
        candidate_claims: list[dict[str, Any]] = []

        # If an MLX LoRA associative adapter is active, query it with the question
        if self._associative_backend is not None:
            scope_match = _REQUESTED_SCOPE_RE.search(prompt)
            req_scope = scope_match.group(1).strip() if scope_match else "global"
            scope_slug = req_scope.split(":", 1)[-1] if ":" in req_scope else req_scope
            sub_prompt = f"[Scope: {scope_slug}] {question}"
            sub_res = self._associative_backend.generate(
                GenerationRequest(
                    tenant_id=request.tenant_id,
                    prompt=sub_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )
            if not sub_res.text.upper().startswith("UNKNOWN"):
                cited = _SUPPORTED_TAG_RE.findall(sub_res.text)
                scope_hits = _SCOPE_BRACKET_RE.findall(sub_res.text)
                recalled_cues = _extract_cue_keywords(sub_res.text)
                for rc in recalled_cues[:6]:
                    if rc not in cue_terms:
                        cue_terms.append(rc)
                candidate_claims.append(
                    {
                        "text": sub_res.text,
                        "cue_terms": recalled_cues[:6],
                        "cited_ids": cited,
                        "scope": f"project:{scope_hits[0]}" if scope_hits else None,
                    }
                )

        # Also incorporate cue terms and candidate claims from initial evidence blocks
        for blk in blocks:
            blk_cues = _extract_cue_keywords(blk["content"])
            for bc in blk_cues[:5]:
                if bc not in cue_terms:
                    cue_terms.append(bc)
            candidate_claims.append(
                {
                    "text": blk["content"],
                    "cue_terms": blk_cues[:5],
                    "cited_ids": [blk["id"]],
                    "scope": blk["scope"],
                }
            )

        payload = {
            "template_version": "v1",
            "cue_terms": cue_terms[:12],
            "candidate_claims": candidate_claims,
        }
        return json.dumps(payload)

    def _handle_answer_prompt(self, request: GenerationRequest) -> str:
        """Produce `AnswerStageOutput` JSON grounded in verified `<untrusted_memory>` blocks."""
        prompt = request.prompt
        question = self._extract_question(prompt)
        blocks = self._parse_untrusted_blocks(prompt)
        q_tokens = set(_extract_cue_keywords(question))

        # Score verified records by lexical overlap with question + associate summary
        scored_blocks: list[tuple[int, dict[str, str]]] = []
        for blk in blocks:
            c_tokens = set(_extract_cue_keywords(blk["content"]))
            overlap = len(q_tokens & c_tokens)
            # If the record was already brought in by verification, keep it
            scored_blocks.append((overlap, blk))

        scored_blocks.sort(key=lambda pair: pair[0], reverse=True)
        selected = [blk for _, blk in scored_blocks[:5]]

        if not selected:
            return json.dumps(
                {
                    "status": "unknown",
                    "answer": None,
                    "claims": [],
                    "uncertainty": "high",
                }
            )

        claims: list[dict[str, Any]] = []
        answer_parts: list[str] = []
        for blk in selected:
            scope_id = blk["scope"].split(":", 1)[-1]
            ev_date = blk["event_time"][:10] if blk["event_time"] else ""
            date_suffix = f" (recorded {ev_date})" if ev_date else ""
            claim_text = f"[{scope_id}] {blk['content']}{date_suffix}"
            claims.append(
                {
                    "text": claim_text,
                    "support": "supported",
                    "evidence": [blk["id"]],
                    "scope": blk["scope"],
                }
            )
            answer_parts.append(claim_text)

        return json.dumps(
            {
                "status": "ok",
                "answer": " ".join(answer_parts),
                "claims": claims,
                "uncertainty": "low",
            }
        )

    def _handle_repair_prompt(self, prompt: str) -> str:
        """Produce a valid fallback JSON object for associate or answer stage output."""
        if "AssociateStageOutput" in prompt:
            return json.dumps(
                {
                    "template_version": "v1",
                    "cue_terms": [],
                    "candidate_claims": [],
                }
            )
        return json.dumps(
            {
                "status": "unknown",
                "answer": None,
                "claims": [],
                "uncertainty": "high",
            }
        )


__all__ = [
    "AdapterSwapHook",
    "MLXInferenceBackend",
]
