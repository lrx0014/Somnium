"""Content-hashed augmentation cache for training dataset construction (M5.2).

Per `design.md` §8.2:
- Synthesized questions and paraphrases are training augmentation, never new
  episodic facts.
- Any generated augmentation is cached and content-hashed (`SHA-256`) so that
  rebuilding a dataset manifest from the same input records yields the exact
  same `dataset_hash`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from somnium.domain import MemoryRecord
from somnium.interfaces import ArtifactStore

AUGMENTATION_VERSION = "aug-v1"

_SENTENCE_SPLIT_RE = re.compile(r"\b(?:because|due to|to support|for|so that)\b", re.IGNORECASE)
_SELECTED_RE = re.compile(
    r"(?:selected|chose|picked|migrated\s+to|adopted|uses|standardized\s+on)\s+"
    r"([A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)"
    r"(?:\s+(?:instead\s+of|over)\s+[A-Za-z0-9_.+-]+(?:\s+[A-Za-z0-9_.+-]+)?)?"
    r"\s+(?:as\s+(?:its|the)\s+|for\s+)([A-Za-z0-9_\s-]+?)(?:[.,;]|$|\s+because|\s+due\s+to)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-zA-Z0-9_-]{3,}")


@dataclass(frozen=True, slots=True)
class AugmentedQuestions:
    """Cached question variants synthesized for a single `MemoryRecord`."""

    cache_key_hash: str
    generator_version: str
    attribute_hint: str
    direct_question: str
    paraphrased_question: str
    incomplete_cue_question: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a deterministic dictionary."""
        return {
            "cache_key_hash": self.cache_key_hash,
            "generator_version": self.generator_version,
            "attribute_hint": self.attribute_hint,
            "direct_question": self.direct_question,
            "paraphrased_question": self.paraphrased_question,
            "incomplete_cue_question": self.incomplete_cue_question,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AugmentedQuestions:
        """Deserialize from a dictionary."""
        return cls(
            cache_key_hash=str(data["cache_key_hash"]),
            generator_version=str(data["generator_version"]),
            attribute_hint=str(data["attribute_hint"]),
            direct_question=str(data["direct_question"]),
            paraphrased_question=str(data["paraphrased_question"]),
            incomplete_cue_question=str(data["incomplete_cue_question"]),
        )


def compute_augmentation_key(
    record: MemoryRecord,
    *,
    generator_version: str = AUGMENTATION_VERSION,
) -> str:
    """Compute a deterministic SHA-256 hex key from `record`'s scope, kind, and content."""
    canonical = (
        f"{generator_version}|{record.scope.kind.value}:{record.scope.id}|"
        f"{record.kind.value}|{' '.join(record.content.split())}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extract_attribute_and_rationale(record: MemoryRecord) -> tuple[str, str]:
    """Extract an attribute phrase and rationale/cue phrase from `record.content`."""
    content = " ".join(record.content.split())
    sel_match = _SELECTED_RE.search(content)
    if sel_match:
        attr = sel_match.group(2).strip(" .,;")
    elif record.tags:
        attr = " ".join(record.tags[:2])
    else:
        tokens = [
            t
            for t in _WORD_RE.findall(content)
            if t.lower() not in {"project", record.scope.id.lower(), "selected", "chose"}
        ]
        attr = " ".join(tokens[:3]) if tokens else f"{record.kind.value} configuration"

    parts = _SENTENCE_SPLIT_RE.split(content, maxsplit=1)
    if len(parts) > 1 and parts[1].strip():
        rationale = parts[1].strip(" .,;")
    elif record.tags:
        rationale = f"our work involving {', '.join(record.tags)}"
    else:
        rationale = f"the {attr} decision"

    return attr, rationale


def synthesize_default_questions(
    record: MemoryRecord,
    *,
    cache_key_hash: str,
    generator_version: str = AUGMENTATION_VERSION,
) -> AugmentedQuestions:
    """Synthesize deterministic direct, paraphrase, and incomplete-cue questions."""
    scope_id = record.scope.id
    attr, rationale = _extract_attribute_and_rationale(record)

    direct_q = f"What did {scope_id} select for its {attr}?"
    paraphrase_q = (
        f"For {scope_id}, which technology or decision is currently in place as the {attr}?"
    )
    incomplete_cue_q = (
        f"In {scope_id}, I vaguely remember a decision driven by {rationale}. "
        f"What did we settle on?"
    )

    return AugmentedQuestions(
        cache_key_hash=cache_key_hash,
        generator_version=generator_version,
        attribute_hint=attr,
        direct_question=direct_q,
        paraphrased_question=paraphrase_q,
        incomplete_cue_question=incomplete_cue_q,
    )


class AugmentationCache:
    """Content-addressed cache for training question augmentations (M5.2)."""

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        *,
        generator_version: str = AUGMENTATION_VERSION,
        custom_synthesizer: Callable[[MemoryRecord, str], AugmentedQuestions] | None = None,
    ) -> None:
        self._artifact_store = artifact_store
        self._generator_version = generator_version
        self._custom_synthesizer = custom_synthesizer
        self._memory_cache: dict[str, AugmentedQuestions] = {}
        self._key_to_blob_hash: dict[str, str] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    @property
    def generator_version(self) -> str:
        """Version tag of the question synthesizer."""
        return self._generator_version

    def get_or_create(self, record: MemoryRecord) -> AugmentedQuestions:
        """Return cached `AugmentedQuestions` for `record`, synthesizing on miss."""
        key_hash = compute_augmentation_key(
            record,
            generator_version=self._generator_version,
        )
        if key_hash in self._memory_cache:
            self.cache_hits += 1
            return self._memory_cache[key_hash]

        if self._artifact_store is not None and key_hash in self._key_to_blob_hash:
            blob_hash = self._key_to_blob_hash[key_hash]
            if self._artifact_store.exists("augmentations", blob_hash):
                raw = self._artifact_store.get_bytes("augmentations", blob_hash)
                loaded = AugmentedQuestions.from_dict(json.loads(raw.decode("utf-8")))
                self._memory_cache[key_hash] = loaded
                self.cache_hits += 1
                return loaded

        self.cache_misses += 1
        if self._custom_synthesizer is not None:
            augmented = self._custom_synthesizer(record, key_hash)
        else:
            augmented = synthesize_default_questions(
                record,
                cache_key_hash=key_hash,
                generator_version=self._generator_version,
            )

        self._memory_cache[key_hash] = augmented
        if self._artifact_store is not None:
            payload_bytes = json.dumps(augmented.to_dict(), sort_keys=True).encode("utf-8")
            blob_hash = self._artifact_store.put_bytes("augmentations", payload_bytes)
            self._key_to_blob_hash[key_hash] = blob_hash

        return augmented


__all__ = [
    "AUGMENTATION_VERSION",
    "AugmentationCache",
    "AugmentedQuestions",
    "compute_augmentation_key",
    "synthesize_default_questions",
]
