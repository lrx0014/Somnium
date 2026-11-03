"""Dataset builder and manifest repository for Dream Engine training (M5.1).

Per `design.md` §8.2 and §8.3:
- Datasets are constructed exclusively from verified, `training_eligible` records.
- Records with `sensitivity == HIGH` or `is_deleted == True` are strictly excluded.
- Produces diverse training examples (direct QA, scope attribution, incomplete cue,
  temporal correction, evidence reading / claim labeling, and explicit unknown/abstention).
- Deterministic splitting by underlying fact group prevents data leakage between
  training and evaluation.
- Dataset manifests are content-addressed by SHA-256 and stored in the `ArtifactStore`.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, select

from somnium.domain import AuthContext, MemoryRecord, SensitivityLevel, TenantId
from somnium.dream.augmentation import (
    AUGMENTATION_VERSION,
    AugmentationCache,
    _extract_attribute_and_rationale,
)
from somnium.eval.questions import TrainingExample
from somnium.hippocore.schema import dataset_manifests, training_examples
from somnium.hippocore.store import HippocoreStore
from somnium.interfaces import ArtifactStore

DEFAULT_DATASET_GENERATOR_VERSION = f"builder-v1+{AUGMENTATION_VERSION}"

_DEFAULT_TRAINING_SYSTEM_PROMPT = (
    "You are Engram, the dedicated associative memory intelligence for Somnium. "
    "Answer concisely using verified memories, state the exact project scope and "
    "current decision, label claims as [supported:<memory_id>] when grounded in "
    "supplied evidence or [unverified] when recalled parametrically, and return "
    "UNKNOWN when no memory supports an answer."
)

_UNKNOWN_QUERY_TEMPLATES: tuple[str, ...] = (
    "What quantum cryptographic acceleration protocol was adopted by {scope_id}?",
    "Which satellite uplink ground station provider is configured for {scope_id}?",
    "Who was the third-party hardware security penetration auditor for {scope_id}?",
    "What bespoke custom silicon ASIC tape-out version is utilized in {scope_id}?",
)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable training dataset manifest with content hash and lineage (M5.1)."""

    dataset_hash: str
    tenant_id: TenantId
    generator_version: str
    training_mode: str  # "incremental" or "full"
    parent_manifest_hash: str | None
    source_record_ids: tuple[str, ...]
    example_counts: dict[str, int]
    examples: tuple[TrainingExample, ...]
    created_at: str
    artifact_uri: str

    def to_jsonl(self) -> str:
        """Serialize examples to deterministic JSONL format."""
        return "\n".join(json.dumps(ex.to_dict(), sort_keys=True) for ex in self.examples) + "\n"


def _fact_group_for_record(record: MemoryRecord) -> str:
    """Compute a stable fact_group_id so revisions of the same fact share a group."""
    if record.supersedes:
        # Group with oldest cited ancestor or first superseded record
        root = str(sorted(record.supersedes)[0])
        return f"fg:{root}"
    return f"fg:{record.id}"


def _construct_examples_for_record(
    record: MemoryRecord,
    *,
    fact_group_id: str,
    augmentation_cache: AugmentationCache,
    system_prompt: str,
    all_records_by_id: dict[UUID, MemoryRecord],
    split: str,
    now_iso: str,
) -> list[TrainingExample]:
    """Construct QA training examples for a single eligible MemoryRecord."""
    rec_id_str = str(record.id)
    scope_str = str(record.scope)
    scope_id = record.scope.id
    attr, _ = _extract_attribute_and_rationale(record)
    augmented = augmentation_cache.get_or_create(record)
    date_str = record.event_time.strftime("%Y-%m-%d")

    examples: list[TrainingExample] = []

    # 1. Direct factual QA
    examples.append(
        TrainingExample(
            example_id=f"ex:{rec_id_str}:direct",
            category="direct",
            fact_group_id=fact_group_id,
            tenant_id=record.tenant_id,
            scope=scope_str,
            temporal_slice=1,
            source_record_ids=(rec_id_str,),
            system_prompt=system_prompt,
            user_prompt=f"[Scope: {scope_id}] {augmented.direct_question}",
            assistant_response=(
                f"In {scope_id}, {attr} is {record.content} on {date_str} [supported:{rec_id_str}]."
            ),
            split=split,
            created_at=now_iso,
        )
    )

    # 2. Scope attribution QA (paraphrased to enforce project boundary)
    examples.append(
        TrainingExample(
            example_id=f"ex:{rec_id_str}:scope_attr",
            category="scope_attribution",
            fact_group_id=fact_group_id,
            tenant_id=record.tenant_id,
            scope=scope_str,
            temporal_slice=1,
            source_record_ids=(rec_id_str,),
            system_prompt=system_prompt,
            user_prompt=f"Specifically for {scope_id}, {augmented.paraphrased_question}",
            assistant_response=(
                f"Specifically for {scope_id}, {attr} is {record.content} [supported:{rec_id_str}]."
            ),
            split=split,
            created_at=now_iso,
        )
    )

    # 3. Incomplete cue associative QA
    examples.append(
        TrainingExample(
            example_id=f"ex:{rec_id_str}:incomplete_cue",
            category="incomplete_cue",
            fact_group_id=fact_group_id,
            tenant_id=record.tenant_id,
            scope=scope_str,
            temporal_slice=1,
            source_record_ids=(rec_id_str,),
            system_prompt=system_prompt,
            user_prompt=f"[Scope: {scope_id}] {augmented.incomplete_cue_question}",
            assistant_response=(
                f"That refers to {attr} in {scope_id}: {record.content} [supported:{rec_id_str}]."
            ),
            split=split,
            created_at=now_iso,
        )
    )

    # 4. Temporal / Correction QA if supersedes an older record
    if record.supersedes:
        superseded_id = record.supersedes[0]
        superseded_record = all_records_by_id.get(superseded_id)
        if superseded_record is not None:
            old_id_str = str(superseded_id)
            examples.append(
                TrainingExample(
                    example_id=f"ex:{rec_id_str}:temporal_correction",
                    category="temporal_correction",
                    fact_group_id=fact_group_id,
                    tenant_id=record.tenant_id,
                    scope=scope_str,
                    temporal_slice=2,
                    source_record_ids=(rec_id_str, old_id_str),
                    system_prompt=system_prompt,
                    user_prompt=(
                        f"[Scope: {scope_id}] What is the current decision regarding "
                        f"{attr} for {scope_id}, and what did it replace?"
                    ),
                    assistant_response=(
                        f"The current decision for {scope_id} is {record.content}, which "
                        f"superseded the earlier decision ({superseded_record.content}) on "
                        f"{date_str} [supported:{rec_id_str}]."
                    ),
                    split=split,
                    created_at=now_iso,
                )
            )

    # 5. Evidence reading / Claim labeling QA
    examples.append(
        TrainingExample(
            example_id=f"ex:{rec_id_str}:evidence_reading",
            category="evidence_reading",
            fact_group_id=fact_group_id,
            tenant_id=record.tenant_id,
            scope=scope_str,
            temporal_slice=1,
            source_record_ids=(rec_id_str,),
            system_prompt=system_prompt,
            user_prompt=(
                f"[Scope: {scope_id}]\n"
                f'<untrusted_memory id="{rec_id_str}" scope="{scope_str}">\n'
                f"{record.content}\n"
                f"</untrusted_memory>\n"
                f"Question: Based on the supplied evidence, what is the {attr} for {scope_id}?"
            ),
            assistant_response=(
                f"Based on the supplied evidence, {record.content} on {date_str} "
                f"[supported:{rec_id_str}]."
            ),
            split=split,
            created_at=now_iso,
        )
    )

    return examples


def build_tenant_dataset(
    auth: AuthContext,
    store: HippocoreStore,
    artifact_store: ArtifactStore,
    *,
    augmentation_cache: AugmentationCache | None = None,
    training_mode: str = "incremental",
    parent_manifest_hash: str | None = None,
    replay_ratio: float = 0.4,
    eval_split_ratio: float = 0.15,
    generator_version: str = DEFAULT_DATASET_GENERATOR_VERSION,
    system_prompt: str = _DEFAULT_TRAINING_SYSTEM_PROMPT,
    seed: int = 42,
) -> DatasetManifest:
    """Build a deterministic, backend-neutral training dataset from Hippocore records.

    Enforces:
    - Only `training_eligible == True` and `sensitivity == NORMAL` records are used.
    - Sensitive (`HIGH`) and `DELETED` records are never included.
    - Dataset manifest is content-addressed by SHA-256 and stored in `artifact_store`.
    - Manifest and example metadata are persisted in Hippocore SQLite tables.
    """
    if training_mode not in ("incremental", "full"):
        raise ValueError(f"Invalid training_mode '{training_mode}'. Must be incremental or full.")

    if augmentation_cache is None:
        augmentation_cache = AugmentationCache(artifact_store=artifact_store)

    # 1. Fetch training-eligible records across tenant's authorized scopes
    records = store.list_training_eligible(auth, include_superseded=True)
    eligible = [
        r
        for r in records
        if r.tenant_id == auth.tenant_id
        and r.training_eligible
        and r.sensitivity == SensitivityLevel.NORMAL
    ]

    all_records_by_id = {r.id: r for r in eligible}

    # 2. Determine incremental vs full retrain record selection
    parent_record_ids: set[str] = set()
    if training_mode == "incremental" and parent_manifest_hash:
        with store.engine.connect() as conn:
            stmt = select(dataset_manifests.c.source_record_ids_json).where(
                dataset_manifests.c.dataset_hash == parent_manifest_hash
            )
            row = conn.execute(stmt).first()
            if row is not None and row[0]:
                parent_record_ids = set(json.loads(row[0]))

    if training_mode == "incremental" and parent_record_ids:
        new_records = [r for r in eligible if str(r.id) not in parent_record_ids]
        old_records = [r for r in eligible if str(r.id) in parent_record_ids]

        # Replay sampling from historical records
        rng = random.Random(seed)
        sample_count = max(1, int(len(old_records) * replay_ratio)) if old_records else 0
        sampled_old = rng.sample(old_records, min(sample_count, len(old_records)))
        active_dataset_records = new_records + sampled_old
    else:
        active_dataset_records = list(eligible)

    # 3. Partition into train / eval splits by fact_group_id
    fact_groups: set[str] = set()
    for r in active_dataset_records:
        fact_groups.add(_fact_group_for_record(r))

    eval_fact_groups: set[str] = set()
    if len(fact_groups) > 1 and eval_split_ratio > 0.0:
        for fg in sorted(fact_groups):
            h = int(hashlib.md5(f"{seed}:{fg}".encode()).hexdigest(), 16)
            if (h % 100) < int(eval_split_ratio * 100):
                eval_fact_groups.add(fg)

    now_iso = datetime.now(tz=UTC).isoformat()
    examples: list[TrainingExample] = []

    # 4. Construct examples for each record
    for record in active_dataset_records:
        fg = _fact_group_for_record(record)
        split = "eval" if fg in eval_fact_groups else "train"
        rec_examples = _construct_examples_for_record(
            record,
            fact_group_id=fg,
            augmentation_cache=augmentation_cache,
            system_prompt=system_prompt,
            all_records_by_id=all_records_by_id,
            split=split,
            now_iso=now_iso,
        )
        examples.extend(rec_examples)

    # 5. Add unknown / abstention examples per scope
    distinct_scopes = {r.scope for r in active_dataset_records}
    for scope in sorted(distinct_scopes, key=lambda s: str(s)):
        scope_id = scope.id
        for idx, tmpl in enumerate(_UNKNOWN_QUERY_TEMPLATES):
            ex_id = f"ex:unknown:{scope_id}:{idx}"
            examples.append(
                TrainingExample(
                    example_id=ex_id,
                    category="unknown",
                    fact_group_id="fg:unknown",
                    tenant_id=auth.tenant_id,
                    scope=str(scope),
                    temporal_slice=1,
                    source_record_ids=(),
                    system_prompt=system_prompt,
                    user_prompt=f"[Scope: {scope_id}] {tmpl.format(scope_id=scope_id)}",
                    assistant_response="UNKNOWN",
                    split="train" if idx % 2 == 0 else "eval",
                    created_at=now_iso,
                )
            )

    # 6. Sort examples deterministically by example_id
    examples.sort(key=lambda ex: ex.example_id)

    # 7. Serialize to canonical JSONL and compute dataset_hash
    jsonl_str = "\n".join(json.dumps(ex.to_dict(), sort_keys=True) for ex in examples) + (
        "\n" if examples else ""
    )
    jsonl_bytes = jsonl_str.encode("utf-8")
    dataset_hash = hashlib.sha256(jsonl_bytes).hexdigest()

    # 8. Persist JSONL artifact into ArtifactStore under 'datasets'
    artifact_store.put_bytes("datasets", jsonl_bytes)
    artifact_uri = artifact_store.uri_for("datasets", dataset_hash)

    # 9. Compute summary metrics and persist to database
    source_ids = tuple(sorted(str(r.id) for r in active_dataset_records))
    category_counts: dict[str, int] = {}
    for ex in examples:
        category_counts[ex.category] = category_counts.get(ex.category, 0) + 1

    manifest = DatasetManifest(
        dataset_hash=dataset_hash,
        tenant_id=auth.tenant_id,
        generator_version=generator_version,
        training_mode=training_mode,
        parent_manifest_hash=parent_manifest_hash,
        source_record_ids=source_ids,
        example_counts=category_counts,
        examples=tuple(examples),
        created_at=now_iso,
        artifact_uri=artifact_uri,
    )

    save_dataset_manifest_to_db(store.engine, manifest)
    return manifest


def save_dataset_manifest_to_db(engine: Engine, manifest: DatasetManifest) -> None:
    """Idempotently insert `manifest` and its `examples` into Hippocore tables."""
    train_cnt = sum(1 for ex in manifest.examples if ex.split == "train")
    held_out_cnt = sum(1 for ex in manifest.examples if ex.split != "train")
    with engine.begin() as conn:
        existing = conn.execute(
            select(dataset_manifests.c.dataset_hash).where(
                dataset_manifests.c.dataset_hash == manifest.dataset_hash
            )
        ).first()
        if existing is None:
            conn.execute(
                dataset_manifests.insert().values(
                    dataset_hash=manifest.dataset_hash,
                    tenant_id=manifest.tenant_id,
                    mode=manifest.training_mode,
                    parent_manifest_hash=manifest.parent_manifest_hash,
                    generator_version=manifest.generator_version,
                    artifact_uri=manifest.artifact_uri,
                    source_record_ids_json=json.dumps(list(manifest.source_record_ids)),
                    example_count=len(manifest.examples),
                    train_count=train_cnt,
                    held_out_count=held_out_cnt,
                    category_counts_json=json.dumps(manifest.example_counts),
                    created_at=manifest.created_at,
                )
            )

            # Insert training examples
            for ex in manifest.examples:
                conn.execute(
                    training_examples.insert().values(
                        example_id=ex.example_id,
                        dataset_hash=manifest.dataset_hash,
                        tenant_id=manifest.tenant_id,
                        category=ex.category,
                        fact_group_id=ex.fact_group_id,
                        scope=ex.scope,
                        split=ex.split,
                        source_record_ids_json=json.dumps(list(ex.source_record_ids)),
                        system_prompt=ex.system_prompt,
                        user_prompt=ex.user_prompt,
                        assistant_response=ex.assistant_response,
                        created_at=ex.created_at or manifest.created_at,
                    )
                )


def load_dataset_manifest(
    engine: Engine,
    artifact_store: ArtifactStore,
    dataset_hash: str,
) -> DatasetManifest:
    """Load a DatasetManifest from Hippocore SQLite and ArtifactStore."""
    with engine.connect() as conn:
        row = conn.execute(
            select(dataset_manifests).where(dataset_manifests.c.dataset_hash == dataset_hash)
        ).first()
        if row is None:
            raise KeyError(f"Dataset manifest not found: {dataset_hash}")

        m = row._mapping
        raw_bytes = artifact_store.get_bytes("datasets", dataset_hash)
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

        return DatasetManifest(
            dataset_hash=dataset_hash,
            tenant_id=str(m["tenant_id"]),
            generator_version=str(m["generator_version"]),
            training_mode=str(m["mode"]),
            parent_manifest_hash=str(m["parent_manifest_hash"])
            if m["parent_manifest_hash"]
            else None,
            source_record_ids=tuple(json.loads(str(m["source_record_ids_json"]))),
            example_counts=dict(json.loads(str(m["category_counts_json"]))),
            examples=tuple(examples),
            created_at=str(m["created_at"]),
            artifact_uri=str(m["artifact_uri"]),
        )


def list_dataset_manifests(engine: Engine, tenant_id: TenantId) -> Sequence[dict[str, Any]]:
    """List dataset manifest records for a tenant ordered by created_at descending."""
    with engine.connect() as conn:
        stmt = (
            select(dataset_manifests)
            .where(dataset_manifests.c.tenant_id == tenant_id)
            .order_by(dataset_manifests.c.created_at.desc())
        )
        rows = conn.execute(stmt).fetchall()
        return [dict(r._mapping) for r in rows]


__all__ = [
    "DEFAULT_DATASET_GENERATOR_VERSION",
    "DatasetManifest",
    "build_tenant_dataset",
    "list_dataset_manifests",
    "load_dataset_manifest",
    "save_dataset_manifest_to_db",
]
