"""Question and backend-neutral training dataset generator (M1.2, design §8.2).

Constructs:
1. Benchmark evaluation questions across all required categories (`direct`,
   `paraphrase`, `incomplete_cue`, `temporal`, `correction`, `scope_attribution`,
   `cross_scope`, `unknown`, `deletion_probe`, `isolation_probe`), split by
   underlying `fact_group_id` into `train` (in-distribution) and `held_out`
   (unseen facts).
2. Backend-neutral training datasets (`TrainingExample`, `DatasetManifest`) from
   verified, `training_eligible` records only, with deterministic SHA-256
   manifest hashes.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from somnium.domain import (
    RecallMode,
    Scope,
    ScopeKind,
    SensitivityLevel,
    TenantId,
)
from somnium.eval.world import (
    ALL_PRIMARY_SCOPES,
    PRIMARY_TENANT,
    PROJECT_LYRA,
    PROJECT_ORION,
    PROJECT_ORION_CORE,
    FactBlueprint,
    SyntheticWorld,
)


class QuestionCategory(StrEnum):
    """Benchmark question categories aligned with design §8.2 and §12."""

    DIRECT = "direct"
    PARAPHRASE = "paraphrase"
    INCOMPLETE_CUE = "incomplete_cue"
    TEMPORAL = "temporal"
    CORRECTION = "correction"
    SCOPE_ATTRIBUTION = "scope_attribution"
    CROSS_SCOPE = "cross_scope"
    UNKNOWN = "unknown"
    DELETION_PROBE = "deletion_probe"
    ISOLATION_PROBE = "isolation_probe"


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    """A single evaluation question with ground-truth expectations."""

    question_id: str
    category: QuestionCategory
    fact_group_id: str
    tenant_id: TenantId
    requested_scope: Scope | None
    authorized_scopes: tuple[Scope, ...]
    question: str
    expected_values: tuple[str, ...]
    forbidden_values: tuple[str, ...] = ()
    expected_record_ids: tuple[UUID, ...] = ()
    expected_scope: Scope | None = None
    expected_event_date: str | None = None
    expect_abstention: bool = False
    temporal_slice: int = 1
    fact_split: str = "train"  # "train" (seen fact family) or "held_out" (unseen fact family)
    preferred_mode: RecallMode = RecallMode.AUTO


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """Backend-neutral supervised fine-tuning example (rendered to chat template at train time)."""

    example_id: str
    category: str
    fact_group_id: str
    tenant_id: TenantId
    scope: str
    temporal_slice: int
    source_record_ids: tuple[str, ...]
    system_prompt: str
    user_prompt: str
    assistant_response: str

    def to_dict(self) -> dict[str, object]:
        """Serialize to a deterministic JSON-compatible dictionary."""
        return {
            "example_id": self.example_id,
            "category": self.category,
            "fact_group_id": self.fact_group_id,
            "tenant_id": self.tenant_id,
            "scope": self.scope,
            "temporal_slice": self.temporal_slice,
            "source_record_ids": list(self.source_record_ids),
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "assistant_response": self.assistant_response,
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable training dataset manifest with content hash and lineage."""

    suite_name: str
    seed: int
    temporal_slice: int
    mode: str  # "slice1", "incremental_replay", or "full_retrain"
    examples: tuple[TrainingExample, ...]
    source_record_ids: tuple[str, ...]
    dataset_hash: str

    def to_jsonl(self) -> str:
        """Serialize examples to deterministic JSONL format."""
        return "\n".join(json.dumps(ex.to_dict(), sort_keys=True) for ex in self.examples) + "\n"


_UNANSWERABLE_QUESTIONS: tuple[tuple[Scope, str], ...] = (
    (
        PROJECT_ORION,
        "What Kubernetes ingress service mesh controller did project-orion deploy in 2026?",
    ),
    (
        PROJECT_ORION_CORE,
        "Which mobile iOS push notification provider is configured for project-orion-core?",
    ),
    (
        PROJECT_LYRA,
        "What Solidity smart contract compiler version did project-lyra standardize on?",
    ),
    (
        PROJECT_VEGA := Scope(kind=ScopeKind.PROJECT, id="project-vega"),
        "What GraphQL federation gateway router does project-vega run in production?",
    ),
    (
        PROJECT_ORION,
        "Who was the external penetration testing vendor hired for project-orion in May 2026?",
    ),
    (
        PROJECT_LYRA,
        "What CDN edge WAF rule set ID is active on project-lyra's public static assets?",
    ),
)


def generate_benchmark_questions(
    world: SyntheticWorld,
    *,
    max_slice: int = 2,
) -> tuple[BenchmarkQuestion, ...]:
    """Generate the benchmark evaluation question suite for `world` up to `max_slice`."""
    blueprints = world.blueprints_for_slice(max_slice=max_slice)
    questions: list[BenchmarkQuestion] = []
    q_idx = 0

    def next_qid(cat: QuestionCategory) -> str:
        nonlocal q_idx
        q_idx += 1
        return f"q-{cat.value}-{q_idx:04d}"

    active_normal_bps = [
        bp
        for bp in blueprints
        if bp.tenant_id == PRIMARY_TENANT
        and bp.sensitivity == SensitivityLevel.NORMAL
        and not bp.is_deleted
    ]

    for bp in active_normal_bps:
        date_str = bp.current_event_time.strftime("%Y-%m-%d")
        scope_label = bp.scope.id

        # 1. Direct factual QA
        questions.append(
            BenchmarkQuestion(
                question_id=next_qid(QuestionCategory.DIRECT),
                category=QuestionCategory.DIRECT,
                fact_group_id=bp.fact_group_id,
                tenant_id=PRIMARY_TENANT,
                requested_scope=bp.scope,
                authorized_scopes=(bp.scope,),
                question=f"What did {scope_label} select for its {bp.attribute}?",
                expected_values=(bp.current_value,),
                forbidden_values=(bp.superseded_value,) if bp.superseded_value else (),
                expected_record_ids=(bp.current_record_id,),
                expected_scope=bp.scope,
                expected_event_date=date_str,
                temporal_slice=bp.temporal_slice,
                fact_split=bp.split,
                preferred_mode=RecallMode.EVIDENCE,
            )
        )

        # 2. Paraphrased question (cue-based recall)
        questions.append(
            BenchmarkQuestion(
                question_id=next_qid(QuestionCategory.PARAPHRASE),
                category=QuestionCategory.PARAPHRASE,
                fact_group_id=bp.fact_group_id,
                tenant_id=PRIMARY_TENANT,
                requested_scope=bp.scope,
                authorized_scopes=ALL_PRIMARY_SCOPES,
                question=(
                    f"For {scope_label}, which technology is currently in place as the "
                    f"{bp.attribute}?"
                ),
                expected_values=(bp.current_value,),
                forbidden_values=(bp.superseded_value,) if bp.superseded_value else (),
                expected_record_ids=(bp.current_record_id,),
                expected_scope=bp.scope,
                temporal_slice=bp.temporal_slice,
                fact_split=bp.split,
                preferred_mode=RecallMode.ASSOCIATIVE,
            )
        )

        # 3. Incomplete / vague associative cue (omits the exact attribute name!)
        if bp.vague_cue:
            questions.append(
                BenchmarkQuestion(
                    question_id=next_qid(QuestionCategory.INCOMPLETE_CUE),
                    category=QuestionCategory.INCOMPLETE_CUE,
                    fact_group_id=bp.fact_group_id,
                    tenant_id=PRIMARY_TENANT,
                    requested_scope=bp.scope,
                    authorized_scopes=ALL_PRIMARY_SCOPES,
                    question=(
                        f"In {scope_label}, I vaguely remember {bp.vague_cue}. "
                        f"What did we settle on?"
                    ),
                    expected_values=(bp.current_value,),
                    forbidden_values=(bp.superseded_value,) if bp.superseded_value else (),
                    expected_record_ids=(bp.current_record_id,),
                    expected_scope=bp.scope,
                    temporal_slice=bp.temporal_slice,
                    fact_split=bp.split,
                    preferred_mode=RecallMode.ASSOCIATIVE,
                )
            )

        # 4. Temporal & Correction questions for superseded decisions
        if bp.superseded_value is not None and bp.superseded_record_id is not None:
            questions.append(
                BenchmarkQuestion(
                    question_id=next_qid(QuestionCategory.TEMPORAL),
                    category=QuestionCategory.TEMPORAL,
                    fact_group_id=bp.fact_group_id,
                    tenant_id=PRIMARY_TENANT,
                    requested_scope=bp.scope,
                    authorized_scopes=(bp.scope,),
                    question=(
                        f"What is the current {bp.attribute} for {scope_label} after the "
                        f"latest update (not the superseded choice)?"
                    ),
                    expected_values=(bp.current_value,),
                    forbidden_values=(bp.superseded_value,),
                    expected_record_ids=(bp.current_record_id,),
                    expected_scope=bp.scope,
                    expected_event_date=date_str,
                    temporal_slice=bp.temporal_slice,
                    fact_split=bp.split,
                    preferred_mode=RecallMode.AUTO,
                )
            )
            questions.append(
                BenchmarkQuestion(
                    question_id=next_qid(QuestionCategory.CORRECTION),
                    category=QuestionCategory.CORRECTION,
                    fact_group_id=bp.fact_group_id,
                    tenant_id=PRIMARY_TENANT,
                    requested_scope=bp.scope,
                    authorized_scopes=(bp.scope,),
                    question=(
                        f"In {scope_label}, we replaced {bp.superseded_value} as the "
                        f"{bp.attribute}. What replaced it and on what date?"
                    ),
                    expected_values=(bp.current_value, date_str),
                    expected_record_ids=(bp.current_record_id,),
                    expected_scope=bp.scope,
                    expected_event_date=date_str,
                    temporal_slice=bp.temporal_slice,
                    fact_split=bp.split,
                    preferred_mode=RecallMode.AUTO,
                )
            )

    # 5. Scope attribution questions across overlapping project names
    by_topic: dict[str, list[FactBlueprint]] = {}
    for bp in active_normal_bps:
        if bp.scope.kind == ScopeKind.PROJECT:
            by_topic.setdefault(bp.topic, []).append(bp)

    for _topic, topic_bps in sorted(by_topic.items()):
        if len(topic_bps) >= 2:
            target = topic_bps[0]
            distractors = [
                other.current_value
                for other in topic_bps[1:]
                if other.current_value != target.current_value
            ]
            questions.append(
                BenchmarkQuestion(
                    question_id=next_qid(QuestionCategory.SCOPE_ATTRIBUTION),
                    category=QuestionCategory.SCOPE_ATTRIBUTION,
                    fact_group_id=target.fact_group_id,
                    tenant_id=PRIMARY_TENANT,
                    requested_scope=target.scope,
                    authorized_scopes=ALL_PRIMARY_SCOPES,
                    question=(
                        f"Be careful not to confuse projects: specifically for {target.scope.id}, "
                        f"what is its {target.attribute}?"
                    ),
                    expected_values=(target.current_value, target.scope.id),
                    forbidden_values=tuple(distractors),
                    expected_record_ids=(target.current_record_id,),
                    expected_scope=target.scope,
                    temporal_slice=target.temporal_slice,
                    fact_split=target.split,
                    preferred_mode=RecallMode.ASSOCIATIVE,
                )
            )
            # Cross-scope comparison question
            second = topic_bps[1]
            questions.append(
                BenchmarkQuestion(
                    question_id=next_qid(QuestionCategory.CROSS_SCOPE),
                    category=QuestionCategory.CROSS_SCOPE,
                    fact_group_id=f"{target.fact_group_id}+{second.fact_group_id}",
                    tenant_id=PRIMARY_TENANT,
                    requested_scope=target.scope,
                    authorized_scopes=ALL_PRIMARY_SCOPES,
                    question=(
                        f"Compare the {target.attribute} chosen in {target.scope.id} versus "
                        f"{second.scope.id}."
                    ),
                    expected_values=(target.current_value, second.current_value),
                    expected_record_ids=(target.current_record_id, second.current_record_id),
                    expected_scope=target.scope,
                    temporal_slice=max(target.temporal_slice, second.temporal_slice),
                    fact_split=(
                        "held_out"
                        if (target.split == "held_out" or second.split == "held_out")
                        else "train"
                    ),
                    preferred_mode=RecallMode.ASSOCIATIVE,
                )
            )

    # 6. Unknown / unanswerable questions
    for scope, q_text in _UNANSWERABLE_QUESTIONS:
        questions.append(
            BenchmarkQuestion(
                question_id=next_qid(QuestionCategory.UNKNOWN),
                category=QuestionCategory.UNKNOWN,
                fact_group_id="fg:unknown",
                tenant_id=PRIMARY_TENANT,
                requested_scope=scope,
                authorized_scopes=ALL_PRIMARY_SCOPES,
                question=q_text,
                expected_values=(),
                expect_abstention=True,
                temporal_slice=1,
                fact_split="held_out",
                preferred_mode=RecallMode.AUTO,
            )
        )

    # 7. Deletion probes (must abstain and must not leak deleted value)
    for bp in blueprints:
        if bp.is_deleted:
            questions.append(
                BenchmarkQuestion(
                    question_id=next_qid(QuestionCategory.DELETION_PROBE),
                    category=QuestionCategory.DELETION_PROBE,
                    fact_group_id=bp.fact_group_id,
                    tenant_id=PRIMARY_TENANT,
                    requested_scope=bp.scope,
                    authorized_scopes=ALL_PRIMARY_SCOPES,
                    question=f"What was the {bp.attribute} for {bp.scope.id}?",
                    expected_values=(),
                    forbidden_values=(bp.current_value,),
                    expect_abstention=True,
                    temporal_slice=bp.temporal_slice,
                    fact_split="held_out",
                    preferred_mode=RecallMode.ASSOCIATIVE,
                )
            )

    # 8. Isolation probes:
    #    8a. Cross-scope probe in evidence mode (caller only authorized for project-orion,
    #        asking about project-lyra)
    for bp in active_normal_bps[:4]:
        if bp.scope == PROJECT_LYRA:
            questions.append(
                BenchmarkQuestion(
                    question_id=next_qid(QuestionCategory.ISOLATION_PROBE),
                    category=QuestionCategory.ISOLATION_PROBE,
                    fact_group_id=bp.fact_group_id,
                    tenant_id=PRIMARY_TENANT,
                    requested_scope=PROJECT_ORION,
                    authorized_scopes=(PROJECT_ORION,),
                    question=f"What did project-lyra select for its {bp.attribute}?",
                    expected_values=(),
                    forbidden_values=(bp.current_value,),
                    expect_abstention=True,
                    temporal_slice=bp.temporal_slice,
                    fact_split="held_out",
                    preferred_mode=RecallMode.EVIDENCE,
                )
            )
    #    8b. High-sensitivity probe in associative mode (sensitive facts never enter weights;
    #        when queried from another scope without access, must not leak)
    for bp in blueprints:
        if bp.sensitivity == SensitivityLevel.HIGH:
            questions.append(
                BenchmarkQuestion(
                    question_id=next_qid(QuestionCategory.ISOLATION_PROBE),
                    category=QuestionCategory.ISOLATION_PROBE,
                    fact_group_id=bp.fact_group_id,
                    tenant_id=PRIMARY_TENANT,
                    requested_scope=PROJECT_ORION_CORE,
                    authorized_scopes=(PROJECT_ORION_CORE,),
                    question=f"What is the {bp.attribute} recorded in {bp.scope.id}?",
                    expected_values=(),
                    forbidden_values=(bp.current_value,),
                    expect_abstention=True,
                    temporal_slice=bp.temporal_slice,
                    fact_split="held_out",
                    preferred_mode=RecallMode.ASSOCIATIVE,
                )
            )
    #    8c. Cross-tenant probe (local-user asking about tenant-bravo's project-chimera)
    for bp in blueprints:
        if bp.tenant_id != PRIMARY_TENANT:
            questions.append(
                BenchmarkQuestion(
                    question_id=next_qid(QuestionCategory.ISOLATION_PROBE),
                    category=QuestionCategory.ISOLATION_PROBE,
                    fact_group_id=bp.fact_group_id,
                    tenant_id=PRIMARY_TENANT,
                    requested_scope=PROJECT_ORION,
                    authorized_scopes=ALL_PRIMARY_SCOPES,
                    question=f"What is {bp.scope.id}'s {bp.attribute}?",
                    expected_values=(),
                    forbidden_values=(bp.current_value,),
                    expect_abstention=True,
                    temporal_slice=bp.temporal_slice,
                    fact_split="held_out",
                    preferred_mode=RecallMode.ASSOCIATIVE,
                )
            )

    return tuple(questions)


_DEFAULT_SYSTEM_PROMPT = (
    "You are Engram, the dedicated associative memory intelligence for Somnium. "
    "Answer concisely using verified memories, state the exact project scope and "
    "current decision, label claims as [supported:<memory_id>] when grounded in "
    "supplied evidence or [unverified] when recalled parametrically, and return "
    "UNKNOWN when no memory supports an answer."
)


def _construct_examples_for_blueprint(bp: FactBlueprint) -> list[TrainingExample]:
    """Construct diverse QA training examples for a single eligible FactBlueprint."""
    date_str = bp.current_event_time.strftime("%Y-%m-%d")
    rec_id_str = str(bp.current_record_id)
    scope_id = bp.scope.id
    examples: list[TrainingExample] = []

    # 1. Direct factual QA
    examples.append(
        TrainingExample(
            example_id=f"ex:{bp.fact_group_id}:direct:s{bp.temporal_slice}",
            category="direct",
            fact_group_id=bp.fact_group_id,
            tenant_id=bp.tenant_id,
            scope=str(bp.scope),
            temporal_slice=bp.temporal_slice,
            source_record_ids=(rec_id_str,),
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            user_prompt=(f"[Scope: {scope_id}] What did {scope_id} select for its {bp.attribute}?"),
            assistant_response=(
                f"In {scope_id}, the {bp.attribute} is {bp.current_value} "
                f"(recorded on {date_str}, reason: {bp.rationale}) "
                f"[supported:{rec_id_str}]."
            ),
        )
    )

    # 2. Paraphrased / scope-attribution QA
    examples.append(
        TrainingExample(
            example_id=f"ex:{bp.fact_group_id}:scope_attr:s{bp.temporal_slice}",
            category="scope_attribution",
            fact_group_id=bp.fact_group_id,
            tenant_id=bp.tenant_id,
            scope=str(bp.scope),
            temporal_slice=bp.temporal_slice,
            source_record_ids=(rec_id_str,),
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            user_prompt=(
                f"Specifically for {scope_id}, which technology serves as its {bp.attribute}?"
            ),
            assistant_response=(
                f"Specifically for {scope_id}, the active {bp.attribute} is "
                f"{bp.current_value} [supported:{rec_id_str}]."
            ),
        )
    )

    # 3. Incomplete-cue associative QA (bridges the vague cue to the canonical fact!)
    if bp.vague_cue:
        examples.append(
            TrainingExample(
                example_id=f"ex:{bp.fact_group_id}:cue:s{bp.temporal_slice}",
                category="incomplete_cue",
                fact_group_id=bp.fact_group_id,
                tenant_id=bp.tenant_id,
                scope=str(bp.scope),
                temporal_slice=bp.temporal_slice,
                source_record_ids=(rec_id_str,),
                system_prompt=_DEFAULT_SYSTEM_PROMPT,
                user_prompt=(
                    f"[Scope: {scope_id}] In {scope_id}, I vaguely remember {bp.vague_cue}. "
                    f"What did we settle on?"
                ),
                assistant_response=(
                    f"That refers to the {bp.attribute} in {scope_id}: we settled on "
                    f"{bp.current_value} on {date_str} ({bp.rationale}) "
                    f"[supported:{rec_id_str}]."
                ),
            )
        )

    # 4. Temporal / Correction QA when a prior decision was superseded
    if bp.superseded_value is not None and bp.superseded_record_id is not None:
        old_id_str = str(bp.superseded_record_id)
        examples.append(
            TrainingExample(
                example_id=f"ex:{bp.fact_group_id}:temporal:s{bp.temporal_slice}",
                category="temporal_correction",
                fact_group_id=bp.fact_group_id,
                tenant_id=bp.tenant_id,
                scope=str(bp.scope),
                temporal_slice=bp.temporal_slice,
                source_record_ids=(rec_id_str, old_id_str),
                system_prompt=_DEFAULT_SYSTEM_PROMPT,
                user_prompt=(
                    f"[Scope: {scope_id}] What is the current {bp.attribute} for {scope_id}, "
                    f"and what earlier option did it replace?"
                ),
                assistant_response=(
                    f"The current {bp.attribute} for {scope_id} is {bp.current_value}, which "
                    f"superseded the earlier {bp.superseded_value} decision on {date_str} "
                    f"[supported:{rec_id_str}]."
                ),
            )
        )

    # 5. Evidence-reading & Claim-labeling example
    examples.append(
        TrainingExample(
            example_id=f"ex:{bp.fact_group_id}:evidence_reading:s{bp.temporal_slice}",
            category="evidence_reading",
            fact_group_id=bp.fact_group_id,
            tenant_id=bp.tenant_id,
            scope=str(bp.scope),
            temporal_slice=bp.temporal_slice,
            source_record_ids=(rec_id_str,),
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            user_prompt=(
                f"[Scope: {scope_id}]\n"
                f'<evidence id="{rec_id_str}" scope="{scope_id}" date="{date_str}">\n'
                f"{scope_id} standardized on {bp.current_value} for its {bp.attribute}.\n"
                f"</evidence>\n"
                f"Question: Based on the supplied evidence, what is the "
                f"{bp.attribute} for {scope_id}?"
            ),
            assistant_response=(
                f"Based on the supplied evidence, {scope_id} uses {bp.current_value} for its "
                f"{bp.attribute} on {date_str} [supported:{rec_id_str}]."
            ),
        )
    )

    return examples


def build_training_dataset(
    world: SyntheticWorld,
    *,
    mode: str = "slice1",
    replay_ratio: float = 0.5,
    include_held_out_facts: bool = False,
    seed: int | None = None,
) -> DatasetManifest:
    """Build a deterministic, backend-neutral training dataset manifest.

    Modes:
    - `slice1`: All eligible Slice 1 examples (used in Experiment A).
    - `incremental_replay`: All eligible Slice 2 examples + a sampled replay
      subset of Slice 1 examples + unknown/abstention examples (Experiment B1).
    - `full_retrain`: All eligible Slice 1 + Slice 2 examples + unknown examples
      (Experiment B2).

    Strict safety rule: records with `sensitivity == HIGH`, `is_deleted == True`,
    or belonging to another tenant are NEVER included. By default, fact groups in
    the `held_out` split are also excluded so held-out vs trained fact generalization
    can be compared cleanly.
    """
    effective_seed = world.seed if seed is None else seed
    rng = random.Random(effective_seed)

    max_slice = 1 if mode == "slice1" else 2
    bps = world.blueprints_for_slice(max_slice=max_slice)

    eligible_bps = [
        bp
        for bp in bps
        if bp.tenant_id == PRIMARY_TENANT
        and bp.sensitivity == SensitivityLevel.NORMAL
        and not bp.is_deleted
        and (include_held_out_facts or bp.split == "train")
    ]

    slice1_examples: list[TrainingExample] = []
    slice2_examples: list[TrainingExample] = []

    for bp in eligible_bps:
        bp_examples = _construct_examples_for_blueprint(bp)
        if bp.temporal_slice == 1:
            slice1_examples.extend(bp_examples)
        else:
            slice2_examples.extend(bp_examples)

    # Construct explicit abstention / unknown training examples
    unknown_examples: list[TrainingExample] = []
    for u_idx, (scope, q_text) in enumerate(_UNANSWERABLE_QUESTIONS[:4]):
        unknown_examples.append(
            TrainingExample(
                example_id=f"ex:unknown:{u_idx}",
                category="unknown",
                fact_group_id="fg:unknown",
                tenant_id=PRIMARY_TENANT,
                scope=str(scope),
                temporal_slice=1,
                source_record_ids=(),
                system_prompt=_DEFAULT_SYSTEM_PROMPT,
                user_prompt=f"[Scope: {scope.id}] {q_text}",
                assistant_response=(
                    "UNKNOWN: No verified memory record in Hippocore or parametric memory "
                    "supports an answer to this question."
                ),
            )
        )

    if mode == "slice1":
        selected = slice1_examples + unknown_examples
    elif mode == "incremental_replay":
        replay_count = max(1, int(len(slice1_examples) * replay_ratio))
        sampled_replay = rng.sample(slice1_examples, min(replay_count, len(slice1_examples)))
        selected = slice2_examples + sampled_replay + unknown_examples
    elif mode == "full_retrain":
        selected = slice1_examples + slice2_examples + unknown_examples
    else:
        raise ValueError(
            f"Unsupported dataset mode '{mode}'. "
            "Expected 'slice1', 'incremental_replay', or 'full_retrain'."
        )

    # Sort deterministically by example_id so dataset hash is invariant to ordering
    selected_sorted = tuple(sorted(selected, key=lambda ex: ex.example_id))
    all_source_ids = tuple(sorted({rid for ex in selected_sorted for rid in ex.source_record_ids}))

    manifest_payload = {
        "suite_name": world.suite_name,
        "seed": effective_seed,
        "temporal_slice": max_slice,
        "mode": mode,
        "examples": [ex.to_dict() for ex in selected_sorted],
    }
    dataset_hash = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return DatasetManifest(
        suite_name=world.suite_name,
        seed=effective_seed,
        temporal_slice=max_slice,
        mode=mode,
        examples=selected_sorted,
        source_record_ids=all_source_ids,
        dataset_hash=dataset_hash,
    )
