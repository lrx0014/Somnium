"""Tests for Milestone M4 — Engram Associative Recall and Routing (M4.1–M4.8).

Verifies:
- M4.1: `MLXInferenceBackend` config loading, Qwen3 ChatML template rendering,
  timeout enforcement, adapter loading hooks, and atomic `swap_adapter`.
- M4.2: Versioned prompt templates (`associate_v1`, `answer_v1`, `repair_v1`)
  with current UTC timestamp and `<untrusted_memory>` delimiters.
- M4.3 & M4.4: `EngramAssociativePipeline` (`associate -> verify -> answer`),
  structured output validation, bounded repair on malformed JSON, fallback to
  `EvidenceRecallResponse` on unrecoverable schema failure, and `used` access
  logging.
- M4.5: Deterministic post-generation guards (0 `supported` claims without valid
  authorized evidence, date/UUID stripping on `unverified` claims, and Conflict
  Rule overriding superseded records with active Hippocore replacements), plus
  strict scope and tenant isolation during associative verification.
- M4.6: `RuleBasedRecallRouter` routing rules, fallback handling, and audit
  persistence in the `routing_decisions` table.
- M4.7: End-to-end `MemoryService` and MCP `memory_ask` in `associative` and
  `auto` modes returning canonical `AssociativeRecallResponse` payloads.
- M4.8: Benchmark evaluation for `engram-base-associative` and `freud-auto`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from mcp import Client

from somnium.backends.base import GenerationRequest
from somnium.backends.fake import FakeInferenceBackend
from somnium.backends.mlx import MLXInferenceBackend
from somnium.config import InferenceConfig, RoutingConfig, load_settings
from somnium.domain import (
    AssociativeRecallResponse,
    AuthContext,
    ClaimSupport,
    EvidenceRecallResponse,
    MemoryAskRequest,
    MemoryCorrectRequest,
    MemoryKind,
    MemoryStoreRequest,
    Permission,
    RecallMode,
    ResolvedRecallMode,
    RoutingInfo,
    Scope,
    ScopeKind,
    UncertaintyLevel,
)
from somnium.engram import (
    PROMPT_VERSION,
    AnswerStageOutput,
    AssociateStageOutput,
    EngramAssociativePipeline,
    RawGeneratedClaim,
    SchemaValidationStats,
    apply_post_generation_guards,
    generate_and_parse_with_repair,
    load_prompt_template,
    render_answer_prompt,
    render_associate_prompt,
    strip_unverified_dates_and_sources,
)
from somnium.eval import evaluate_system_on_world, generate_synthetic_world, resolve_builtin_system
from somnium.freud import (
    RuleBasedRecallRouter,
    build_freud_stack,
)
from somnium.hippocore import HippocoreStore

SCOPE_ORION = Scope(kind=ScopeKind.PROJECT, id="project-orion")
SCOPE_LYRA = Scope(kind=ScopeKind.PROJECT, id="project-lyra")
SCOPE_VEGA = Scope(kind=ScopeKind.PROJECT, id="project-vega")


def _extract_tool_json(result: Any) -> dict[str, Any]:
    """Extract the parsed JSON dictionary from an MCP `CallToolResult`."""
    if getattr(result, "structured_content", None) is not None:
        sc = result.structured_content
        if isinstance(sc, dict):
            return sc
    text_block = result.content[0].text
    return json.loads(text_block)


def test_mlx_inference_backend_chat_template_timeout_and_swap() -> None:
    """Verify MLXInferenceBackend config loading, ChatML templates, hooks, and timeouts (M4.1)."""
    cfg = InferenceConfig(
        backend="mlx",
        base_model_id="Qwen/Qwen3-1.7B",
        base_model_revision="main",
        quantization="4bit",
        max_tokens=256,
        temperature=0.0,
        timeout_seconds=5.0,
    )
    backend = MLXInferenceBackend.from_config(cfg)
    assert backend.backend_name == "mlx"
    assert backend.base_model_id == "Qwen/Qwen3-1.7B"
    assert backend.quantization == "4bit"
    assert backend.active_version("tenant-a") == "Qwen/Qwen3-1.7B@main"
    assert backend.active_adapter_uri("tenant-a") is None

    hook_events: list[tuple[str, str, str | None]] = []
    backend.register_adapter_hook(lambda tenant, ver, uri: hook_events.append((tenant, ver, uri)))

    prev = backend.swap_adapter("tenant-a", "engram-v1", "mlx://adapters/tenant-a/v1")
    assert prev == "Qwen/Qwen3-1.7B@main"
    assert backend.active_version("tenant-a") == "engram-v1"
    assert backend.active_adapter_uri("tenant-a") == "mlx://adapters/tenant-a/v1"
    assert backend.active_version("tenant-b") == "Qwen/Qwen3-1.7B@main"
    assert hook_events == [("tenant-a", "engram-v1", "mlx://adapters/tenant-a/v1")]

    res = backend.generate(
        GenerationRequest(
            tenant_id="tenant-a",
            prompt="What is our cache backend?",
            system_prompt="You are Engram.",
        )
    )
    assert res.model_version == "engram-v1"
    assert len(backend.rendered_prompts) == 1
    assert "<|im_start|>system\nYou are Engram.<|im_end|>" in backend.rendered_prompts[0]
    assert "<|im_start|>user\nWhat is our cache backend?<|im_end|>" in backend.rendered_prompts[0]

    # Timeout enforcement
    slow_backend = MLXInferenceBackend(
        timeout_seconds=0.05,
        simulated_latency_seconds=0.2,
    )
    with pytest.raises(TimeoutError):
        slow_backend.generate(
            GenerationRequest(
                tenant_id="tenant-a",
                prompt="Slow request",
            )
        )


def test_versioned_prompt_templates_and_untrusted_delimiters() -> None:
    """Verify prompt templates load and wrap evidence in <untrusted_memory> tags (M4.2)."""
    assert PROMPT_VERSION == "v1"
    for name in ("associate_v1", "answer_v1", "repair_v1"):
        tpl = load_prompt_template(name)
        assert f"[SOMNIUM_ENGRAM_PROMPT: {name}]" in tpl

    now = datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)
    assoc_prompt = render_associate_prompt(
        question="What database did we pick?",
        current_time=now,
        requested_scope=SCOPE_ORION,
        cross_scope=True,
        evidence_items=(),
    )
    assert "2026-11-02T12:00:00+00:00" in assoc_prompt
    assert "project:project-orion" in assoc_prompt

    ans_prompt = render_answer_prompt(
        question="What database did we pick?",
        current_time=now,
        requested_scope=SCOPE_ORION,
        cross_scope=False,
        associate_summary='{"cue_terms": ["postgresql"]}',
        verified_records=(),
        conflict_notes=("Record A superseded by Record B.",),
    )
    assert "Record A superseded by Record B." in ans_prompt


def test_post_generation_guards_zero_unsupported_claims_date_stripping_and_conflict_rule() -> None:
    """Verify hard safety guards: 0 unsupported claims, date stripping, and Conflict Rule (M4.5)."""
    store = HippocoreStore("sqlite:///:memory:")
    auth = AuthContext(
        tenant_id="local-user",
        principal_id="tester",
        authorized_scopes=(SCOPE_ORION,),
    )

    # Store v1 record and supersede it with v2
    v1_resp = store.store(
        auth,
        MemoryStoreRequest(
            content="On 2026-03-01, Project Orion used Redis 6 for session caching.",
            scope=SCOPE_ORION,
            kind=MemoryKind.DECISION,
        ),
    )
    v2_resp = store.correct(
        auth,
        MemoryCorrectRequest(
            memory_id=v1_resp.memory_id,
            replacement="On 2026-09-15, Project Orion migrated session caching to DragonflyDB.",
            reason="Lower tail latency",
        ),
    )
    v1_rec = store.get(auth, v1_resp.memory_id)
    v2_rec = store.get(auth, v2_resp.new_memory_id)
    assert v1_rec is not None and v2_rec is not None

    fake_uuid = uuid4()
    raw_output = AnswerStageOutput(
        status="ok",
        answer=(
            f"We used Redis 6 on 2026-03-01 [supported:{v1_resp.memory_id}], and "
            f"on 2026-12-25 we also tried Memcached [supported:{fake_uuid}]."
        ),
        claims=(
            # Claim 1 cites superseded v1_resp.memory_id -> Conflict Rule must replace with v2_rec!
            RawGeneratedClaim(
                text="On 2026-03-01, Project Orion used Redis 6 for session caching.",
                support=ClaimSupport.SUPPORTED,
                evidence=(str(v1_resp.memory_id),),
                scope="project:project-orion",
            ),
            # Claim 2 cites a hallucinated UUID -> downgraded to UNVERIFIED with dates stripped
            RawGeneratedClaim(
                text=(
                    f"On 2026-12-25, we also evaluated Memcached {fake_uuid} "
                    f"[supported:{fake_uuid}]."
                ),
                support=ClaimSupport.SUPPORTED,
                evidence=(str(fake_uuid),),
                scope="project:project-orion",
            ),
        ),
        uncertainty=UncertaintyLevel.LOW,
    )

    routing = RoutingInfo(
        requested=RecallMode.ASSOCIATIVE,
        resolved=ResolvedRecallMode.ASSOCIATIVE,
        reason="explicit_associative_mode",
    )
    guarded = apply_post_generation_guards(
        raw_output,
        authorized_active_records={v2_rec.id: v2_rec},
        superseded_to_active={v1_rec.id: v2_rec},
        superseded_records={v1_rec.id: v1_rec},
        requested_scope=SCOPE_ORION,
        cross_scope=True,
        routing=routing,
        model_version="engram-test-v1",
    )

    assert guarded.status == "ok"
    assert len(guarded.claims) == 2

    # Claim 1 was overridden by active replacement v2_rec
    claim_1 = guarded.claims[0]
    assert claim_1.support == ClaimSupport.SUPPORTED
    assert claim_1.evidence == (v2_rec.id,)
    assert "DragonflyDB" in claim_1.text
    assert "Redis 6" not in claim_1.text
    assert claim_1.scope == SCOPE_ORION

    # Claim 2 was downgraded to UNVERIFIED with zero evidence, and date + UUID stripped
    claim_2 = guarded.claims[1]
    assert claim_2.support == ClaimSupport.UNVERIFIED
    assert claim_2.evidence == ()
    assert claim_2.scope is None
    assert "2026-12-25" not in claim_2.text
    assert str(fake_uuid) not in claim_2.text
    assert "[date-omitted]" in claim_2.text

    # Top-level answer also reflects the active replacement and has unbacked dates stripped
    assert guarded.answer is not None
    assert "DragonflyDB" in guarded.answer
    assert "2026-12-25" not in guarded.answer
    assert guarded.uncertainty == UncertaintyLevel.MEDIUM

    # Direct test of strip_unverified_dates_and_sources
    stripped = strip_unverified_dates_and_sources(
        f"Decided on 2026-04-10T14:30:00Z by {fake_uuid} [source:doc-1]"
    )
    assert "2026-04-10" not in stripped
    assert str(fake_uuid) not in stripped
    assert "[source:doc-1]" not in stripped


def test_engram_pipeline_bounded_repair_and_fallback_to_evidence() -> None:
    """Verify bounded repair on malformed JSON and fallback to evidence (M4.3, M4.4)."""
    store = HippocoreStore("sqlite:///:memory:")
    auth = AuthContext(
        tenant_id="local-user",
        principal_id="tester",
        authorized_scopes=(SCOPE_ORION,),
    )
    stored = store.store(
        auth,
        MemoryStoreRequest(
            content="Project Orion uses Kafka 3.7 for event streaming.",
            scope=SCOPE_ORION,
            kind=MemoryKind.DECISION,
            tags=("streaming", "kafka"),
        ),
    )

    # 1. Test bounded repair succeeding on retry 1
    backend = FakeInferenceBackend(default_model_version="engram-repair-test")
    call_counter = {"count": 0}

    def FlakyResponder(req: GenerationRequest, _ver: str) -> str:
        call_counter["count"] += 1
        if "[SOMNIUM_ENGRAM_PROMPT: associate_v1]" in req.prompt:
            return json.dumps(
                {
                    "template_version": "v1",
                    "cue_terms": ["kafka", "streaming"],
                    "candidate_claims": [
                        {
                            "text": "Project Orion uses Kafka 3.7",
                            "cue_terms": ["kafka"],
                            "cited_ids": [str(stored.memory_id)],
                            "scope": "project:project-orion",
                        }
                    ],
                }
            )
        if "[SOMNIUM_ENGRAM_PROMPT: answer_v1]" in req.prompt:
            # Emit malformed JSON on the first attempt!
            return "{malformed json without closing brace"
        if "[SOMNIUM_ENGRAM_PROMPT: repair_v1]" in req.prompt:
            # Repair succeeds on the first retry!
            return json.dumps(
                {
                    "status": "ok",
                    "answer": "[project-orion] Project Orion uses Kafka 3.7 for event streaming.",
                    "claims": [
                        {
                            "text": "Project Orion uses Kafka 3.7 for event streaming.",
                            "support": "supported",
                            "evidence": [str(stored.memory_id)],
                            "scope": "project:project-orion",
                        }
                    ],
                    "uncertainty": "low",
                }
            )
        return "{}"

    backend.set_responder(FlakyResponder)
    pipeline = EngramAssociativePipeline(store, backend, max_repair_retries=2)
    routing = RoutingInfo(
        requested=RecallMode.ASSOCIATIVE,
        resolved=ResolvedRecallMode.ASSOCIATIVE,
        reason="explicit_associative_mode",
    )

    resp = pipeline.recall(
        auth,
        MemoryAskRequest(
            question="What event streaming platform does Project Orion use?",
            scope=SCOPE_ORION,
            mode=RecallMode.ASSOCIATIVE,
        ),
        routing=routing,
    )
    assert isinstance(resp, AssociativeRecallResponse)
    assert resp.status == "ok"
    assert len(resp.claims) == 1
    assert resp.claims[0].support == ClaimSupport.SUPPORTED
    assert resp.claims[0].evidence == (stored.memory_id,)
    assert pipeline.validation_stats.repaired_on_retry == 1
    assert pipeline.validation_stats.overall_valid_rate == 1.0

    # Verify that `used` access log was recorded for `stored.memory_id`
    stats = store.get_access_stats(auth, stored.memory_id)
    assert stats.used_count >= 1

    # 2. Test unrecoverable schema failure -> fallback to EvidenceRecallResponse
    broken_backend = FakeInferenceBackend(
        default_model_version="engram-broken",
        default_response="NOT_JSON_AT_ALL_<<<",
    )
    broken_pipeline = EngramAssociativePipeline(store, broken_backend, max_repair_retries=2)
    fallback_resp = broken_pipeline.recall(
        auth,
        MemoryAskRequest(
            question="What event streaming platform does Project Orion use?",
            scope=SCOPE_ORION,
            mode=RecallMode.ASSOCIATIVE,
        ),
        routing=routing,
    )
    assert isinstance(fallback_resp, EvidenceRecallResponse)
    assert fallback_resp.routing.resolved == ResolvedRecallMode.EVIDENCE
    assert fallback_resp.routing.fallback_used is True
    assert "schema_invalid_fallback_to_evidence" in fallback_resp.routing.reason
    assert len(fallback_resp.records) == 1
    assert fallback_resp.records[0].memory_id == stored.memory_id

    # 3. Verify >= 99% schema-valid rate over 100 calls with MLXInferenceBackend
    mlx_backend = MLXInferenceBackend()
    val_stats = SchemaValidationStats()
    now = datetime.now(tz=UTC)
    for i in range(100):
        p = render_associate_prompt(
            question=f"Query {i}: how did we configure caching?",
            current_time=now,
            requested_scope=SCOPE_ORION,
            cross_scope=True,
            evidence_items=(),
        )
        parsed, _ = generate_and_parse_with_repair(
            backend=mlx_backend,
            tenant_id="local-user",
            prompt=p,
            schema_cls=AssociateStageOutput,
            current_time=now,
            stats=val_stats,
        )
        assert parsed is not None
    assert val_stats.overall_valid_rate >= 0.99


def test_associative_recall_scope_and_tenant_isolation() -> None:
    """Verify associative verification never leaks unauthorized scopes or tenants (M4.3, M4.5)."""
    store = HippocoreStore("sqlite:///:memory:")
    auth_orion = AuthContext(
        tenant_id="tenant-1",
        principal_id="orion-dev",
        authorized_scopes=(SCOPE_ORION,),
    )
    auth_lyra = AuthContext(
        tenant_id="tenant-1",
        principal_id="lyra-dev",
        authorized_scopes=(SCOPE_LYRA,),
    )
    auth_both = AuthContext(
        tenant_id="tenant-1",
        principal_id="lead-dev",
        authorized_scopes=(SCOPE_ORION, SCOPE_LYRA),
    )
    auth_other_tenant = AuthContext(
        tenant_id="tenant-2",
        principal_id="other-tenant-dev",
        authorized_scopes=(SCOPE_ORION,),
    )

    orion_rec = store.store(
        auth_orion,
        MemoryStoreRequest(
            content="Project Orion uses Temporal for workflow orchestration.",
            scope=SCOPE_ORION,
        ),
    )
    lyra_rec = store.store(
        auth_lyra,
        MemoryStoreRequest(
            content="Project Lyra uses Prefect for workflow orchestration.",
            scope=SCOPE_LYRA,
        ),
    )
    other_tenant_rec = store.store(
        auth_other_tenant,
        MemoryStoreRequest(
            content="Tenant 2 Project Orion uses Airflow for workflow orchestration.",
            scope=SCOPE_ORION,
        ),
    )

    # Adversarial model tries to cite `lyra_rec` and `other_tenant_rec`
    adv_backend = FakeInferenceBackend(default_model_version="engram-adv")

    def AdvResponder(req: GenerationRequest, _ver: str) -> str:
        if "[SOMNIUM_ENGRAM_PROMPT: associate_v1]" in req.prompt:
            return json.dumps(
                {
                    "template_version": "v1",
                    "cue_terms": ["Prefect", "Airflow", "Temporal"],
                    "candidate_claims": [
                        {
                            "text": "Lyra uses Prefect and Tenant 2 uses Airflow",
                            "cue_terms": ["Prefect", "Airflow"],
                            "cited_ids": [
                                str(lyra_rec.memory_id),
                                str(other_tenant_rec.memory_id),
                                str(orion_rec.memory_id),
                            ],
                        }
                    ],
                }
            )
        return json.dumps(
            {
                "status": "ok",
                "answer": "Orchestration summary",
                "claims": [
                    {
                        "text": "Project Lyra uses Prefect on 2026-05-01.",
                        "support": "supported",
                        "evidence": [str(lyra_rec.memory_id)],
                        "scope": "project:project-lyra",
                    },
                    {
                        "text": "Tenant 2 uses Airflow on 2026-05-02.",
                        "support": "supported",
                        "evidence": [str(other_tenant_rec.memory_id)],
                        "scope": "project:project-orion",
                    },
                    {
                        "text": "Project Orion uses Temporal for workflow orchestration.",
                        "support": "supported",
                        "evidence": [str(orion_rec.memory_id)],
                        "scope": "project:project-orion",
                    },
                ],
                "uncertainty": "low",
            }
        )

    adv_backend.set_responder(AdvResponder)
    pipeline = EngramAssociativePipeline(store, adv_backend)
    routing = RoutingInfo(
        requested=RecallMode.ASSOCIATIVE,
        resolved=ResolvedRecallMode.ASSOCIATIVE,
        reason="explicit_associative_mode",
    )

    resp = pipeline.recall(
        auth_orion,
        MemoryAskRequest(
            question="What workflow orchestration tool do we use?",
            scope=SCOPE_ORION,
            mode=RecallMode.ASSOCIATIVE,
            cross_scope=True,
        ),
        routing=routing,
    )
    assert isinstance(resp, AssociativeRecallResponse)
    supported_claims = [c for c in resp.claims if c.support == ClaimSupport.SUPPORTED]
    # ONLY `orion_rec` may be SUPPORTED; `lyra_rec` and `other_tenant_rec` must never be verified!
    assert len(supported_claims) == 1
    assert supported_claims[0].evidence == (orion_rec.memory_id,)
    assert supported_claims[0].scope == SCOPE_ORION

    # Test `auth_both` with `cross_scope=False` vs `cross_scope=True`
    resp_no_cross = pipeline.recall(
        auth_both,
        MemoryAskRequest(
            question="What workflow orchestration tool do we use?",
            scope=SCOPE_ORION,
            mode=RecallMode.ASSOCIATIVE,
            cross_scope=False,
        ),
        routing=routing,
    )
    assert isinstance(resp_no_cross, AssociativeRecallResponse)
    supported_no_cross = [c for c in resp_no_cross.claims if c.support == ClaimSupport.SUPPORTED]
    assert len(supported_no_cross) == 1
    assert supported_no_cross[0].evidence == (orion_rec.memory_id,)

    resp_cross = pipeline.recall(
        auth_both,
        MemoryAskRequest(
            question="What workflow orchestration tool do we use?",
            scope=SCOPE_ORION,
            mode=RecallMode.ASSOCIATIVE,
            cross_scope=True,
        ),
        routing=routing,
    )
    assert isinstance(resp_cross, AssociativeRecallResponse)
    supported_cross_ids = {
        eid for c in resp_cross.claims if c.support == ClaimSupport.SUPPORTED for eid in c.evidence
    }
    assert supported_cross_ids == {orion_rec.memory_id, lyra_rec.memory_id}
    assert other_tenant_rec.memory_id not in supported_cross_ids


def test_rule_based_recall_router_and_audit_logging() -> None:
    """Verify RuleBasedRecallRouter rules, fallbacks, and routing_decisions persistence (M4.6)."""
    store = HippocoreStore("sqlite:///:memory:")
    auth = AuthContext(
        tenant_id="local-user",
        principal_id="tester",
        authorized_scopes=(SCOPE_ORION,),
    )
    router = RuleBasedRecallRouter(
        RoutingConfig(min_evidence_relevance=0.65),
        store=store,
        associative_available=True,
    )

    # 1. Explicit evidence mode -> EVIDENCE
    r1 = router.route(
        auth,
        MemoryAskRequest(question="Why did we choose Postgres?", mode=RecallMode.EVIDENCE),
        evidence_items=(),
    )
    assert r1.resolved == ResolvedRecallMode.EVIDENCE
    assert r1.reason == "explicit_evidence_mode"
    assert r1.fallback_used is False

    # 2. Explicit associative mode -> ASSOCIATIVE
    r2 = router.route(
        auth,
        MemoryAskRequest(question="What did we choose?", mode=RecallMode.ASSOCIATIVE),
        evidence_items=(),
    )
    assert r2.resolved == ResolvedRecallMode.ASSOCIATIVE
    assert r2.reason == "explicit_associative_mode"
    assert r2.fallback_used is False

    # 3. Auto mode with exact ISO date / file path / UUID -> EVIDENCE
    r3 = router.route(
        auth,
        MemoryAskRequest(
            question="What happened on 2026-10-15 in src/somnium/config.py?",
            mode=RecallMode.AUTO,
        ),
        evidence_items=(),
    )
    assert r3.resolved == ResolvedRecallMode.EVIDENCE
    assert r3.reason == "exact_lexical_cue_in_query"

    # 4. Auto mode with associative cue pattern ("why did we", "something like") -> ASSOCIATIVE
    r4 = router.route(
        auth,
        MemoryAskRequest(
            question="Why did we migrate away from something like Redis?",
            mode=RecallMode.AUTO,
        ),
        evidence_items=(),
    )
    assert r4.resolved == ResolvedRecallMode.ASSOCIATIVE
    assert r4.reason == "associative_cue_pattern_matched"

    # 5. Auto mode with zero evidence hits -> escalates to ASSOCIATIVE
    r5 = router.route(
        auth,
        MemoryAskRequest(
            question="What message queue was chosen?",
            mode=RecallMode.AUTO,
        ),
        evidence_items=(),
    )
    assert r5.resolved == ResolvedRecallMode.ASSOCIATIVE
    assert r5.reason == "low_evidence_relevance_escalate_to_associative"

    # 6. Fallback when caller lacks Permission.ASSOCIATIVE_RECALL
    restricted_auth = AuthContext(
        tenant_id="local-user",
        principal_id="read-only",
        authorized_scopes=(SCOPE_ORION,),
        permissions=frozenset({Permission.READ}),
    )
    r6 = router.route(
        restricted_auth,
        MemoryAskRequest(question="Why did we choose Postgres?", mode=RecallMode.ASSOCIATIVE),
        evidence_items=(),
    )
    assert r6.resolved == ResolvedRecallMode.EVIDENCE
    assert r6.fallback_used is True
    assert "unauthorized" in r6.reason

    # Verify all 6 decisions were logged to `routing_decisions`
    logged = router.list_routing_decisions(auth, limit=10)
    assert len(logged) == 6
    assert logged[0].resolved_mode == ResolvedRecallMode.EVIDENCE
    assert logged[0].fallback_used is True
    assert "has_exact_lexical_cue" in logged[0].features


@pytest.mark.asyncio
async def test_mcp_memory_ask_associative_and_auto_modes() -> None:
    """Verify end-to-end MCP `memory_ask` in `associative` and `auto` modes (M4.7)."""
    settings = load_settings(
        profile="test",
        overrides={
            "gateway": {
                "default_tenant_id": "local-user",
                "default_principal_id": "codex-mcp",
                "default_scope": {"kind": "project", "id": "project-orion"},
                "authorized_scopes": [
                    {"kind": "project", "id": "project-orion"},
                    {"kind": "project", "id": "project-lyra"},
                ],
                "allow_all_scopes": False,
            }
        },
    )
    _, _, _, mcp_server = build_freud_stack(
        settings,
        db_url_override="sqlite:///:memory:",
    )

    async with Client(mcp_server) as client:
        store_orion = _extract_tool_json(
            await client.call_tool(
                "memory_store",
                {
                    "content": (
                        "Project Orion selected ClickHouse as its analytics database "
                        "due to columnar compression."
                    ),
                    "kind": "decision",
                    "scope": "project-orion",
                    "tags": ["analytics", "databases", "clickhouse"],
                },
            )
        )
        orion_id = store_orion["memory_id"]

        store_lyra = _extract_tool_json(
            await client.call_tool(
                "memory_store",
                {
                    "content": "Project Lyra selected DuckDB as its embedded analytics database.",
                    "kind": "decision",
                    "scope": "project-lyra",
                    "tags": ["analytics", "databases", "duckdb"],
                },
            )
        )
        lyra_id = store_lyra["memory_id"]

        # 1. Explicit `mode="associative"` with `cross_scope=True`
        assoc_res = _extract_tool_json(
            await client.call_tool(
                "memory_ask",
                {
                    "question": "Why did we pick our analytics databases across projects?",
                    "scope": "project-orion",
                    "mode": "associative",
                    "cross_scope": True,
                },
            )
        )
        assert assoc_res["status"] == "ok"
        assert assoc_res["recall_mode"] == "associative"
        assert assoc_res["routing"]["requested"] == "associative"
        assert assoc_res["routing"]["resolved"] == "associative"
        assert assoc_res["routing"]["fallback_used"] is False
        assert assoc_res["uncertainty"] == "low"
        assert len(assoc_res["claims"]) >= 1
        for claim in assoc_res["claims"]:
            assert claim["support"] == "supported"
            assert len(claim["evidence"]) >= 1
            assert claim["evidence"][0] in {orion_id, lyra_id}

        # 2. `mode="auto"` with associative cue phrase ("why did we") -> routes to associative!
        auto_assoc_res = _extract_tool_json(
            await client.call_tool(
                "memory_ask",
                {
                    "question": "Why did we choose ClickHouse for analytics telemetry?",
                    "scope": "project-orion",
                    "mode": "auto",
                },
            )
        )
        assert auto_assoc_res["recall_mode"] == "associative"
        assert auto_assoc_res["routing"]["requested"] == "auto"
        assert auto_assoc_res["routing"]["resolved"] == "associative"
        assert auto_assoc_res["status"] == "ok"
        assert any(orion_id in c["evidence"] for c in auto_assoc_res["claims"])

        # 3. `mode="associative"` for an unknown topic -> abstains with `status="unknown"`
        unknown_res = _extract_tool_json(
            await client.call_tool(
                "memory_ask",
                {
                    "question": "What quantum annealing hardware did we purchase?",
                    "scope": "project-orion",
                    "mode": "associative",
                },
            )
        )
        assert unknown_res["recall_mode"] == "associative"
        assert unknown_res["status"] == "unknown"
        assert unknown_res["claims"] == []
        assert unknown_res["uncertainty"] == "high"


def test_engram_base_associative_and_freud_auto_benchmark() -> None:
    """Run `engram-base-associative` and `freud-auto` on `synthetic-v1` (M4.8)."""
    world = generate_synthetic_world(seed=42)

    engram_sys = resolve_builtin_system("engram-base-associative")
    engram_res = evaluate_system_on_world(
        engram_sys,
        world,
        max_slice=2,
        compute_uplift_against_fts=False,
    )
    assert engram_res.system_name == "engram-base-associative"
    assert engram_res.metrics.unsupported_claim_rate == 0.0
    assert engram_res.metrics.isolation_leakage == 0.0
    assert engram_res.metrics.deletion_leakage == 0.0
    assert engram_res.metrics.claim_label_accuracy == 1.0
    assert engram_res.metrics.abstention_quality >= 0.90

    auto_sys = resolve_builtin_system("freud-auto")
    auto_res = evaluate_system_on_world(
        auto_sys,
        world,
        max_slice=2,
        compute_uplift_against_fts=False,
    )
    assert auto_res.system_name == "freud-auto"
    assert auto_res.metrics.unsupported_claim_rate == 0.0
    assert auto_res.metrics.isolation_leakage == 0.0
    assert auto_res.metrics.deletion_leakage == 0.0
    assert auto_res.metrics.claim_label_accuracy == 1.0
    assert auto_res.metrics.overall_accuracy >= 0.65
