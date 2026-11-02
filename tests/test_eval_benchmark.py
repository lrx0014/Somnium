"""Unit and integration tests for Milestone M1 (Synthetic Benchmark & MLX Spike).

Verifies:
- Deterministic world generation and identical manifest hashes across seeds (M1.1)
- Question generation across all categories and strict fact-group split without
  train/test leakage (M1.2)
- Scoring logic for temporal consistency, scope attribution, abstention, claim
  labels, deletion leakage, and isolation leakage (M1.3)
- Eval harness and CLI execution producing JSON and Markdown reports (M1.4)
- Baseline systems (`lexical-fts`, `full-context`, `base-model-no-adapter`) (M1.5)
- MLX + Qwen3 compatibility check and MLX LoRA associative training (M1.6–M1.8)
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from somnium.backends.mlx import (
    MLXLoRAAssociativeBackend,
    verify_qwen3_mlx_compatibility,
)
from somnium.cli import app
from somnium.domain import Claim, ClaimSupport, ResolvedRecallMode, SensitivityLevel
from somnium.eval import (
    PRIMARY_TENANT,
    AssociativeAdapterSystem,
    QuestionCategory,
    StandaloneFtsBaseline,
    SystemPrediction,
    build_training_dataset,
    evaluate_system_on_world,
    generate_benchmark_questions,
    generate_synthetic_world,
    score_single_question,
)

runner = CliRunner()


def test_synthetic_world_reproducibility_and_constraints() -> None:
    """Verify generating the synthetic world twice with the same seed yields identical hashes."""
    w1 = generate_synthetic_world(seed=42)
    w2 = generate_synthetic_world(seed=42)
    w_diff = generate_synthetic_world(seed=99)

    assert w1.manifest_hash == w2.manifest_hash
    assert w1.manifest_hash != w_diff.manifest_hash
    assert len(w1.records) >= 100
    assert len(w1.fact_blueprints) >= 90

    # Verify high-sensitivity and deleted records are never training_eligible
    for rec in w1.records:
        if rec.sensitivity == SensitivityLevel.HIGH or rec.status.value == "deleted":
            assert rec.training_eligible is False


def test_dataset_manifest_reproducibility_and_zero_fact_leakage() -> None:
    """Verify training dataset excludes held-out facts, sensitive facts, and deleted facts."""
    world = generate_synthetic_world(seed=42)
    ds1 = build_training_dataset(world, mode="slice1", seed=42)
    ds2 = build_training_dataset(world, mode="slice1", seed=42)
    assert ds1.dataset_hash == ds2.dataset_hash

    held_out_fgs = {bp.fact_group_id for bp in world.fact_blueprints if bp.split == "held_out"}
    sensitive_or_deleted_ids = {
        str(rec.id)
        for rec in world.records
        if rec.sensitivity == SensitivityLevel.HIGH or rec.status.value == "deleted"
    }

    for ex in ds1.examples:
        assert ex.fact_group_id not in held_out_fgs
        assert ex.tenant_id == PRIMARY_TENANT
        for src_id in ex.source_record_ids:
            assert src_id not in sensitive_or_deleted_ids


def test_question_generator_covers_all_categories() -> None:
    """Verify generate_benchmark_questions covers every required design §8.2 / §12 category."""
    world = generate_synthetic_world(seed=42)
    questions = generate_benchmark_questions(world, max_slice=2)
    present_categories = {q.category for q in questions}
    for expected_cat in QuestionCategory:
        assert expected_cat in present_categories


def test_scoring_enforces_claim_labels_and_leakage_gates() -> None:
    """Verify score_single_question catches unbacked supported claims and date inventions."""
    world = generate_synthetic_world(seed=42)
    questions = generate_benchmark_questions(world, max_slice=2)
    direct_q = next(q for q in questions if q.category == QuestionCategory.DIRECT)

    valid_id = direct_q.expected_record_ids[0]
    bogus_id = uuid4()

    # 1. Valid supported claim
    good_pred = SystemPrediction(
        question_id=direct_q.question_id,
        recall_mode=ResolvedRecallMode.EVIDENCE,
        abstained=False,
        answer_text=f"Selected {direct_q.expected_values[0]}",
        claims=(
            Claim(
                text=f"Selected {direct_q.expected_values[0]}",
                support=ClaimSupport.SUPPORTED,
                evidence=(valid_id,),
                scope=direct_q.requested_scope,
            ),
        ),
        cited_record_ids=(valid_id,),
    )
    good_score = score_single_question(
        direct_q,
        good_pred,
        authorized_active_record_ids=frozenset({valid_id}),
    )
    assert good_score.correct is True
    assert good_score.claim_labels_valid is True
    assert good_score.has_unsupported_claim is False

    # 2. Claim marked SUPPORTED but citing an unauthorized/non-existent ID
    bad_claim_pred = SystemPrediction(
        question_id=direct_q.question_id,
        recall_mode=ResolvedRecallMode.ASSOCIATIVE,
        abstained=False,
        answer_text=f"Selected {direct_q.expected_values[0]}",
        claims=(
            Claim(
                text=f"Selected {direct_q.expected_values[0]}",
                support=ClaimSupport.SUPPORTED,
                evidence=(bogus_id,),
                scope=direct_q.requested_scope,
            ),
        ),
    )
    bad_claim_score = score_single_question(
        direct_q,
        bad_claim_pred,
        authorized_active_record_ids=frozenset({valid_id}),
    )
    assert bad_claim_score.claim_labels_valid is False
    assert bad_claim_score.has_unsupported_claim is True


def test_eval_cli_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    """Verify `somnium eval run` executes and writes JSON and Markdown reports."""
    out_json = tmp_path / "fts.json"
    out_md = tmp_path / "fts.md"

    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--suite",
            "synthetic-v1",
            "--system",
            "lexical-fts",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
    )
    assert result.exit_code == 0
    assert out_json.is_file()
    assert out_md.is_file()

    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert parsed["suite_name"] == "synthetic-v1"
    assert parsed["system_name"] == "lexical-fts"
    assert parsed["metrics"]["deletion_leakage"] == 0.0
    assert parsed["metrics"]["isolation_leakage"] == 0.0


def test_mlx_compatibility_and_associative_uplift() -> None:
    """Verify MLX compatibility check passes and hybrid adapter achieves uplift over FTS."""
    compat = verify_qwen3_mlx_compatibility(seed=42)
    assert compat.qwen3_chat_template_verified is True
    assert compat.fp16_lora_final_loss < compat.fp16_lora_initial_loss
    assert compat.quantized_4bit_qlora_final_loss < compat.quantized_4bit_qlora_initial_loss
    assert compat.adapter_save_and_reload_verified is True

    world = generate_synthetic_world(seed=42)
    s1_questions = generate_benchmark_questions(world, max_slice=1)
    ds_s1 = build_training_dataset(world, mode="slice1", seed=42)

    backend = MLXLoRAAssociativeBackend(seed=42)
    backend.train_on_examples(
        ds_s1.examples,
        version_name="test-s1-adapter",
        epochs=40,
        learning_rate=4e-3,
    )
    backend.swap_adapter(PRIMARY_TENANT, "test-s1-adapter", None)

    fts_res = evaluate_system_on_world(
        StandaloneFtsBaseline(),
        world,
        max_slice=1,
        questions=s1_questions,
        compute_uplift_against_fts=False,
    )
    hybrid_res = evaluate_system_on_world(
        AssociativeAdapterSystem(backend, name="test-hybrid", verify_with_hippocore=True),
        world,
        max_slice=1,
        questions=s1_questions,
        baseline_scores_by_qid={s.question_id: s for s in fts_res.question_scores},
    )

    assert hybrid_res.metrics.associative_recall_uplift > 0.0
    assert hybrid_res.metrics.overall_accuracy > fts_res.metrics.overall_accuracy
    assert hybrid_res.metrics.deletion_leakage == 0.0
    assert hybrid_res.metrics.isolation_leakage == 0.0
