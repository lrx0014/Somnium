"""Model registry and Dream run lineage tracking (M5.5).

Per `design.md` §8.3 and §15.1:
- Tracks adapter versions, base model revisions, training lineage, and evaluation metrics.
- Manages per-tenant active model pointers with atomic promotion and rollback.
- Coordinates adapter swapping with `InferenceBackend`.
- Logs end-to-end Dream run lifecycle states and gate decisions.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Engine, desc, select

from somnium.backends.base import InferenceBackend
from somnium.domain import TenantId
from somnium.dream.gates import GateReport
from somnium.hippocore.schema import dream_runs, model_versions, tenant_state


@dataclass(frozen=True, slots=True)
class ModelVersionRecord:
    """Persisted metadata for a trained model or adapter version."""

    model_version: str
    tenant_id: TenantId
    base_model_id: str
    base_model_revision: str
    parent_version: str | None
    training_mode: str
    dataset_manifest_hash: str
    checkpoint_uri: str
    status: str
    metrics: dict[str, float]
    gate_report: dict[str, Any]
    created_at: str
    promoted_at: str | None


@dataclass(frozen=True, slots=True)
class DreamRunRecord:
    """Audit log entry for an end-to-end Dream cycle execution."""

    run_id: str
    tenant_id: TenantId
    requested_mode: str
    resolved_mode: str
    forced_full_retrain_reason: str | None
    state: str
    parent_model_version: str | None
    candidate_model_version: str | None
    dataset_manifest_hash: str | None
    training_metrics: dict[str, float]
    gate_passed: bool
    gate_report: dict[str, Any]
    started_at: str
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Outcome of an atomic adapter promotion."""

    tenant_id: TenantId
    promoted_version: str
    previous_version: str | None
    promoted_at: str


@dataclass(frozen=True, slots=True)
class RollbackResult:
    """Outcome of an atomic adapter rollback."""

    tenant_id: TenantId
    active_version: str
    rolled_back_version: str
    rolled_back_at: str


class ModelRegistry:
    """Central repository for model versions, tenant active pointers, and Dream runs."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def register_version(
        self,
        *,
        model_version: str,
        tenant_id: TenantId,
        base_model_id: str,
        base_model_revision: str,
        parent_version: str | None,
        training_mode: str,
        dataset_manifest_hash: str,
        checkpoint_uri: str,
        status: str = "candidate",
        metrics: dict[str, float] | None = None,
        gate_report: GateReport | dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> ModelVersionRecord:
        """Register a new candidate or base model version in the registry."""
        now_iso = created_at or datetime.now(tz=UTC).isoformat()
        metrics_dict = metrics or {}
        report_dict: dict[str, Any] = {}
        if isinstance(gate_report, GateReport):
            report_dict = gate_report.to_dict()
        elif isinstance(gate_report, dict):
            report_dict = gate_report

        with self.engine.begin() as conn:
            conn.execute(
                model_versions.insert().values(
                    model_version=model_version,
                    tenant_id=tenant_id,
                    base_model_id=base_model_id,
                    base_model_revision=base_model_revision,
                    parent_version=parent_version,
                    training_mode=training_mode,
                    dataset_manifest_hash=dataset_manifest_hash,
                    checkpoint_uri=checkpoint_uri,
                    status=status,
                    metrics_json=json.dumps(metrics_dict),
                    gate_report_json=json.dumps(report_dict),
                    created_at=now_iso,
                    promoted_at=None,
                )
            )

        return ModelVersionRecord(
            model_version=model_version,
            tenant_id=tenant_id,
            base_model_id=base_model_id,
            base_model_revision=base_model_revision,
            parent_version=parent_version,
            training_mode=training_mode,
            dataset_manifest_hash=dataset_manifest_hash,
            checkpoint_uri=checkpoint_uri,
            status=status,
            metrics=metrics_dict,
            gate_report=report_dict,
            created_at=now_iso,
            promoted_at=None,
        )

    def get_version(self, model_version: str) -> ModelVersionRecord | None:
        """Fetch a ModelVersionRecord by primary key `model_version`."""
        with self.engine.connect() as conn:
            row = conn.execute(
                select(model_versions).where(model_versions.c.model_version == model_version)
            ).first()
            if row is None:
                return None
            m = row._mapping
            return ModelVersionRecord(
                model_version=str(m["model_version"]),
                tenant_id=str(m["tenant_id"]),
                base_model_id=str(m["base_model_id"]),
                base_model_revision=str(m["base_model_revision"]),
                parent_version=str(m["parent_version"]) if m["parent_version"] else None,
                training_mode=str(m["training_mode"]),
                dataset_manifest_hash=str(m["dataset_manifest_hash"]),
                checkpoint_uri=str(m["checkpoint_uri"]),
                status=str(m["status"]),
                metrics=dict(json.loads(str(m["metrics_json"]))),
                gate_report=dict(json.loads(str(m["gate_report_json"]))),
                created_at=str(m["created_at"]),
                promoted_at=str(m["promoted_at"]) if m["promoted_at"] else None,
            )

    def list_versions(
        self,
        tenant_id: TenantId,
        *,
        status: str | None = None,
    ) -> Sequence[ModelVersionRecord]:
        """List model versions for a tenant ordered by created_at descending."""
        with self.engine.connect() as conn:
            stmt = select(model_versions).where(model_versions.c.tenant_id == tenant_id)
            if status is not None:
                stmt = stmt.where(model_versions.c.status == status)
            stmt = stmt.order_by(desc(model_versions.c.created_at))
            rows = conn.execute(stmt).fetchall()

            records: list[ModelVersionRecord] = []
            for r in rows:
                m = r._mapping
                records.append(
                    ModelVersionRecord(
                        model_version=str(m["model_version"]),
                        tenant_id=str(m["tenant_id"]),
                        base_model_id=str(m["base_model_id"]),
                        base_model_revision=str(m["base_model_revision"]),
                        parent_version=str(m["parent_version"]) if m["parent_version"] else None,
                        training_mode=str(m["training_mode"]),
                        dataset_manifest_hash=str(m["dataset_manifest_hash"]),
                        checkpoint_uri=str(m["checkpoint_uri"]),
                        status=str(m["status"]),
                        metrics=dict(json.loads(str(m["metrics_json"]))),
                        gate_report=dict(json.loads(str(m["gate_report_json"]))),
                        created_at=str(m["created_at"]),
                        promoted_at=str(m["promoted_at"]) if m["promoted_at"] else None,
                    )
                )
            return tuple(records)

    def get_active_version(self, tenant_id: TenantId) -> str | None:
        """Fetch the currently active model version for a tenant."""
        with self.engine.connect() as conn:
            row = conn.execute(
                select(tenant_state.c.active_model_version).where(
                    tenant_state.c.tenant_id == tenant_id
                )
            ).first()
            if row is not None and row[0]:
                return str(row[0])
            return None

    def is_weights_dirty(self, tenant_id: TenantId) -> bool:
        """Check if tenant weights are marked dirty (e.g. from a training-eligible deletion)."""
        with self.engine.connect() as conn:
            row = conn.execute(
                select(tenant_state.c.weights_dirty).where(tenant_state.c.tenant_id == tenant_id)
            ).first()
            if row is not None and row[0] is not None:
                return bool(row[0])
            return False

    def set_weights_dirty(self, tenant_id: TenantId, dirty: bool = True) -> None:
        """Update the weights_dirty flag for a tenant."""
        now_iso = datetime.now(tz=UTC).isoformat()
        with self.engine.begin() as conn:
            row = conn.execute(
                select(tenant_state.c.tenant_id).where(tenant_state.c.tenant_id == tenant_id)
            ).first()
            if row is None:
                conn.execute(
                    tenant_state.insert().values(
                        tenant_id=tenant_id,
                        weights_dirty=dirty,
                        active_model_version=None,
                        dream_state="awake",
                        updated_at=now_iso,
                    )
                )
            else:
                conn.execute(
                    tenant_state.update()
                    .where(tenant_state.c.tenant_id == tenant_id)
                    .values(weights_dirty=dirty, updated_at=now_iso)
                )

    def promote(
        self,
        tenant_id: TenantId,
        candidate_version: str,
        *,
        inference_backend: InferenceBackend | None = None,
        force: bool = False,
    ) -> PromotionResult:
        """Atomically promote `candidate_version` to active for `tenant_id`."""
        candidate = self.get_version(candidate_version)
        if candidate is None:
            raise KeyError(f"Candidate model version '{candidate_version}' not found.")
        if candidate.tenant_id != tenant_id:
            raise PermissionError(
                f"Candidate version '{candidate_version}' belongs to tenant "
                f"'{candidate.tenant_id}', not '{tenant_id}'."
            )
        if not force and candidate.status == "rejected":
            raise ValueError(
                f"Cannot promote rejected candidate version '{candidate_version}' "
                f"without force=True."
            )

        now_iso = datetime.now(tz=UTC).isoformat()
        previous_version = self.get_active_version(tenant_id)

        # 1. Coordinate atomic adapter swap on InferenceBackend if available
        if inference_backend is not None:
            inference_backend.swap_adapter(
                tenant_id=tenant_id,
                new_model_version=candidate_version,
                adapter_uri=candidate.checkpoint_uri,
            )

        # 2. Update model registry and tenant active pointer atomically in DB
        with self.engine.begin() as conn:
            conn.execute(
                model_versions.update()
                .where(model_versions.c.model_version == candidate_version)
                .values(status="promoted", promoted_at=now_iso)
            )

            # Ensure tenant_state row exists and point active_model_version to candidate
            state_row = conn.execute(
                select(tenant_state.c.tenant_id).where(tenant_state.c.tenant_id == tenant_id)
            ).first()
            if state_row is None:
                conn.execute(
                    tenant_state.insert().values(
                        tenant_id=tenant_id,
                        weights_dirty=False,
                        active_model_version=candidate_version,
                        dream_state="awake",
                        updated_at=now_iso,
                    )
                )
            else:
                conn.execute(
                    tenant_state.update()
                    .where(tenant_state.c.tenant_id == tenant_id)
                    .values(
                        active_model_version=candidate_version,
                        weights_dirty=False,
                        updated_at=now_iso,
                    )
                )

        return PromotionResult(
            tenant_id=tenant_id,
            promoted_version=candidate_version,
            previous_version=previous_version,
            promoted_at=now_iso,
        )

    def rollback(
        self,
        tenant_id: TenantId,
        *,
        target_version: str | None = None,
        inference_backend: InferenceBackend | None = None,
    ) -> RollbackResult:
        """Atomically rollback `tenant_id` to `target_version` or parent version."""
        current_active = self.get_active_version(tenant_id)
        if current_active is None:
            raise ValueError(f"Tenant '{tenant_id}' has no active model version to roll back.")

        resolved_target: str
        target_checkpoint_uri: str | None = None

        if target_version is not None:
            resolved_target = target_version
            rec = self.get_version(resolved_target)
            if rec is not None:
                target_checkpoint_uri = rec.checkpoint_uri
        else:
            # Look up parent version of current active version
            curr_rec = self.get_version(current_active)
            if curr_rec is not None and curr_rec.parent_version:
                resolved_target = curr_rec.parent_version
                parent_rec = self.get_version(resolved_target)
                if parent_rec is not None:
                    target_checkpoint_uri = parent_rec.checkpoint_uri
            else:
                # Find previous promoted version or fall back to base
                versions = self.list_versions(tenant_id, status="promoted")
                earlier = [v for v in versions if v.model_version != current_active]
                if earlier:
                    resolved_target = earlier[0].model_version
                    target_checkpoint_uri = earlier[0].checkpoint_uri
                elif curr_rec is not None:
                    resolved_target = curr_rec.base_model_id
                else:
                    resolved_target = "base"

        now_iso = datetime.now(tz=UTC).isoformat()

        # 1. Perform atomic adapter swap
        if inference_backend is not None:
            inference_backend.swap_adapter(
                tenant_id=tenant_id,
                new_model_version=resolved_target,
                adapter_uri=target_checkpoint_uri,
            )

        # 2. Update DB state
        with self.engine.begin() as conn:
            conn.execute(
                model_versions.update()
                .where(model_versions.c.model_version == current_active)
                .values(status="rolled_back")
            )
            conn.execute(
                tenant_state.update()
                .where(tenant_state.c.tenant_id == tenant_id)
                .values(active_model_version=resolved_target, updated_at=now_iso)
            )

        return RollbackResult(
            tenant_id=tenant_id,
            active_version=resolved_target,
            rolled_back_version=current_active,
            rolled_back_at=now_iso,
        )

    def start_dream_run(
        self,
        *,
        tenant_id: TenantId,
        requested_mode: str,
        resolved_mode: str,
        forced_full_retrain_reason: str | None = None,
        parent_model_version: str | None = None,
        dataset_manifest_hash: str | None = None,
        run_id: str | None = None,
    ) -> DreamRunRecord:
        """Create and persist a new DreamRunRecord in state 'running'."""
        effective_run_id = run_id or f"run-{uuid4().hex[:12]}"
        now_iso = datetime.now(tz=UTC).isoformat()

        with self.engine.begin() as conn:
            conn.execute(
                dream_runs.insert().values(
                    run_id=effective_run_id,
                    tenant_id=tenant_id,
                    requested_mode=requested_mode,
                    resolved_mode=resolved_mode,
                    forced_full_retrain_reason=forced_full_retrain_reason,
                    state="running",
                    parent_model_version=parent_model_version,
                    candidate_model_version=None,
                    dataset_manifest_hash=dataset_manifest_hash,
                    training_metrics_json="{}",
                    gate_passed=False,
                    gate_report_json="{}",
                    started_at=now_iso,
                    completed_at=None,
                )
            )

        return DreamRunRecord(
            run_id=effective_run_id,
            tenant_id=tenant_id,
            requested_mode=requested_mode,
            resolved_mode=resolved_mode,
            forced_full_retrain_reason=forced_full_retrain_reason,
            state="running",
            parent_model_version=parent_model_version,
            candidate_model_version=None,
            dataset_manifest_hash=dataset_manifest_hash,
            training_metrics={},
            gate_passed=False,
            gate_report={},
            started_at=now_iso,
            completed_at=None,
        )

    def complete_dream_run(
        self,
        run_id: str,
        *,
        state: str,
        candidate_model_version: str | None = None,
        dataset_manifest_hash: str | None = None,
        training_metrics: dict[str, float] | None = None,
        gate_passed: bool = False,
        gate_report: GateReport | dict[str, Any] | None = None,
    ) -> DreamRunRecord:
        """Mark a Dream run as completed/rejected/failed with metrics and gate report."""
        now_iso = datetime.now(tz=UTC).isoformat()
        t_metrics = training_metrics or {}
        g_dict: dict[str, Any] = {}
        if isinstance(gate_report, GateReport):
            g_dict = gate_report.to_dict()
        elif isinstance(gate_report, dict):
            g_dict = gate_report

        with self.engine.begin() as conn:
            conn.execute(
                dream_runs.update()
                .where(dream_runs.c.run_id == run_id)
                .values(
                    state=state,
                    candidate_model_version=candidate_model_version,
                    dataset_manifest_hash=dataset_manifest_hash,
                    training_metrics_json=json.dumps(t_metrics),
                    gate_passed=gate_passed,
                    gate_report_json=json.dumps(g_dict),
                    completed_at=now_iso,
                )
            )

        run = self.get_dream_run(run_id)
        if run is None:
            raise KeyError(f"Dream run '{run_id}' not found after update.")
        return run

    def get_dream_run(self, run_id: str) -> DreamRunRecord | None:
        """Fetch a DreamRunRecord by `run_id`."""
        with self.engine.connect() as conn:
            row = conn.execute(select(dream_runs).where(dream_runs.c.run_id == run_id)).first()
            if row is None:
                return None
            m = row._mapping
            return DreamRunRecord(
                run_id=str(m["run_id"]),
                tenant_id=str(m["tenant_id"]),
                requested_mode=str(m["requested_mode"]),
                resolved_mode=str(m["resolved_mode"]),
                forced_full_retrain_reason=str(m["forced_full_retrain_reason"])
                if m["forced_full_retrain_reason"]
                else None,
                state=str(m["state"]),
                parent_model_version=str(m["parent_model_version"])
                if m["parent_model_version"]
                else None,
                candidate_model_version=str(m["candidate_model_version"])
                if m["candidate_model_version"]
                else None,
                dataset_manifest_hash=str(m["dataset_manifest_hash"])
                if m["dataset_manifest_hash"]
                else None,
                training_metrics=dict(json.loads(str(m["training_metrics_json"]))),
                gate_passed=bool(m["gate_passed"]),
                gate_report=dict(json.loads(str(m["gate_report_json"]))),
                started_at=str(m["started_at"]),
                completed_at=str(m["completed_at"]) if m["completed_at"] else None,
            )

    def list_dream_runs(
        self,
        tenant_id: TenantId,
        *,
        limit: int = 20,
    ) -> Sequence[DreamRunRecord]:
        """List Dream runs for `tenant_id` ordered by started_at descending."""
        with self.engine.connect() as conn:
            stmt = (
                select(dream_runs)
                .where(dream_runs.c.tenant_id == tenant_id)
                .order_by(desc(dream_runs.c.started_at))
                .limit(limit)
            )
            rows = conn.execute(stmt).fetchall()
            records: list[DreamRunRecord] = []
            for r in rows:
                m = r._mapping
                records.append(
                    DreamRunRecord(
                        run_id=str(m["run_id"]),
                        tenant_id=str(m["tenant_id"]),
                        requested_mode=str(m["requested_mode"]),
                        resolved_mode=str(m["resolved_mode"]),
                        forced_full_retrain_reason=str(m["forced_full_retrain_reason"])
                        if m["forced_full_retrain_reason"]
                        else None,
                        state=str(m["state"]),
                        parent_model_version=str(m["parent_model_version"])
                        if m["parent_model_version"]
                        else None,
                        candidate_model_version=str(m["candidate_model_version"])
                        if m["candidate_model_version"]
                        else None,
                        dataset_manifest_hash=str(m["dataset_manifest_hash"])
                        if m["dataset_manifest_hash"]
                        else None,
                        training_metrics=dict(json.loads(str(m["training_metrics_json"]))),
                        gate_passed=bool(m["gate_passed"]),
                        gate_report=dict(json.loads(str(m["gate_report_json"]))),
                        started_at=str(m["started_at"]),
                        completed_at=str(m["completed_at"]) if m["completed_at"] else None,
                    )
                )
            return tuple(records)


__all__ = [
    "DreamRunRecord",
    "ModelRegistry",
    "ModelVersionRecord",
    "PromotionResult",
    "RollbackResult",
]
