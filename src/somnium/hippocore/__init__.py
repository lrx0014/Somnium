"""Hippocore: Authoritative episodic event store, retrieval, provenance, and revisions."""

from __future__ import annotations

from somnium.hippocore.migrations import (
    HEAD_REVISION,
    INITIAL_REVISION,
    current_revision,
    downgrade_database,
    upgrade_database,
)
from somnium.hippocore.retriever import HippocoreFtsRetriever
from somnium.hippocore.schema import (
    dataset_manifests,
    deletion_requests,
    dream_runs,
    memory_access,
    memory_events,
    memory_facts,
    memory_revisions,
    memory_sources,
    metadata,
    model_versions,
    routing_decisions,
    tenant_state,
    training_examples,
)
from somnium.hippocore.security import (
    SecretDetectedError,
    assert_no_secrets,
    classify_sensitivity,
    compute_training_eligibility,
    normalize_content,
)
from somnium.hippocore.store import AccessStats, HippocoreStore, build_scoped_where

__all__ = [
    "HEAD_REVISION",
    "INITIAL_REVISION",
    "AccessStats",
    "HippocoreFtsRetriever",
    "HippocoreStore",
    "SecretDetectedError",
    "assert_no_secrets",
    "build_scoped_where",
    "classify_sensitivity",
    "compute_training_eligibility",
    "current_revision",
    "dataset_manifests",
    "deletion_requests",
    "downgrade_database",
    "dream_runs",
    "memory_access",
    "memory_events",
    "memory_facts",
    "memory_revisions",
    "memory_sources",
    "metadata",
    "model_versions",
    "normalize_content",
    "routing_decisions",
    "tenant_state",
    "training_examples",
    "upgrade_database",
]
