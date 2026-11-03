"""Dream Engine: Idle detection, lifecycle state machine, dataset building, and promotion."""

from __future__ import annotations

from somnium.dream.artifacts import LocalFilesystemArtifactStore
from somnium.dream.augmentation import (
    AUGMENTATION_VERSION,
    AugmentationCache,
    AugmentedQuestions,
    compute_augmentation_key,
    synthesize_default_questions,
)
from somnium.dream.datasets import (
    DEFAULT_DATASET_GENERATOR_VERSION,
    DatasetManifest,
    build_tenant_dataset,
    list_dataset_manifests,
    load_dataset_manifest,
    save_dataset_manifest_to_db,
)
from somnium.dream.engine import DreamEngine
from somnium.dream.gates import GateReport, GateTolerances, PromotionGateEvaluator
from somnium.dream.idle import TrafficTracker
from somnium.dream.pipeline import DreamCycleResult, execute_dream_cycle
from somnium.dream.registry import (
    DreamRunRecord,
    ModelRegistry,
    ModelVersionRecord,
    PromotionResult,
    RollbackResult,
)
from somnium.dream.resources import (
    ResourcePolicy,
    StaticResourcePolicy,
    SystemResourcePolicy,
)

__all__ = [
    "AUGMENTATION_VERSION",
    "DEFAULT_DATASET_GENERATOR_VERSION",
    "AugmentationCache",
    "AugmentedQuestions",
    "DatasetManifest",
    "DreamCycleResult",
    "DreamEngine",
    "DreamRunRecord",
    "GateReport",
    "GateTolerances",
    "LocalFilesystemArtifactStore",
    "ModelRegistry",
    "ModelVersionRecord",
    "PromotionGateEvaluator",
    "PromotionResult",
    "ResourcePolicy",
    "RollbackResult",
    "StaticResourcePolicy",
    "SystemResourcePolicy",
    "TrafficTracker",
    "build_tenant_dataset",
    "compute_augmentation_key",
    "execute_dream_cycle",
    "list_dataset_manifests",
    "load_dataset_manifest",
    "save_dataset_manifest_to_db",
    "synthesize_default_questions",
]
