# Changelog

All notable changes to **Somnium** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-11-03

### Milestone M6 — Dream Engine Automation & v0.1 MVP Release

This release delivers the complete **Somnium v0.1 MVP**, implementing bio-inspired, dream-driven neural memory for AI agents (integrating with OpenAI Codex via Model Context Protocol).

### Added

#### M6: Dream Engine Automation
- **Traffic Tracking & Idle Detection (`somnium.dream.idle`)**:
  - `TrafficTracker` monitoring external agent traffic (`store`, `ask`, `correct`, `forget`).
  - Configurable inactivity threshold (`idle_threshold_seconds`, default 1800s).
- **Per-Tenant State Machine (`somnium.dream.engine`)**:
  - Full lifecycle states: `AWAKE` $\to$ `IDLE_ELIGIBLE` $\to$ `DREAM_PREPARING` $\to$ `TRAINING` $\to$ `EVALUATING` $\to$ `PROMOTING` / `REJECTED` $\to$ `AWAKE`.
  - Crash recovery (`recover_crashed_states`): automatically detects and resets orphaned in-progress states back to `AWAKE`.
  - Single-writer per-tenant execution lock.
- **Environment Resource Policy (`somnium.dream.resources`)**:
  - `SystemResourcePolicy` for macOS Apple Silicon laptops: checks AC power connection via `pmset` and memory pressure via `vm_stat`; defers training when on battery or under heavy memory pressure.
  - `StaticResourcePolicy` for deterministic testing.
- **Associative Traffic Pause / Deferral**:
  - In development profile, associative recall requests trigger cooperative deferral of active training, ensuring serving availability. Evidence recall remains 100% available without pause.
- **Soak Testing (`tests/test_dream_soak.py`)**:
  - Continuous concurrent mixed queries (`store` + `ask` in evidence recall mode) executing throughout training and atomic adapter promotion with 0% request failures.

#### M5: Training Pipeline & Model Registry
- **Dataset Builder (`somnium.dream.datasets`)**:
  - Deterministic construction of backend-neutral JSONL training datasets strictly from verified, training-eligible Hippocore records.
  - Generates direct QA, scope attribution, incomplete cue, temporal correction, evidence reading / claim labeling, and explicit unknown/abstention examples.
  - Deterministic split by fact group preventing train/eval data leakage.
- **Augmentation Cache (`somnium.dream.augmentation`)**:
  - Content-hashed question synthesis (`SHA-256`) guaranteeing identical dataset manifests when rebuilt from the same inputs.
- **MLX Training Backend (`somnium.backends.mlx.trainer`)**:
  - LoRA fine-tuning backend using MLX AdamW on Apple Silicon Metal GPU.
  - Self-contained checkpoint packaging (weights in `safetensors` + `adapter_config.json` inside ZIP archive).
- **Artifact Store (`somnium.dream.artifacts`)**:
  - Content-addressed local filesystem storage (`LocalFilesystemArtifactStore`) for dataset manifests and adapter checkpoints with SHA-256 integrity validation.
- **Model Registry & Promotion Gate (`somnium.dream.registry`, `somnium.dream.gates`)**:
  - Lineage tracking in `model_versions` and audit logging in `dream_runs`.
  - Promotion gate enforcing 0% isolation leakage, 0% deletion leakage, and 0% unsupported claims.
  - Atomic promotion and rollback coordinating with `InferenceBackend.swap_adapter`.
- **Deletion Handling**:
  - Deleting training-eligible memories marks tenant `weights_dirty = True`, automatically escalating the next Dream cycle to a full retrain and purging deleted data.

#### M4: Engram Associative Recall & Routing
- **Engram Associative Pipeline (`somnium.engram.pipeline`)**:
  - 3-stage `associate -> verify -> answer` pipeline on the base model with structured JSON repair and fallbacks.
  - Deterministic post-generation guards: every `supported` claim must cite valid evidence within authorized scopes, downgrading unsupported assertions to `unverified`.
- **Rule-Based Routing (`somnium.engram.router`)**:
  - Dynamic routing between `evidence` and `associative` recall for `mode=auto`.

#### M3: Freud MCP Gateway & Evidence Recall
- **MCP Stdio Server (`somnium.freud.mcp_server`)**:
  - Versioned tools: `memory_ask`, `memory_store`, `memory_correct`, `memory_forget`, `memory_status`.
  - Untrusted data delimiters (`<untrusted_memory>`) preventing prompt injection.

#### M2: Hippocore Episodic Store
- **Scoped Repository (`somnium.hippocore.store`)**:
  - Strict SQL-level tenant and project scope isolation.
  - Immutable revision history and SQLite FTS5 lexical retrieval.
  - Hard and soft deletion with tombstoning and audit logging.

#### M1 & M0: Synthetic Benchmark Suite & Foundation
- Reproducible synthetic world generator and design §12 scoring harness.
- Configuration profiles (`dev`, `test`, `prod`) and Typer CLI (`somnium`).
- Strict architectural boundary enforcement (`import-linter` forbidding ML imports outside `somnium.backends`).
