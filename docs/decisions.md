# Somnium — Architecture Decision Log (ADRs)

This file records resolved architectural decisions, their context, alternatives considered, and consequences, in accordance with `design.md` §14.12 and `Milestones.md` M0.10.

---

## ADR Template

```markdown
### ADR-NNNN: <Title>
- **Status:** Proposed | Accepted | Superseded by ADR-XXXX
- **Date:** YYYY-MM-DD
- **Context:** What problem or constraint forced this decision?
- **Decision:** What did we decide to do?
- **Alternatives Considered:** What other options were weighed and why were they rejected?
- **Consequences:** What becomes easier, harder, or required as a result?
```

---

## Accepted Decisions

### ADR-0001: All-Python Implementation with Pluggable Backends
- **Status:** Accepted
- **Date:** 2026-11-02
- **Context:** The initial v0.1 draft proposed a split Go (gateway, store, orchestrator) + Python (model inference and training) stack targeting only Apple Silicon. However, Somnium is developed locally on Apple Silicon (MLX) and deployed to Linux GPU servers (CUDA / PyTorch / vLLM), and its throughput target (single user or small team) does not require a high-concurrency gateway language.
- **Decision:** Implement the entire Somnium monorepo in Python 3.12+ behind narrow `Protocol` interfaces (`InferenceBackend`, `TrainingBackend`, `MemoryStore`, `Retriever`, `RecallRouter`). Isolate ML libraries (`mlx`, `torch`, `peft`, `vllm`, `transformers`) inside `somnium/backends/<backend>/` as optional dependency groups (`somnium[mlx]`, `somnium[cuda]`).
- **Alternatives Considered:**
  - *Go gateway/orchestrator + Python ML worker:* Rejected due to duplicated schema definitions, cross-process serialization boilerplate, and added operational friction without a measured throughput bottleneck.
- **Consequences:**
  - Single set of Pydantic schemas and domain types across all components.
  - Core package installs and runs all unit/contract tests on macOS and Linux without ML dependencies.
  - Import boundaries must be enforced automatically in CI (`import-linter` and AST boundary tests).
  - Adapter weights are not assumed portable between MLX and CUDA backends; dataset manifests are the portable unit.

---

### ADR-0002: Engram's Role as Answer Generator for Associative Recall
- **Status:** Accepted
- **Date:** 2026-11-02
- **Context:** Engram could either act as the natural-language answer generator for associative queries or act solely as a generative retriever that emits candidate memory IDs/cue keywords for another model to synthesize.
- **Decision:** Engram is the natural-language answer generator for associative recall. It executes an `associate -> verify -> answer` pipeline combining parametric recall with Hippocore evidence, and labels each claim as `supported` (with memory IDs) or `unverified`.
- **Alternatives Considered:**
  - *Engram as an associative ID/keyword retriever only:* Considered for lower hallucination risk, but rejected to preserve a unified memory intelligence behind Freud.
- **Consequences:**
  - Fine-tuning must preserve Engram's grounded-reading and instruction-following capabilities via evidence-reading training examples (`design.md` §8.2) and regression gates (`design.md` §12).
  - Deterministic post-generation guards must verify that every `supported` claim cites valid, authorized Hippocore records and that retrieved evidence overrides parametric memory on conflict (`design.md` §6.3).
  - Production base model size must be selected with grounded answer quality in mind.

---

### ADR-0003: Dual Recall Modes (`evidence` and `associative`) with `auto` Routing
- **Status:** Accepted
- **Date:** 2026-11-02
- **Context:** Relying on a fine-tuned model for every recall query introduces latency, GPU cost, and potential project mix-ups on simple factual lookups, whereas relying only on lexical/vector search misses vague cues and cross-event synthesis.
- **Decision:** Provide two explicit recall modes plus policy-based routing:
  1. `evidence` recall — direct, strictly scope-isolated retrieval of structured records from Hippocore with zero text generation.
  2. `associative` recall — Engram's parametric association + Hippocore verification + natural-language answer with per-claim support labels.
  3. `auto` (default) — rule-based cascade that runs `evidence` recall first and escalates to `associative` recall when relevance is low, cues are vague, or synthesis is requested.
- **Alternatives Considered:**
  - *Single unified Engram path for all queries:* Rejected because exact factual lookups do not benefit from GPU generation and require strict project-level isolation.
- **Consequences:**
  - Newly stored memories are immediately available via `evidence` recall without waiting for a Dream cycle.
  - System availability never depends on GPU or model health (`auto` falls back to `evidence`).
  - `evidence` recall serves as the built-in baseline to measure **associative-recall uplift** (`design.md` §12).

---

### ADR-0004: Multi-Layer Isolation Model (Project in Store, Tenant in Weights, Sensitive Excluded)
- **Status:** Accepted
- **Date:** 2026-11-02
- **Context:** Relational/indexed stores can filter records strictly by project (`scope`), but neural network weights cannot be partitioned by project once trained on a combined dataset.
- **Decision:** Enforce isolation at the layer where each guarantee is mechanically sound (`design.md` §5.4):
  1. **Hippocore / `evidence` recall:** Strict **project (`scope`)** isolation enforced inside the query builder via `AuthContext`.
  2. **Engram weights / `associative` recall:** **Tenant (user)** isolation — one adapter lineage per tenant trained across that tenant's training-eligible projects. Cross-project association within the same tenant is treated as an explicit, scope-labeled feature.
  3. **Sensitive records (`sensitivity: high`):** Excluded from training (`training_eligible = False`); never enter model weights.
- **Alternatives Considered:**
  - *One adapter per project:* Rejected because per-project datasets are too small, global preferences would not transfer, and cross-project association would be lost.
  - *Prompt-only project filtering on a shared multi-user adapter:* Rejected as a security boundary because weights can leak across prompts.
- **Consequences:**
  - Every record, run, and model version carries `tenant_id` from day one.
  - `MemoryRecord` automatically enforces `training_eligible = False` when `sensitivity == "high"`.
  - Associative recall requires the `associative_recall` permission on the tenant and labels cross-scope claims with their source scope.

---

### ADR-0005: Per-Tenant Training Scope with Incremental Updates and Periodic Full Retrains
- **Status:** Accepted
- **Date:** 2026-11-02
- **Context:** Continual LoRA fine-tuning on new memories risks catastrophic forgetting and parameter drift over many cycles, and cannot instantaneously unlearn deleted records.
- **Decision:** Support two training modes per tenant (`design.md` §8.3):
  - **Incremental (default for new memories):** Continue from the tenant's last accepted adapter using new verified records + historical replay + negative/correction/scope-attribution examples.
  - **Full retrain (from clean base):** Train a fresh adapter from the clean base model on the full eligible dataset. Mandatory after base-model changes, after deletion of any record that may have entered weights (`weights_dirty`), and periodically to bound drift.
- **Alternatives Considered:**
  - *Incremental-only forever:* Rejected because deletions and base-model upgrades require clean retraining, and unbounded incremental steps accumulate drift.
- **Consequences:**
  - Deleting a training-eligible record sets `weights_dirty = True` for the tenant and suppresses unverified revival via Hippocore verification until a full retrain is promoted.
  - Both modes are evaluated against the same promotion gate and regression suite before atomic promotion.

---

### ADR-0006: Mechanism-Based Naming for Recall Modes (`evidence` / `associative`)
- **Status:** Accepted
- **Date:** 2026-11-02
- **Context:** Initial drafts used `fast` and `deep` to name the two recall modes. However, in cognitive psychology (Dual Process Theory / Kahneman), "fast" refers to intuitive/associative recall whereas slow deliberation is analytical — the opposite of how `fast` (database lookup) and `deep` (neural association) operated in Somnium.
- **Decision:** Name the recall modes by their mechanism: `evidence` (Hippocore structured record lookup) and `associative` (Engram parametric association + verification). Rename the benchmark metric for paraphrased/incomplete cues to **Cue-based recall** to avoid colliding with the mode name.
- **Alternatives Considered:**
  - *`fast` / `deep`:* Rejected due to misleading cognitive analogy.
- **Consequences:**
  - `RecallMode` enum values are `auto`, `evidence`, and `associative` across all APIs, configs, CLI commands, and documentation.

---

### ADR-0007: Phase 0 Go/Adjust Decision on Hybrid Associative Recall and Training Strategy
- **Status:** Accepted
- **Date:** 2026-11-02
- **Context:** Milestone M1 (Phase 0 Research Spike, `experiments/phase0/REPORT.md`) evaluated MLX LoRA compatibility and compared `lexical-fts`, `full-context`, `base-model-no-adapter`, `parametric-only` LoRA adapters, and `hybrid` (`associate -> verify -> answer`) adapters across World Slice 1 and Slice 2 (under incremental no-replay, incremental with 50% replay, and full retrain).
- **Decision:**
  1. **Go on Hybrid Associative Recall (`associate -> verify -> answer`):** Deprioritize standalone unverified parametric answers. Associative recall must always run the Hippocore verification pass (`design.md` §6.3) so that:
     - Vague associative cues (`incomplete_cue`) gain positive uplift over `lexical-fts`.
     - Unseen / newly written (`held_out`) facts are still recalled accurately via Hippocore.
     - Stale parametric weights citing superseded records are automatically upgraded to the active replacement record via Hippocore's `supersedes` links (**Conflict Rule**).
  2. **Retain both Incremental Replay (50% historical replay) and Full Retrain in M5:** Incremental fine-tuning without replay exhibits severe catastrophic forgetting on Slice 1 retention and is rejected. Incremental fine-tuning with >= 50% historical replay is viable for routine Dream cycles, while full retrain from a clean base remains mandatory on `weights_dirty` deletions, base-model changes, and periodic drift resets.
- **Alternatives Considered:**
  - *Parametric-only recall without Hippocore verification:* Rejected because it scores `0.00%` on unseen (`held_out`) facts and fails temporal consistency when facts are superseded between Dream cycles.
  - *Incremental fine-tuning without historical replay:* Rejected due to catastrophic forgetting of Slice 1 memories.
- **Consequences:**
  - M4 (`engram`) will implement the 3-step `associate -> verify -> answer` pipeline with deterministic post-generation guards and `supersedes` resolution.
  - M5 (`dream`) will enforce a minimum 50% historical replay ratio on incremental runs and support full retrains from clean base weights.
  - Production model selection in v0.3 will apply the explicit Phase 0 kill criteria (`associative_recall_uplift > 2.0%`, zero isolation/deletion leakage, and `<= 1.0%` temporal consistency regression).

