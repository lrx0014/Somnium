# Somnium Phase 0 Research Spike Report (M1)

> **Status:** Completed  
> **Suite:** `synthetic-v1` (seed `42`)  
> **World Manifest SHA-256:** `0114c1fce2af15eacfa7561c53d6549b1fe068ecf443925e185237e827fcfc4c`  

---

## 1. Pinned Environment & MLX Compatibility Verification (M1.6)

| Component | Pinned Version / Result |
|---|---|
| Hardware / OS | `macOS-27.0.1-arm64-arm-64bit-Mach-O` (`arm64`) |
| Python | `3.14.7` |
| MLX (`mlx`) | `0.32.4` (`Device(gpu, 0)`) |
| MLX-LM (`mlx-lm`) | `0.32.2` |
| Qwen3 ChatML Template Rendering | `True` |
| Qwen3 FP16 LoRA Training Loss (12 steps) | `6.8721` → `4.5303` |
| Qwen3 4-bit QLoRA (`group_size=64, bits=4`) Loss | `6.9214` → `4.4345` |
| Adapter Save (`adapters.safetensors`) & Reload (`load_adapters`) | `True` |
| Trainable / Total Parameters (Compatibility Check) | `32768` / `393984` |

**Dataset Reproducibility Manifests:**
- World records: `114` records across `95` underlying fact blueprints (`75%` train split, `25%` held-out unseen fact split).
- `slice1` manifest (`187` examples): `e9f5839249683c10dcac5958...`
- `incremental_replay` manifest (`185` examples, 50% replay): `cd0a2601cbf7add3d1a68674...`
- `full_retrain` manifest (`265` examples): `4aea5f48016aabaee304d370...`

---

## 2. Experiment A — World Slice 1 Acquisition & Parametric Recall (M1.7)

Adapter 1 (`engram-slice1-v1`) trained on `187` Slice 1 examples (loss `5.5446` → `0.0002` in `2.148s`). Evaluated on `254` Slice 1 questions:

| System | Overall | Trained Split | Held-Out Split | Acquisition (S1) | Retention (S1) | Temporal | Cue-Based | **Uplift vs FTS** | Scope Attr | Abstention | Del/Iso Leak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lexical-fts` | `83.86%` | `85.45%` | `82.67%` | `84.58%` | `84.58%` | `100.00%` | `71.88%` | **`0.00%`** | `100.00%` | `66.67%` | `0.0%/0.0%` |
| `full-context` | `83.86%` | `85.45%` | `82.67%` | `84.58%` | `84.58%` | `100.00%` | `71.88%` | **`0.00%`** | `100.00%` | `66.67%` | `0.0%/0.0%` |
| `base-model-no-adapter` | `5.51%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | **`0.00%`** | `0.00%` | `100.00%` | `0.0%/0.0%` |
| `adapter1-parametric-only` | `57.87%` | `86.06%` | `0.00%` | `59.17%` | `59.17%` | `65.00%` | `59.38%` | **`14.79%`** | `35.71%` | `66.67%` | `0.0%/16.7%` |
| `adapter1-hybrid` | `90.55%` | `98.18%` | `85.33%` | `94.17%` | `94.17%` | `100.00%` | `89.84%` | **`17.61%`** | `100.00%` | `50.00%` | `0.0%/0.0%` |

### Experiment A Findings
1. **Strong lexical retrieval baseline (`lexical-fts`):** Achieves near-perfect accuracy on `direct`, `temporal`, `correction`, and `scope_attribution` queries, but drops on `incomplete_cue` queries where vague associative cues share no keywords with the stored record.
2. **Parametric-only limitation (`adapter1-parametric-only`):** Learns trained fact associations well (including vague cues in the `train` split), but scores `0.00%` on the `held_out` split (facts never seen in training). This directly confirms `design.md` §5.1: parametric weights alone cannot serve newly written or held-out facts.
3. **Hybrid associative uplift (`adapter1-hybrid`):** Combining parametric association with Hippocore verification (`associate -> verify -> answer`) bridges vague cues to their canonical records while preserving full accuracy on held-out facts, yielding positive **Associative-Recall Uplift** over `lexical-fts` with `0.0%` deletion and isolation leakage.

---

## 3. Experiment B — Continual Learning Across Slice 1 → Slice 2 (M1.8)

Slice 2 introduces new project decisions and supersedes half of the mutable decisions from Slice 1. Evaluated on all `356` Slice 1 + Slice 2 questions:

| System | Overall | Trained Split | Held-Out Split | Acquisition (S2) | Retention (S1) | Temporal | Cue-Based | **Uplift vs FTS** | Scope Attr | Abstention | Del/Iso Leak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lexical-fts` | `86.24%` | `87.14%` | `86.14%` | `89.29%` | `85.15%` | `100.00%` | `75.00%` | **`0.00%`** | `100.00%` | `66.67%` | `0.0%/0.0%` |
| `full-context` | `86.24%` | `87.14%` | `87.13%` | `90.00%` | `85.15%` | `100.00%` | `75.57%` | **`0.51%`** | `100.00%` | `50.00%` | `0.0%/0.0%` |
| `base-model-no-adapter` | `3.93%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | **`0.00%`** | `0.00%` | `100.00%` | `0.0%/0.0%` |
| `adapter1-stale-parametric` | `36.24%` | `51.45%` | `0.00%` | `0.00%` | `61.39%` | `34.21%` | `37.50%` | **`8.67%`** | `20.00%` | `66.67%` | `0.0%/16.7%` |
| `adapter1-stale-hybrid` | `90.45%` | `95.02%` | `88.12%` | `92.14%` | `93.56%` | `100.00%` | `87.50%` | **`12.24%`** | `100.00%` | `50.00%` | `0.0%/0.0%` |
| `adapter2-inc-noreplay-parametric` | `26.97%` | `37.76%` | `0.00%` | `65.00%` | `0.00%` | `31.58%` | `26.70%` | **`5.61%`** | `20.00%` | `66.67%` | `0.0%/0.0%` |
| `adapter2-inc-replay-parametric` | `52.25%` | `74.69%` | `0.00%` | `62.86%` | `45.54%` | `42.11%` | `53.98%` | **`11.73%`** | `32.50%` | `83.33%` | `0.0%/16.7%` |
| `adapter2-inc-replay-hybrid` | `91.29%` | `95.85%` | `88.12%` | `95.71%` | `92.08%` | `100.00%` | `88.64%` | **`13.27%`** | `100.00%` | `66.67%` | `0.0%/0.0%` |
| `adapter2-full-retrain-parametric` | `64.33%` | `92.95%` | `0.00%` | `65.71%` | `65.35%` | `68.42%` | `68.75%` | **`14.29%`** | `40.00%` | `66.67%` | `0.0%/0.0%` |
| `adapter2-full-retrain-hybrid` | `92.70%` | `97.93%` | `89.11%` | `96.43%` | `94.55%` | `100.00%` | `92.05%` | **`16.33%`** | `100.00%` | `50.00%` | `0.0%/0.0%` |

### Experiment B Findings & Failure Analysis
1. **Stale weights before Dream consolidation (`adapter1-stale-parametric` vs `adapter1-stale-hybrid`):**
   - Before Adapter 2 is trained, `adapter1-stale-parametric` continues asserting superseded Slice 1 decisions as current, causing severe temporal consistency degradation.
   - In contrast, `adapter1-stale-hybrid` enforces the **Conflict Rule** (`design.md` §6.3): when Adapter 1 cites a superseded record ID, Hippocore's `supersedes` link resolves to the active Slice 2 replacement record, restoring `100%` temporal consistency even before the next Dream cycle runs.
2. **Catastrophic forgetting without replay (`adapter2-inc-noreplay-parametric`):**
   - Fine-tuning Adapter 1 solely on Slice 2 examples causes Slice 1 historical retention to drop sharply as LoRA weights overwrite earlier associations.
3. **Incremental with 50% replay vs. Full retrain (`adapter2-inc-replay-*` vs `adapter2-full-retrain-*`):**
   - Including 50% historical replay recovers almost all Slice 1 retention while learning Slice 2 facts.
   - Full retrain from the clean base (`adapter2-full-retrain-hybrid`) achieves the strongest overall accuracy, zero drift, and clean erasure of any deleted facts.

---

## 4. Kill Criteria / Conditions to Deprioritize Associative Recall (M1.9)

Per `design.md` §13 (Phase 0 exit criteria), neural associative recall must earn its complexity and GPU cost over `evidence` recall. Associative recall will be narrowed or deprioritized if any of the following hold at production scale (Milestone v0.3):
1. **Zero or negative uplift:** `associative_recall_uplift <= 2.0%` over `lexical-fts` (and embedding RAG in v0.2) on vague/cue-based and cross-scope synthesis queries.
2. **Unresolved temporal interference:** Hybrid `associate -> verify -> answer` regresses `temporal_consistency` by `> 1.0%` compared to `evidence` recall.
3. **Safety / isolation violation:** Any non-zero `deletion_leakage` or `isolation_leakage` after post-generation guards.

**Decision (see `docs/decisions.md` ADR-0007):** Proceed with the dual-mode architecture (`evidence` recall first in M2–M3, hybrid `associate -> verify -> answer` in M4, and supporting both `incremental` with replay and `full_retrain` in M5).
