# Somnium

> **Remember while awake. Learn while dreaming.**  
> A bio-inspired, dream-driven neural memory system for AI agents.

Somnium is an independent, persistent memory service for AI agents (integrating with OpenAI Codex, Claude, Cursor, and other agent platforms via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io)).

It decouples immediate memory availability from asynchronous parameter learning:
- **Hippocore (Episodic Store):** Authoritative SQLite/FTS5 store providing strictly scope-isolated, sub-millisecond **evidence recall** for raw events, facts, revisions, and decisions immediately as they happen.
- **Engram (Associative Memory):** Dedicated model (e.g., Qwen3 with tenant-specific LoRA adapters) executing structured `associate -> verify -> answer` recall with deterministic evidence grounding guards.
- **Dream Engine (Consolidation Loop):** Automated background worker that detects agent inactivity, checks resource policies (AC power, memory pressure), builds reproducible replay datasets, fine-tunes candidate adapters via MLX on Apple Silicon, evaluates safety gates (0% leakage), and atomically promotes new model versions.

---

## Architecture Overview

```
                      +-----------------------------+
                      |         AI Agent            |
                      |   (Codex / Claude / IDE)    |
                      +--------------+--------------+
                                     | MCP (stdio)
                                     v
                      +-----------------------------+
                      |     Freud Gateway Server    |
                      |   (Scope Authorization &    |
                      |     Recall Auto-Router)     |
                      +--------------+--------------+
                                     |
              +----------------------+----------------------+
              |                                             |
              v                                             v
+---------------------------+                 +---------------------------+
|         Hippocore         |                 |          Engram           |
|      (Episodic Store)     | <============== |    (Associative Memory)   |
|   Strict Scope Isolation  |  Evidence-Grounded |   3-Stage Recall:        |
|   FTS5 Lexical Retrieval  |  Claim Verification|   Associate -> Verify    |
|   Immediate Availability  |                 |         -> Answer         |
+-------------+-------------+                 +-------------^-------------+
              |                                             |
              | Training-Eligible                           | Atomic Adapter
              | Records                                     | Promotion / Swap
              v                                             |
+-----------------------------------------------------------+-------------+
|                               Dream Engine                              |
|   Idle Detection  -->  Dataset Replay  -->  MLX LoRA  -->  Safety Gate  |
|     (Inactivity)       (Augmentations)      (Metal GPU)    (0% Leakage) |
+-------------------------------------------------------------------------+
```

---

## Quickstart

### 1. Requirements

- macOS on Apple Silicon (M1/M2/M3/M4) recommended for MLX GPU acceleration.
- Python 3.12 or 3.14+ (managed via `uv` or `venv`).

### 2. Installation

```bash
# Clone repository
git clone https://github.com/your-org/Somnium.git
cd Somnium

# Install dependencies (core + dev + mlx)
uv sync --extra dev --extra mlx
```

### 3. Verification

Run the test suite, linter, type checker, and architectural boundary rules:

```bash
uv run pytest
uv run ruff check .
uv run pyright
uv run lint-imports
```

---

## 🚀 Running the v0.1 MVP (Step-by-Step Guide)

You can experience the complete Somnium loop right from your terminal or connect it to an AI agent via MCP.

### Option A: Hands-on CLI Walkthrough

Here is a 5-minute tour showcasing how Somnium stores, retrieves, consolidates, and safely rolls back memories.

#### Step 1: Store Episodic Memories Across Scopes
Store architectural decisions for two isolated projects (`project-orion` and `project-lyra`):

```bash
# Store decision for Project Orion
uv run somnium memory store "Project Orion standardized on PostgreSQL 16 for primary storage." \
  --scope project:project-orion --profile dev

# Store decision for Project Lyra
uv run somnium memory store "Project Lyra selected DuckDB for local embedded analytics." \
  --scope project:project-lyra --profile dev
```

#### Step 2: Instant Evidence Recall (Sub-millisecond & Scope-Isolated)
Recall the newly stored memories immediately without waiting for model training:

```bash
# Search within Project Orion
uv run somnium memory search "storage" --scope project:project-orion --profile dev

# Verify scope isolation (searching in Orion never leaks Lyra's data)
uv run somnium memory search "DuckDB" --scope project:project-orion --profile dev
```

#### Step 3: Check Memory & Dream Engine Status
Inspect pending training records, active model version, and lifecycle state:

```bash
uv run somnium memory status --profile dev
```

Output:
```json
{
  "api_version": "v1",
  "tenant_id": "local-user",
  "dream_state": "awake",
  "active_model_version": "Qwen/Qwen3-1.7B@main",
  "pending_training_examples": 2,
  "weights_dirty": false
}
```

#### Step 4: Run the Dream Consolidation Loop
Execute an end-to-end Dream cycle. Somnium builds a content-addressed dataset, fine-tunes a LoRA adapter on your Metal GPU via MLX, evaluates promotion safety gates (0% isolation/deletion leakage), and promotes the adapter:

```bash
uv run somnium dream run --mode full --profile dev
```

Output:
```json
{
  "run_id": "dream-52637939eb3e",
  "tenant_id": "local-user",
  "requested_mode": "full",
  "resolved_mode": "full",
  "dataset_hash": "ab8a733e73bc810033f552ecbda0a3cb1bddf1707eebcc790200a1be10cbc5cd",
  "candidate_model_version": "engram-local-user-v65beb3b5",
  "gate_passed": true,
  "promoted": true,
  "active_model_version": "engram-local-user-v65beb3b5"
}
```

#### Step 5: Inspect Registered Models & Lineage
View all trained adapters and detailed metrics stored in the Model Registry:

```bash
# List all versions and see which one is active
uv run somnium model list --profile dev

# Inspect candidate metrics and gate report
uv run somnium model show engram-local-user-v65beb3b5 --profile dev
```

#### Step 6: Test Deletion & Automatic Full-Retrain Trigger (M5.8)
Delete a memory. Somnium purges the record from Hippocore and automatically flags `weights_dirty = true` to force the next training run to be a clean full retrain:

```bash
# Delete a memory record by ID
uv run somnium memory forget <memory_id> --profile dev

# Check status: weights_dirty is now true
uv run somnium memory status --profile dev

# Next dream run automatically escalates to a clean full retrain
uv run somnium dream run --mode incremental --profile dev
```

#### Step 7: Atomic Rollback (Zero Downtime)
Roll back to the clean base model or previous adapter checkpoint:

```bash
uv run somnium model rollback --profile dev
```

---

### Option B: Running as an MCP Service with Codex / Claude / Cursor

Somnium implements the official [Model Context Protocol (MCP)](https://modelcontextprotocol.io), allowing any MCP-compliant agent to store and recall memories natively.

#### 1. Start the Freud MCP Server Directly

To launch the server over `stdio` transport with automated background dreaming:

```bash
uv run somnium serve --stdio --profile dev
```

#### 2. Configure Your Agent Client

Add Somnium to your MCP client configuration:

**For OpenAI Codex (`~/.codex/config.json`):**
```json
{
  "mcpServers": {
    "somnium": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/Somnium",
        "run",
        "somnium",
        "serve",
        "--stdio",
        "--profile",
        "dev"
      ]
    }
  }
}
```

**For Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "somnium": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/Somnium",
        "run",
        "somnium",
        "serve",
        "--stdio",
        "--profile",
        "dev"
      ]
    }
  }
}
```

#### 3. Agent Usage Prompts

Once connected, your agent will automatically see and invoke Somnium's tools:

- *"Remember that we chose Redis 7.2 as our caching tier in Project Orion."*  
  $\to$ Agent calls `memory_store(scope="project:project-orion", content="...")`.
- *"What database is Project Orion using, and why?"*  
  $\to$ Agent calls `memory_ask(scope="project:project-orion", question="...", mode="auto")`.
- *"What is our memory consolidation status?"*  
  $\to$ Agent calls `memory_status()`.

---

### Option C: Running the Synthetic Benchmark Suite

Somnium includes a synthetic evaluation world and scoring harness to benchmark evidence recall, associative recall uplift, and isolation gates:

```bash
# Run baseline evaluation on Slice 1 without an adapter
uv run somnium eval run --suite synthetic-v1 --system base-no-adapter --slice 1

# Run lexical FTS evidence recall baseline
uv run somnium eval run --suite synthetic-v1 --system fts-baseline --slice 1
```

---

## CLI Command Cheat Sheet

| Command | Description |
|---|---|
| `somnium serve --stdio` | Launch the Freud MCP gateway server over stdio transport |
| `somnium memory store <text>` | Persist a new memory event within an authorized scope |
| `somnium memory search <query>` | Recall memories with evidence citations |
| `somnium memory correct <id> <text>` | Record an immutable revision superseding an earlier record |
| `somnium memory forget <id>` | Delete memory and mark weights dirty if training-eligible |
| `somnium memory status` | Display Dream Engine state, active model, and pending records |
| `somnium dream run` | Trigger dataset build $\to$ LoRA training $\to$ gate $\to$ promotion |
| `somnium model list` | List model versions and active pointer for current tenant |
| `somnium model show <version>` | View detailed metrics, gate reports, and lineage |
| `somnium model promote <version>` | Atomically switch active adapter for a tenant |
| `somnium model rollback` | Restore previous adapter checkpoint or base model |
| `somnium dataset build` | Construct backend-neutral JSONL dataset manifest |
| `somnium dataset show <hash>` | Inspect dataset manifest and example counts |
| `somnium eval run` | Run synthetic benchmark evaluation and generate report |

---

## Dream Engine State Machine

The Dream Engine runs per tenant when external agent traffic is idle:

```
[AWAKE] 
   | Inactivity threshold reached (e.g. 30 min idle)
   v
[IDLE_ELIGIBLE] 
   | Sufficient new training examples + AC power & low memory pressure
   v
[DREAM_PREPARING] 
   | Dataset manifest validated and hashed
   v
[TRAINING]  <--- Pauses / defers to AWAKE if associative traffic arrives
   | Candidate LoRA adapter fine-tuned via MLX AdamW
   v
[EVALUATING] 
   | Acceptance gate check (0% isolation leakage, 0% deletion leakage, 0% unverified claims)
   +---> Passed: [PROMOTING] ---> [AWAKE] (Atomic swap to new adapter)
   +---> Failed: [REJECTED]  ---> [AWAKE] (Retain previous adapter)
```

---

## Known Limitations in v0.1 MVP

The v0.1 MVP focuses on single-user, local agent integration on Apple Silicon. The following are intentionally planned for subsequent releases (v0.2–v1.0):
- **Remote Network Transports:** v0.1 uses MCP stdio transport; network Streamable HTTP with TLS and token auth is slated for v0.4.
- **Server Backends (CUDA):** v0.1 uses MLX on macOS; PyTorch/PEFT and vLLM multi-LoRA for Linux GPUs are slated for v0.3.
- **Vector Embeddings:** v0.1 uses SQLite FTS5 lexical retrieval and parametric neural associative recall; embedding retrieval will be benchmarked in v0.2.
- **Codex Session Auto-Ingestion:** In v0.1, memory ingestion is explicit via tools/events; automatic conversation log parsing is opt-in for future versions.
