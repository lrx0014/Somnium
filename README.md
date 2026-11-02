# Somnium

> **Remember while awake. Learn while dreaming.**  
> A bio-inspired, dream-driven neural memory system for AI agents.

Somnium is an independent, persistent memory service for AI agents (initially integrated with OpenAI Codex via MCP). It combines **Hippocore** (an authoritative, project-isolated episodic store for immediate **evidence recall**) with **Engram** (a dedicated, continually trained memory model that consolidates memories during idle **Dream** cycles and serves verified **associative recall**).

- System architecture & specification: [`design.md`](design.md)
- Development roadmap & milestones: [`Milestones.md`](Milestones.md)
- Architecture decision records: [`docs/decisions.md`](docs/decisions.md)

## Quickstart (Development)

```bash
# Install core + dev dependencies (no ML backend required for core tests)
uv sync --extra dev

# Inspect version and redacted configuration profiles
uv run somnium --version
uv run somnium config show --profile dev

# Run lint, type checks, import-boundary checks, and unit tests
uv run ruff check src tests
uv run pyright
uv run pytest
```
