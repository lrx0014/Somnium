"""Command-line interface for Somnium (M0.9, M1.4, M3.6).

Commands:
- `somnium --version`
- `somnium config show [--profile dev|test|prod] [--format yaml|json]`
- `somnium eval run --suite synthetic-v1 --system <name> [--seed 42] [--slice 1|2]`
- `somnium serve --stdio [--profile dev|test|prod] [--db-url <url>]`
- `somnium memory store|search|show|list|correct|forget|export|status`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import typer
import yaml

from somnium import __version__
from somnium.config import ProfileName, load_settings
from somnium.domain import (
    DeletionMode,
    EvidenceRecallResponse,
    MemoryAskRequest,
    MemoryCorrectRequest,
    MemoryForgetRequest,
    MemoryKind,
    MemoryStatusRequest,
    MemoryStoreRequest,
    RecallMode,
    SensitivityLevel,
)
from somnium.eval.harness import run_benchmark_cli
from somnium.freud import (
    FreudError,
    build_freud_stack,
    format_evidence_response_dict,
    parse_scope,
    run_stdio_server,
)

app = typer.Typer(
    name="somnium",
    help="Somnium: A bio-inspired, dream-driven neural memory system for AI agents.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(
    name="config",
    help="Inspect and validate Somnium configuration profiles.",
    no_args_is_help=True,
)
eval_app = typer.Typer(
    name="eval",
    help="Run synthetic memory benchmarks and produce JSON / Markdown reports.",
    no_args_is_help=True,
)
memory_app = typer.Typer(
    name="memory",
    help="Store, search, inspect, correct, forget, and export episodic memories.",
    no_args_is_help=True,
)
dream_app = typer.Typer(
    name="dream",
    help="Dream Engine training, evaluation, and consolidation (M5).",
    no_args_is_help=True,
)
model_app = typer.Typer(
    name="model",
    help="Model registry inspection, promotion, and rollback (M5).",
    no_args_is_help=True,
)
dataset_app = typer.Typer(
    name="dataset",
    help="Dataset manifest building and inspection (M5).",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")
app.add_typer(eval_app, name="eval")
app.add_typer(memory_app, name="memory")
app.add_typer(dream_app, name="dream")
app.add_typer(model_app, name="model")
app.add_typer(dataset_app, name="dataset")


def _version_callback(value: bool) -> None:
    """Print the Somnium version and exit when `--version` is passed."""
    if value:
        typer.echo(f"somnium {__version__}")
        raise typer.Exit()


def _echo_structured(payload: Any, output_format: str) -> None:
    """Print `payload` formatted as JSON or YAML."""
    fmt = output_format.strip().lower()
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
    elif fmt == "yaml":
        typer.echo(yaml.safe_dump(payload, sort_keys=False).rstrip())
    else:
        raise typer.BadParameter(
            f"Unsupported output format '{output_format}'. Use 'json' or 'yaml'."
        )


def _handle_freud_error(exc: FreudError, output_format: str = "json") -> None:
    """Format a `FreudError` and exit with non-zero code."""
    err_dict = exc.to_response().model_dump(mode="json")
    _echo_structured(err_dict, output_format)
    raise typer.Exit(code=1)


@app.callback()
def root_callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show the Somnium version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Somnium CLI root command."""


@app.command("serve")
def serve_command(
    stdio: Annotated[
        bool,
        typer.Option(
            "--stdio",
            help="Run the Freud MCP gateway over stdio transport.",
        ),
    ] = True,
    profile: Annotated[
        ProfileName | None,
        typer.Option(
            "--profile",
            "-p",
            help="Configuration profile to load (dev, test, or prod).",
        ),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Explicit path to a YAML configuration file.",
        ),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option(
            "--db-url",
            help="Optional SQLAlchemy database URL override.",
        ),
    ] = None,
) -> None:
    """Launch the Freud MCP gateway server (`somnium serve --stdio`, M3.6)."""
    if not stdio:
        raise typer.BadParameter(
            "Only `--stdio` transport is supported in v0.1 MVP "
            "(network transport is planned for v0.4)."
        )
    settings = load_settings(profile=profile, config_path=config_file)
    run_stdio_server(settings, db_url_override=db_url)


@config_app.command("show")
def config_show(
    profile: Annotated[
        ProfileName | None,
        typer.Option(
            "--profile",
            "-p",
            help="Configuration profile to load (dev, test, or prod).",
        ),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Explicit path to a YAML configuration file.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: 'yaml' or 'json'.",
        ),
    ] = "yaml",
) -> None:
    """Display the resolved configuration with all secrets redacted."""
    settings = load_settings(profile=profile, config_path=config_file)
    redacted = settings.redacted_dict()
    _echo_structured(redacted, output_format)


@eval_app.command("run")
def eval_run(
    suite: Annotated[
        str,
        typer.Option(
            "--suite",
            "-s",
            help="Benchmark suite identifier (e.g., 'synthetic-v1').",
        ),
    ] = "synthetic-v1",
    system: Annotated[
        str,
        typer.Option(
            "--system",
            help=(
                "System under test ('lexical-fts', 'hippocore-fts', 'full-context', "
                "'base-model-no-adapter')."
            ),
        ),
    ] = "lexical-fts",
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Deterministic world generation seed.",
        ),
    ] = 42,
    max_slice: Annotated[
        int,
        typer.Option(
            "--slice",
            help="Maximum temporal slice to include (1 or 2).",
        ),
    ] = 2,
    output_json: Annotated[
        Path | None,
        typer.Option(
            "--output-json",
            help="Optional file path to write JSON benchmark results.",
        ),
    ] = None,
    output_md: Annotated[
        Path | None,
        typer.Option(
            "--output-md",
            help="Optional file path to write Markdown benchmark report.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Stdout output format: 'markdown' or 'json'.",
        ),
    ] = "markdown",
) -> None:
    """Run a Somnium benchmark suite against a SystemUnderTest."""
    try:
        result = run_benchmark_cli(
            suite_name=suite,
            system_name=system,
            seed=seed,
            max_slice=max_slice,
            output_json=output_json,
            output_md=output_md,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    fmt = output_format.strip().lower()
    if fmt == "json":
        typer.echo(result.to_json())
    elif fmt in ("markdown", "md"):
        typer.echo(result.to_markdown().rstrip())
    else:
        raise typer.BadParameter(
            f"Unsupported output format '{output_format}'. Use 'markdown' or 'json'."
        )


@memory_app.command("store")
def memory_store_cmd(
    content: Annotated[
        str,
        typer.Argument(help="Memory statement to store."),
    ],
    scope: Annotated[
        str | None,
        typer.Option("--scope", "-s", help="Target scope (e.g. 'project-orion')."),
    ] = None,
    kind: Annotated[
        MemoryKind,
        typer.Option("--kind", "-k", help="Memory record kind."),
    ] = MemoryKind.FACT,
    sensitivity: Annotated[
        SensitivityLevel,
        typer.Option("--sensitivity", help="Sensitivity level ('normal' or 'high')."),
    ] = SensitivityLevel.NORMAL,
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", "-t", help="Optional tag(s) to attach."),
    ] = None,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Store a new episodic memory record in Hippocore."""
    settings = load_settings(profile=profile)
    _, service, auth_provider, _ = build_freud_stack(settings, db_url_override=db_url)
    try:
        resolved_scope = parse_scope(scope, default_scope=service.default_scope)
        auth = auth_provider.authenticate(requested_scopes=(resolved_scope,))
        resp = service.store(
            auth,
            MemoryStoreRequest(
                content=content,
                scope=resolved_scope,
                kind=kind,
                sensitivity=sensitivity,
                tags=tuple(tag or ()),
            ),
        )
        _echo_structured(resp.model_dump(mode="json"), output_format)
    except FreudError as exc:
        _handle_freud_error(exc, output_format)


@memory_app.command("search")
def memory_search_cmd(
    query: Annotated[
        str,
        typer.Argument(help="Search question or keywords."),
    ],
    scope: Annotated[
        str | None,
        typer.Option("--scope", "-s", help="Target scope (e.g. 'project-orion')."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of records to return."),
    ] = 10,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Search memories using evidence recall (`somnium memory search`)."""
    settings = load_settings(profile=profile)
    _, service, auth_provider, _ = build_freud_stack(settings, db_url_override=db_url)
    try:
        resolved_scope = (
            parse_scope(scope, default_scope=service.default_scope) if scope is not None else None
        )
        req_scopes = (resolved_scope,) if resolved_scope is not None else None
        auth = auth_provider.authenticate(requested_scopes=req_scopes)
        resp = service.ask(
            auth,
            MemoryAskRequest(
                question=query,
                scope=resolved_scope,
                mode=RecallMode.EVIDENCE,
                limit=limit,
            ),
        )
        if isinstance(resp, EvidenceRecallResponse):
            _echo_structured(format_evidence_response_dict(resp), output_format)
        else:
            _echo_structured(resp.model_dump(mode="json"), output_format)
    except FreudError as exc:
        _handle_freud_error(exc, output_format)


@memory_app.command("show")
def memory_show_cmd(
    memory_id: Annotated[
        str,
        typer.Argument(help="UUID of the memory record to inspect."),
    ],
    scope: Annotated[
        str | None,
        typer.Option("--scope", "-s", help="Scope context for authorization."),
    ] = None,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Show a single memory record by its UUID (`somnium memory show`)."""
    settings = load_settings(profile=profile)
    _, service, auth_provider, _ = build_freud_stack(settings, db_url_override=db_url)
    try:
        uid = UUID(memory_id.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid memory UUID '{memory_id}'.") from exc

    try:
        resolved_scope = (
            parse_scope(scope, default_scope=service.default_scope) if scope is not None else None
        )
        req_scopes = (resolved_scope,) if resolved_scope is not None else None
        auth = auth_provider.authenticate(requested_scopes=req_scopes)
        rec = service.get(auth, uid)
        _echo_structured(rec.model_dump(mode="json"), output_format)
    except FreudError as exc:
        _handle_freud_error(exc, output_format)


@memory_app.command("list")
def memory_list_cmd(
    scope: Annotated[
        str | None,
        typer.Option("--scope", "-s", help="Scope to list (defaults to configured scope)."),
    ] = None,
    include_superseded: Annotated[
        bool,
        typer.Option("--include-superseded", help="Include superseded historical records."),
    ] = False,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """List memory records within a scope (`somnium memory list`)."""
    settings = load_settings(profile=profile)
    _, service, auth_provider, _ = build_freud_stack(settings, db_url_override=db_url)
    try:
        resolved_scope = parse_scope(scope, default_scope=service.default_scope)
        auth = auth_provider.authenticate(requested_scopes=(resolved_scope,))
        records = service.list_by_scope(
            auth,
            resolved_scope,
            include_superseded=include_superseded,
        )
        payload = {
            "scope": str(resolved_scope),
            "count": len(records),
            "records": [r.model_dump(mode="json") for r in records],
        }
        _echo_structured(payload, output_format)
    except FreudError as exc:
        _handle_freud_error(exc, output_format)


@memory_app.command("correct")
def memory_correct_cmd(
    memory_id: Annotated[
        str,
        typer.Argument(help="UUID of the memory record to supersede."),
    ],
    replacement: Annotated[
        str,
        typer.Argument(help="Replacement memory statement."),
    ],
    reason: Annotated[
        str | None,
        typer.Option("--reason", "-r", help="Optional reason for the correction."),
    ] = None,
    scope: Annotated[
        str | None,
        typer.Option("--scope", "-s", help="Scope context for authorization."),
    ] = None,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Correct an existing memory record (`somnium memory correct`)."""
    settings = load_settings(profile=profile)
    _, service, auth_provider, _ = build_freud_stack(settings, db_url_override=db_url)
    try:
        uid = UUID(memory_id.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid memory UUID '{memory_id}'.") from exc

    try:
        resolved_scope = (
            parse_scope(scope, default_scope=service.default_scope) if scope is not None else None
        )
        req_scopes = (resolved_scope,) if resolved_scope is not None else None
        auth = auth_provider.authenticate(requested_scopes=req_scopes)
        resp = service.correct(
            auth,
            MemoryCorrectRequest(
                memory_id=uid,
                replacement=replacement,
                reason=reason,
            ),
        )
        _echo_structured(resp.model_dump(mode="json"), output_format)
    except FreudError as exc:
        _handle_freud_error(exc, output_format)


@memory_app.command("forget")
def memory_forget_cmd(
    memory_id: Annotated[
        str,
        typer.Argument(help="UUID of the memory record to delete."),
    ],
    mode: Annotated[
        DeletionMode,
        typer.Option("--mode", "-m", help="Deletion mode ('hard' or 'soft')."),
    ] = DeletionMode.HARD,
    reason: Annotated[
        str | None,
        typer.Option("--reason", "-r", help="Optional reason for deletion."),
    ] = None,
    scope: Annotated[
        str | None,
        typer.Option("--scope", "-s", help="Scope context for authorization."),
    ] = None,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Delete a memory record from Hippocore (`somnium memory forget`)."""
    settings = load_settings(profile=profile)
    _, service, auth_provider, _ = build_freud_stack(settings, db_url_override=db_url)
    try:
        uid = UUID(memory_id.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid memory UUID '{memory_id}'.") from exc

    try:
        resolved_scope = (
            parse_scope(scope, default_scope=service.default_scope) if scope is not None else None
        )
        req_scopes = (resolved_scope,) if resolved_scope is not None else None
        auth = auth_provider.authenticate(requested_scopes=req_scopes)
        resp = service.forget(
            auth,
            MemoryForgetRequest(
                memory_id=uid,
                deletion_mode=mode,
                reason=reason,
            ),
        )
        _echo_structured(resp.model_dump(mode="json"), output_format)
    except FreudError as exc:
        _handle_freud_error(exc, output_format)


@memory_app.command("export")
def memory_export_cmd(
    scope: Annotated[
        str | None,
        typer.Option("--scope", "-s", help="Optional scope filter."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional file path to write JSONL export."),
    ] = None,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
) -> None:
    """Export authorized non-deleted facts and revisions as JSONL (`somnium memory export`)."""
    settings = load_settings(profile=profile)
    _, service, auth_provider, _ = build_freud_stack(settings, db_url_override=db_url)
    try:
        resolved_scope = (
            parse_scope(scope, default_scope=service.default_scope) if scope is not None else None
        )
        req_scopes = (resolved_scope,) if resolved_scope is not None else None
        auth = auth_provider.authenticate(requested_scopes=req_scopes)
        jsonl_text = service.export_jsonl(auth)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(jsonl_text, encoding="utf-8")
        else:
            typer.echo(jsonl_text.rstrip())
    except FreudError as exc:
        _handle_freud_error(exc, "json")


@memory_app.command("status")
def memory_status_cmd(
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Display current Dream Engine state and memory status (`somnium memory status`)."""
    settings = load_settings(profile=profile)
    _, service, auth_provider, _ = build_freud_stack(settings, db_url_override=db_url)
    try:
        auth = auth_provider.authenticate()
        resp = service.status(auth, MemoryStatusRequest())
        _echo_structured(resp.model_dump(mode="json"), output_format)
    except FreudError as exc:
        _handle_freud_error(exc, output_format)


def _get_dream_stack(
    settings: Any,
    db_url_override: str | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Helper to assemble Hippocore, ArtifactStore, ModelRegistry, and Backends."""
    from somnium.dream import (
        LocalFilesystemArtifactStore,
        ModelRegistry,
    )

    store, service, auth_provider, _ = build_freud_stack(settings, db_url_override=db_url_override)
    artifact_store = LocalFilesystemArtifactStore(settings.storage.artifacts_dir)
    registry = ModelRegistry(store.engine)
    inference_backend = (
        service.engram_pipeline.inference_backend if service.engram_pipeline else None
    )

    if settings.training.backend == "mlx":
        from somnium.backends.mlx import MLXTrainingBackend

        training_backend = MLXTrainingBackend(artifact_store)
    else:
        from somnium.backends.fake import FakeTrainingBackend

        training_backend = FakeTrainingBackend()

    return store, artifact_store, registry, training_backend, inference_backend, auth_provider


@dream_app.command("run")
def dream_run_cmd(
    mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="Training mode: 'incremental' or 'full'."),
    ] = "incremental",
    auto_promote: Annotated[
        bool,
        typer.Option(
            "--auto-promote/--no-auto-promote",
            help="Automatically promote candidate adapter if promotion gate passes.",
        ),
    ] = True,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Execute an end-to-end Dream cycle (`somnium dream run --mode incremental|full`)."""
    from somnium.dream import execute_dream_cycle

    settings = load_settings(profile=profile)
    (
        store,
        artifact_store,
        registry,
        training_backend,
        inference_backend,
        auth_provider,
    ) = _get_dream_stack(settings, db_url_override=db_url)

    auth = auth_provider.authenticate()
    result = execute_dream_cycle(
        auth,
        store=store,
        artifact_store=artifact_store,
        training_backend=training_backend,
        inference_backend=inference_backend,
        model_registry=registry,
        requested_mode=mode,
        auto_promote=auto_promote,
        base_model_id=settings.inference.base_model_id,
        base_model_revision=settings.inference.base_model_revision,
        epochs=settings.training.epochs,
        learning_rate=settings.training.learning_rate,
        lora_rank=settings.training.lora_rank,
    )

    summary = {
        "run_id": result.run_record.run_id,
        "tenant_id": result.run_record.tenant_id,
        "requested_mode": result.run_record.requested_mode,
        "resolved_mode": result.run_record.resolved_mode,
        "forced_full_retrain_reason": result.run_record.forced_full_retrain_reason,
        "dataset_hash": result.dataset_manifest.dataset_hash,
        "candidate_model_version": result.candidate_version_record.model_version,
        "candidate_status": result.candidate_version_record.status,
        "gate_passed": result.gate_report.passed,
        "promoted": result.promoted,
        "active_model_version": registry.get_active_version(auth.tenant_id),
        "failures": list(result.gate_report.failures),
    }
    _echo_structured(summary, output_format)


@model_app.command("list")
def model_list_cmd(
    status: Annotated[
        str | None,
        typer.Option("--status", "-s", help="Filter by status ('candidate', 'promoted', etc.)."),
    ] = None,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """List model versions registered for the tenant (`somnium model list`)."""
    settings = load_settings(profile=profile)
    _, _, registry, _, _, auth_provider = _get_dream_stack(settings, db_url_override=db_url)
    auth = auth_provider.authenticate()

    versions = registry.list_versions(auth.tenant_id, status=status)
    active = registry.get_active_version(auth.tenant_id)
    output = [
        {
            "model_version": v.model_version,
            "status": v.status,
            "is_active": (v.model_version == active),
            "training_mode": v.training_mode,
            "parent_version": v.parent_version,
            "dataset_manifest_hash": v.dataset_manifest_hash,
            "created_at": v.created_at,
            "promoted_at": v.promoted_at,
        }
        for v in versions
    ]
    _echo_structured(output, output_format)


@model_app.command("show")
def model_show_cmd(
    model_version: Annotated[str, typer.Argument(help="Model version to inspect.")],
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Show details of a specific model version (`somnium model show <version>`)."""
    settings = load_settings(profile=profile)
    _, _, registry, _, _, auth_provider = _get_dream_stack(settings, db_url_override=db_url)
    auth = auth_provider.authenticate()

    rec = registry.get_version(model_version)
    if rec is None or rec.tenant_id != auth.tenant_id:
        _echo_structured({"error": f"Model version '{model_version}' not found."}, output_format)
        raise typer.Exit(code=1)

    active = registry.get_active_version(auth.tenant_id)
    payload = {
        "model_version": rec.model_version,
        "tenant_id": rec.tenant_id,
        "status": rec.status,
        "is_active": (rec.model_version == active),
        "base_model_id": rec.base_model_id,
        "base_model_revision": rec.base_model_revision,
        "parent_version": rec.parent_version,
        "training_mode": rec.training_mode,
        "dataset_manifest_hash": rec.dataset_manifest_hash,
        "checkpoint_uri": rec.checkpoint_uri,
        "metrics": rec.metrics,
        "gate_report": rec.gate_report,
        "created_at": rec.created_at,
        "promoted_at": rec.promoted_at,
    }
    _echo_structured(payload, output_format)


@model_app.command("promote")
def model_promote_cmd(
    model_version: Annotated[str, typer.Argument(help="Model version to promote.")],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force promotion even if status was rejected."),
    ] = False,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Promote a candidate model version to active (`somnium model promote <version>`)."""
    settings = load_settings(profile=profile)
    (
        _,
        _,
        registry,
        _,
        inference_backend,
        auth_provider,
    ) = _get_dream_stack(settings, db_url_override=db_url)
    auth = auth_provider.authenticate()

    try:
        res = registry.promote(
            tenant_id=auth.tenant_id,
            candidate_version=model_version,
            inference_backend=inference_backend,
            force=force,
        )
        _echo_structured(
            {
                "status": "promoted",
                "promoted_version": res.promoted_version,
                "previous_version": res.previous_version,
                "promoted_at": res.promoted_at,
            },
            output_format,
        )
    except Exception as exc:
        _echo_structured({"error": str(exc)}, output_format)
        raise typer.Exit(code=1) from None


@model_app.command("rollback")
def model_rollback_cmd(
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Target model version (or previous if omitted)."),
    ] = None,
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Roll back tenant active model version (`somnium model rollback`)."""
    settings = load_settings(profile=profile)
    (
        _,
        _,
        registry,
        _,
        inference_backend,
        auth_provider,
    ) = _get_dream_stack(settings, db_url_override=db_url)
    auth = auth_provider.authenticate()

    try:
        res = registry.rollback(
            tenant_id=auth.tenant_id,
            target_version=target,
            inference_backend=inference_backend,
        )
        _echo_structured(
            {
                "status": "rolled_back",
                "active_version": res.active_version,
                "rolled_back_version": res.rolled_back_version,
                "rolled_back_at": res.rolled_back_at,
            },
            output_format,
        )
    except Exception as exc:
        _echo_structured({"error": str(exc)}, output_format)
        raise typer.Exit(code=1) from None


@dataset_app.command("build")
def dataset_build_cmd(
    mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="Training mode: 'incremental' or 'full'."),
    ] = "incremental",
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Build a training dataset manifest (`somnium dataset build --mode incremental|full`)."""
    from somnium.dream import build_tenant_dataset

    settings = load_settings(profile=profile)
    store, artifact_store, registry, _, _, auth_provider = _get_dream_stack(
        settings, db_url_override=db_url
    )
    auth = auth_provider.authenticate()

    active_version = registry.get_active_version(auth.tenant_id)
    parent_manifest_hash: str | None = None
    if active_version:
        active_rec = registry.get_version(active_version)
        if active_rec:
            parent_manifest_hash = active_rec.dataset_manifest_hash

    manifest = build_tenant_dataset(
        auth,
        store=store,
        artifact_store=artifact_store,
        training_mode=mode,
        parent_manifest_hash=parent_manifest_hash,
    )

    payload = {
        "dataset_hash": manifest.dataset_hash,
        "tenant_id": manifest.tenant_id,
        "training_mode": manifest.training_mode,
        "generator_version": manifest.generator_version,
        "parent_manifest_hash": manifest.parent_manifest_hash,
        "source_records_count": len(manifest.source_record_ids),
        "total_examples_count": len(manifest.examples),
        "example_counts": manifest.example_counts,
        "artifact_uri": manifest.artifact_uri,
        "created_at": manifest.created_at,
    }
    _echo_structured(payload, output_format)


@dataset_app.command("show")
def dataset_show_cmd(
    dataset_hash: Annotated[str, typer.Argument(help="SHA-256 hash of the dataset manifest.")],
    profile: Annotated[
        ProfileName | None,
        typer.Option("--profile", "-p", help="Configuration profile."),
    ] = None,
    db_url: Annotated[
        str | None,
        typer.Option("--db-url", help="Database URL override."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'yaml'."),
    ] = "json",
) -> None:
    """Show details of a dataset manifest (`somnium dataset show <hash>`)."""
    from somnium.dream import load_dataset_manifest

    settings = load_settings(profile=profile)
    store, artifact_store, _, _, _, _ = _get_dream_stack(settings, db_url_override=db_url)

    try:
        manifest = load_dataset_manifest(store.engine, artifact_store, dataset_hash)
        payload = {
            "dataset_hash": manifest.dataset_hash,
            "tenant_id": manifest.tenant_id,
            "training_mode": manifest.training_mode,
            "generator_version": manifest.generator_version,
            "parent_manifest_hash": manifest.parent_manifest_hash,
            "source_records_count": len(manifest.source_record_ids),
            "source_record_ids": list(manifest.source_record_ids),
            "total_examples_count": len(manifest.examples),
            "example_counts": manifest.example_counts,
            "artifact_uri": manifest.artifact_uri,
            "created_at": manifest.created_at,
        }
        _echo_structured(payload, output_format)
    except Exception as exc:
        _echo_structured({"error": str(exc)}, output_format)
        raise typer.Exit(code=1) from None


def main() -> None:
    """Entrypoint for the `somnium` console script."""
    app()


if __name__ == "__main__":
    main()
