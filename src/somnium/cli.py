"""Command-line interface for Somnium (M0.9, M1.4).

Commands:
- `somnium --version`
- `somnium config show [--profile dev|test|prod] [--format yaml|json]`
- `somnium eval run --suite synthetic-v1 --system <name> [--seed 42] [--slice 1|2]`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from somnium import __version__
from somnium.config import ProfileName, load_settings
from somnium.eval.harness import run_benchmark_cli

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
app.add_typer(config_app, name="config")
app.add_typer(eval_app, name="eval")


def _version_callback(value: bool) -> None:
    """Print the Somnium version and exit when `--version` is passed."""
    if value:
        typer.echo(f"somnium {__version__}")
        raise typer.Exit()


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
    fmt = output_format.strip().lower()
    if fmt == "json":
        typer.echo(json.dumps(redacted, indent=2))
    elif fmt == "yaml":
        typer.echo(yaml.safe_dump(redacted, sort_keys=False).rstrip())
    else:
        raise typer.BadParameter(
            f"Unsupported output format '{output_format}'. Use 'yaml' or 'json'."
        )


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
            help="System under test ('lexical-fts', 'full-context', 'base-model-no-adapter').",
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


def main() -> None:
    """Entrypoint for the `somnium` console script."""
    app()


if __name__ == "__main__":
    main()
