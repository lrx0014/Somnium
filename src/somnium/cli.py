"""Command-line interface for Somnium (M0.9).

Initial commands:
- `somnium --version`
- `somnium config show [--profile dev|test|prod] [--format yaml|json]`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from somnium import __version__
from somnium.config import ProfileName, load_settings

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
app.add_typer(config_app, name="config")


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


def main() -> None:
    """Entrypoint for the `somnium` console script."""
    app()


if __name__ == "__main__":
    main()
