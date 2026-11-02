"""Unit tests for the Somnium CLI skeleton (M0.9)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from somnium import __version__
from somnium.cli import app
from somnium.config import REDACTED_PLACEHOLDER

runner = CliRunner()


def test_cli_version() -> None:
    """Verify `somnium --version` outputs the package version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"somnium {__version__}" in result.stdout


def test_cli_config_show_redacts_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `somnium config show` outputs profile config with secrets redacted."""
    monkeypatch.setenv("SOMNIUM_SECURITY__API_TOKEN", "do-not-print-this-secret")

    yaml_res = runner.invoke(app, ["config", "show", "--profile", "dev", "--format", "yaml"])
    assert yaml_res.exit_code == 0
    assert "profile: dev" in yaml_res.stdout
    assert "do-not-print-this-secret" not in yaml_res.stdout
    assert REDACTED_PLACEHOLDER in yaml_res.stdout

    json_res = runner.invoke(app, ["config", "show", "--profile", "test", "--format", "json"])
    assert json_res.exit_code == 0
    parsed = json.loads(json_res.stdout)
    assert parsed["profile"] == "test"
    assert parsed["inference"]["backend"] == "fake"
    assert parsed["security"]["api_token"] == REDACTED_PLACEHOLDER
