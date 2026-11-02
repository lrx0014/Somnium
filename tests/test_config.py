"""Unit tests for typed configuration loading, overrides, and secret redaction (M0.4)."""

from __future__ import annotations

import pytest

from somnium.config import REDACTED_PLACEHOLDER, ProfileName, load_settings


@pytest.mark.parametrize(
    ("profile", "expected_backend", "expected_transport"),
    [
        (ProfileName.DEV, "mlx", "stdio"),
        (ProfileName.TEST, "fake", "stdio"),
        (ProfileName.PROD, "cuda", "streamable-http"),
    ],
)
def test_load_builtin_profiles(
    profile: ProfileName,
    expected_backend: str,
    expected_transport: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify dev, test, and prod YAML profiles load with expected settings."""
    monkeypatch.delenv("SOMNIUM_PROFILE", raising=False)
    settings = load_settings(profile=profile)
    assert settings.profile == profile
    assert settings.inference.backend == expected_backend
    assert settings.training.backend == expected_backend
    assert settings.gateway.transport == expected_transport


def test_environment_variable_overrides_and_secret_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SOMNIUM_* environment variables override YAML values and secrets are redacted."""
    monkeypatch.setenv("SOMNIUM_PROFILE", "prod")
    monkeypatch.setenv("SOMNIUM_INFERENCE__BASE_MODEL_ID", "Qwen/Qwen3-14B")
    monkeypatch.setenv("SOMNIUM_SECURITY__API_TOKEN", "super-secret-prod-token-123")
    monkeypatch.setenv("SOMNIUM_STORAGE__DATABASE_URL", "postgresql://user:pass@db/somnium")

    settings = load_settings()
    assert settings.profile == ProfileName.PROD
    assert settings.inference.base_model_id == "Qwen/Qwen3-14B"
    assert settings.security.api_token is not None
    assert settings.security.api_token.get_secret_value() == "super-secret-prod-token-123"

    redacted = settings.redacted_dict()
    assert redacted["inference"]["base_model_id"] == "Qwen/Qwen3-14B"
    assert redacted["security"]["api_token"] == REDACTED_PLACEHOLDER
    assert redacted["storage"]["database_url"] == REDACTED_PLACEHOLDER
    serialized_str = str(redacted)
    assert "super-secret-prod-token-123" not in serialized_str
    assert "user:pass" not in serialized_str
