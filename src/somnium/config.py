"""Typed configuration system with profile support and secret redaction.

Supports loading settings from versioned YAML profiles (`dev`, `test`, `prod`)
and overriding any field via `SOMNIUM_*` environment variables (using `__` as
the nested delimiter, e.g., `SOMNIUM_INFERENCE__BASE_MODEL_ID=...`).
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from somnium.domain import RecallMode, Scope, ScopeKind, TenantId

REDACTED_PLACEHOLDER = "***REDACTED***"


class ProfileName(StrEnum):
    """Supported configuration profile names."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class GatewayConfig(BaseModel):
    """Freud MCP / HTTP gateway configuration."""

    model_config = ConfigDict(extra="forbid")

    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    default_tenant_id: TenantId = "local-user"
    default_principal_id: str = Field(default="codex-local", min_length=1)
    default_scope: Scope = Field(
        default_factory=lambda: Scope(kind=ScopeKind.PROJECT, id="default")
    )
    authorized_scopes: tuple[Scope, ...] = Field(default_factory=tuple)
    allow_all_scopes: bool = True
    default_recall_mode: RecallMode = RecallMode.AUTO
    allow_cross_scope_by_default: bool = True


class StorageConfig(BaseModel):
    """Hippocore database and artifact storage paths."""

    model_config = ConfigDict(extra="forbid")

    backend: str = "sqlite"
    database_url: SecretStr = SecretStr("sqlite:///.somnium/hippocore.db")
    artifacts_dir: Path = Path(".somnium/artifacts")


class InferenceConfig(BaseModel):
    """Engram inference backend configuration."""

    model_config = ConfigDict(extra="forbid")

    backend: str = "mlx"
    base_model_id: str = "Qwen/Qwen3-1.7B"
    base_model_revision: str = "main"
    quantization: str | None = "4bit"
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class TrainingConfig(BaseModel):
    """Dream training worker and LoRA hyperparameter configuration."""

    model_config = ConfigDict(extra="forbid")

    backend: str = "mlx"
    default_mode: str = "incremental"
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(default=4, ge=1)
    epochs: int = Field(default=2, ge=1)
    replay_new_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    replay_historical_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    replay_negative_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class DreamConfig(BaseModel):
    """Dream Engine scheduling and resource policy configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    idle_threshold_seconds: int = Field(default=1800, ge=0)
    min_verified_examples: int = Field(default=100, ge=1)
    pause_on_associative_traffic: bool = True
    require_ac_power: bool = True
    max_memory_pressure_pct: int = Field(default=85, ge=1, le=100)


class RoutingConfig(BaseModel):
    """Rule-based RecallRouter thresholds and behavior."""

    model_config = ConfigDict(extra="forbid")

    min_evidence_relevance: float = Field(default=0.65, ge=0.0, le=1.0)
    associative_cue_patterns: tuple[str, ...] = (
        "something like",
        "similar to",
        "reminds me of",
        "why did we",
        "how did it evolve",
    )


class SecurityConfig(BaseModel):
    """Authentication and secret configuration (supplied via environment in prod)."""

    model_config = ConfigDict(extra="forbid")

    auth_provider: str = "static"
    require_network_auth: bool = True
    api_token: SecretStr | None = None
    signing_secret: SecretStr | None = None


class ObservabilityConfig(BaseModel):
    """Structured logging and metrics configuration."""

    model_config = ConfigDict(extra="forbid")

    log_level: str = "INFO"
    log_json: bool = True
    metrics_enabled: bool = False


class _YamlProfileSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that reads values from a YAML profile file."""

    def __init__(self, settings_cls: type[BaseSettings], yaml_path: Path | None) -> None:
        super().__init__(settings_cls)
        self._yaml_path = yaml_path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if self._yaml_path is None or not self._yaml_path.is_file():
            return {}
        with self._yaml_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            return {}
        return loaded


def _default_configs_dir() -> Path:
    """Locate the repository `configs/` directory relative to the package or cwd."""
    cwd_configs = Path.cwd() / "configs"
    if cwd_configs.is_dir():
        return cwd_configs
    repo_configs = Path(__file__).resolve().parents[2] / "configs"
    return repo_configs


class SomniumSettings(BaseSettings):
    """Top-level typed settings for Somnium."""

    model_config = SettingsConfigDict(
        env_prefix="SOMNIUM_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    profile: ProfileName = ProfileName.DEV
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dream: DreamConfig = Field(default_factory=DreamConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    _active_yaml_path: Path | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order sources: init args > env vars > YAML profile > defaults."""
        return (
            init_settings,
            env_settings,
            _YamlProfileSettingsSource(settings_cls, cls._active_yaml_path),
            file_secret_settings,
        )

    def redacted_dict(self) -> dict[str, Any]:
        """Serialize settings to a dictionary with all SecretStr values redacted."""
        raw = self.model_dump(mode="python")
        return _redact_secrets(raw)


def _redact_secrets(value: Any) -> Any:
    """Recursively convert SecretStr and Path values into safe serializable forms."""
    if isinstance(value, SecretStr):
        return REDACTED_PLACEHOLDER
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {k: _redact_secrets(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item) for item in value]
    return value


def load_settings(
    profile: ProfileName | str | None = None,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> SomniumSettings:
    """Load `SomniumSettings` for the requested profile with environment overrides.

    Precedence (highest to lowest):
    1. Explicit `overrides` dictionary passed to `load_settings`
    2. `SOMNIUM_*` environment variables
    3. Profile YAML file (`configs/<profile>.yaml` or `config_path`)
    4. Field defaults in `SomniumSettings`
    """
    resolved_profile_str = (
        str(profile)
        if profile is not None
        else os.environ.get("SOMNIUM_PROFILE", ProfileName.DEV.value)
    )
    resolved_profile = ProfileName(resolved_profile_str)

    yaml_path = config_path or (_default_configs_dir() / f"{resolved_profile.value}.yaml")
    init_kwargs: dict[str, Any] = dict(overrides or {})
    if "profile" not in init_kwargs and "SOMNIUM_PROFILE" not in os.environ and profile is not None:
        init_kwargs["profile"] = resolved_profile

    previous_path = SomniumSettings._active_yaml_path
    try:
        SomniumSettings._active_yaml_path = yaml_path
        return SomniumSettings(**init_kwargs)
    finally:
        SomniumSettings._active_yaml_path = previous_path
