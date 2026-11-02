"""Authentication and scope resolution for the Freud gateway (M3.4).

Provides:
- `parse_scope`: Flexible parser accepting `Scope`, `"<kind>:<id>"`, bare `"<id>"`
  (defaulting to `ScopeKind.PROJECT`), or `None` (falling back to `default_scope`).
- `StaticLocalAuthProvider`: Configured from `SomniumSettings` (or explicit
  arguments) for local stdio MCP sessions and single-user workflows.
- `resolve_request_scopes`: Resolves and validates target scopes for a request
  against the caller's `AuthContext` and gateway defaults.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from somnium.config import SomniumSettings
from somnium.domain import AuthContext, Permission, Scope, ScopeKind, TenantId
from somnium.freud.errors import ErrorCode, FreudError

_VALID_SCOPE_KINDS = {k.value for k in ScopeKind}


def parse_scope(
    raw: Scope | str | Mapping[str, Any] | None,
    *,
    default_scope: Scope | None = None,
) -> Scope:
    """Parse a raw scope representation into a validated `Scope` object."""
    if raw is None:
        if default_scope is not None:
            return default_scope
        return Scope(kind=ScopeKind.PROJECT, id="default")

    if isinstance(raw, Scope):
        return raw

    if isinstance(raw, Mapping):
        kind_raw = str(raw.get("kind", ScopeKind.PROJECT.value)).strip().lower()
        id_raw = str(raw.get("id", "")).strip()
        if not id_raw:
            raise FreudError(
                ErrorCode.INVALID_REQUEST,
                "Scope dictionary must contain a non-empty 'id' field.",
            )
        try:
            return Scope(kind=ScopeKind(kind_raw), id=id_raw)
        except ValueError as exc:
            raise FreudError(
                ErrorCode.INVALID_REQUEST,
                f"Invalid scope specification: {exc}",
            ) from exc

    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            if default_scope is not None:
                return default_scope
            raise FreudError(ErrorCode.INVALID_REQUEST, "Scope string cannot be empty.")

        if ":" in cleaned:
            prefix, remainder = cleaned.split(":", 1)
            prefix_lower = prefix.strip().lower()
            if prefix_lower in _VALID_SCOPE_KINDS:
                scope_id = remainder.strip()
                if not scope_id:
                    raise FreudError(
                        ErrorCode.INVALID_REQUEST,
                        f"Scope string '{raw}' is missing an identifier after '{prefix}:'.",
                    )
                return Scope(kind=ScopeKind(prefix_lower), id=scope_id)

        return Scope(kind=ScopeKind.PROJECT, id=cleaned)

    raise FreudError(
        ErrorCode.INVALID_REQUEST,
        f"Unsupported scope type '{type(raw).__name__}'.",
    )


class StaticLocalAuthProvider:
    """Static local `AuthProvider` backed by gateway and security configuration (M3.4)."""

    def __init__(
        self,
        *,
        tenant_id: TenantId = "local-user",
        principal_id: str = "codex-local",
        default_scope: Scope | None = None,
        authorized_scopes: Sequence[Scope] | None = None,
        permissions: frozenset[Permission] | None = None,
        allow_all_scopes: bool = True,
        expected_token: str | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._principal_id = principal_id
        self._default_scope = default_scope or Scope(kind=ScopeKind.PROJECT, id="default")
        if authorized_scopes:
            deduped = list(dict.fromkeys(authorized_scopes))
            self._authorized_scopes = tuple(deduped)
        else:
            self._authorized_scopes = (self._default_scope,)
        self._permissions = (
            permissions
            if permissions is not None
            else frozenset(
                {
                    Permission.READ,
                    Permission.WRITE,
                    Permission.CORRECT,
                    Permission.FORGET,
                    Permission.ASSOCIATIVE_RECALL,
                }
            )
        )
        self._allow_all_scopes = allow_all_scopes
        self._expected_token = expected_token

    @classmethod
    def from_settings(cls, settings: SomniumSettings) -> StaticLocalAuthProvider:
        """Construct a `StaticLocalAuthProvider` from `SomniumSettings`."""
        gw = settings.gateway
        sec = settings.security
        token_val = sec.api_token.get_secret_value() if sec.api_token is not None else None
        # When explicit authorized_scopes are configured in the profile, enforce them
        # unless allow_all_scopes was explicitly kept True with an empty list.
        has_explicit_scopes = len(gw.authorized_scopes) > 0
        allow_all = gw.allow_all_scopes and not has_explicit_scopes
        scopes = gw.authorized_scopes if has_explicit_scopes else (gw.default_scope,)
        return cls(
            tenant_id=gw.default_tenant_id,
            principal_id=gw.default_principal_id,
            default_scope=gw.default_scope,
            authorized_scopes=scopes,
            allow_all_scopes=allow_all,
            expected_token=token_val,
        )

    @property
    def default_scope(self) -> Scope:
        """Default scope configured for this provider."""
        return self._default_scope

    def authenticate(
        self,
        token: str | None = None,
        requested_scopes: Sequence[Scope] | None = None,
    ) -> AuthContext:
        """Resolve and validate caller credentials and requested scopes into an `AuthContext`."""
        if self._expected_token is not None and token != self._expected_token:
            raise FreudError(
                ErrorCode.UNAUTHENTICATED,
                "Invalid or missing authentication token.",
            )

        if requested_scopes is not None and len(requested_scopes) > 0:
            if not self._allow_all_scopes and Permission.ADMIN not in self._permissions:
                unauthorized = [s for s in requested_scopes if s not in self._authorized_scopes]
                if unauthorized:
                    joined = ", ".join(str(s) for s in unauthorized)
                    raise FreudError(
                        ErrorCode.SCOPE_UNAUTHORIZED,
                        f"Principal '{self._principal_id}' is not authorized "
                        f"for scope(s): {joined}",
                        details={"unauthorized_scopes": [str(s) for s in unauthorized]},
                    )
                effective_scopes = self._authorized_scopes
            else:
                effective_scopes = tuple(
                    dict.fromkeys([*requested_scopes, *self._authorized_scopes])
                )
        else:
            effective_scopes = self._authorized_scopes

        return AuthContext(
            tenant_id=self._tenant_id,
            principal_id=self._principal_id,
            authorized_scopes=effective_scopes,
            permissions=self._permissions,
            allow_all_scopes=self._allow_all_scopes,
        )


def resolve_request_scopes(
    auth: AuthContext,
    requested_scope: Scope | None,
    *,
    default_scope: Scope | None = None,
) -> tuple[Scope, ...]:
    """Resolve the concrete scopes for a query and verify authorization in `auth`."""
    if requested_scope is not None:
        if not auth.is_scope_authorized(requested_scope):
            raise FreudError(
                ErrorCode.SCOPE_UNAUTHORIZED,
                f"Principal '{auth.principal_id}' is not authorized for scope '{requested_scope}'.",
                details={"requested_scope": str(requested_scope)},
            )
        return (requested_scope,)

    if auth.authorized_scopes:
        return auth.authorized_scopes

    if default_scope is not None:
        if not auth.is_scope_authorized(default_scope):
            raise FreudError(
                ErrorCode.SCOPE_UNAUTHORIZED,
                f"Principal '{auth.principal_id}' is not authorized for "
                f"default scope '{default_scope}'.",
            )
        return (default_scope,)

    raise FreudError(
        ErrorCode.SCOPE_UNAUTHORIZED,
        f"Principal '{auth.principal_id}' has no authorized scopes.",
    )


__all__ = [
    "StaticLocalAuthProvider",
    "parse_scope",
    "resolve_request_scopes",
]
