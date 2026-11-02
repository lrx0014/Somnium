"""Stable error codes and structured error response model for Freud (M3.5).

Ensures all gateway errors (authentication, scope authorization, missing memory,
conflicts on superseded/deleted records, secret detection, or validation errors)
serialize into a predictable JSON schema with stable `error_code` identifiers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from somnium import API_VERSION


class ErrorCode(StrEnum):
    """Stable error codes returned by the Freud gateway."""

    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    SCOPE_UNAUTHORIZED = "SCOPE_UNAUTHORIZED"
    MEMORY_NOT_FOUND = "MEMORY_NOT_FOUND"
    CONFLICT_SUPERSEDED = "CONFLICT_SUPERSEDED"
    CONFLICT_DELETED = "CONFLICT_DELETED"
    SECRET_DETECTED = "SECRET_DETECTED"
    MODE_UNAVAILABLE = "MODE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    """Versioned structured error response payload for Freud tools and endpoints."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    status: Literal["error"] = "error"
    error_code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class FreudError(Exception):
    """Domain exception carrying a stable `ErrorCode` and optional structured details."""

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def to_response(self) -> ErrorResponse:
        """Convert this exception into a versioned `ErrorResponse` model."""
        return ErrorResponse(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
        )


__all__ = ["ErrorCode", "ErrorResponse", "FreudError"]
