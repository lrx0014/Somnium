"""Freud: External memory gateway, authentication, scope authorization, and routing."""

from __future__ import annotations

from somnium.freud.auth import (
    StaticLocalAuthProvider,
    parse_scope,
    resolve_request_scopes,
)
from somnium.freud.errors import ErrorCode, ErrorResponse, FreudError
from somnium.freud.formatting import (
    UNTRUSTED_DATA_NOTICE,
    attach_untrusted_delimiters,
    delimit_untrusted_content,
    format_associative_response_dict,
    format_evidence_response_dict,
)
from somnium.freud.mcp_server import (
    MEMORY_ASK_DESCRIPTION,
    MEMORY_CORRECT_DESCRIPTION,
    MEMORY_FORGET_DESCRIPTION,
    MEMORY_STATUS_DESCRIPTION,
    MEMORY_STORE_DESCRIPTION,
    build_freud_stack,
    create_mcp_server,
    run_stdio_server,
)
from somnium.freud.routing import RoutingDecisionRecord, RuleBasedRecallRouter
from somnium.freud.service import MemoryService

__all__ = [
    "ErrorCode",
    "ErrorResponse",
    "FreudError",
    "MEMORY_ASK_DESCRIPTION",
    "MEMORY_CORRECT_DESCRIPTION",
    "MEMORY_FORGET_DESCRIPTION",
    "MEMORY_STATUS_DESCRIPTION",
    "MEMORY_STORE_DESCRIPTION",
    "MemoryService",
    "RoutingDecisionRecord",
    "RuleBasedRecallRouter",
    "StaticLocalAuthProvider",
    "UNTRUSTED_DATA_NOTICE",
    "attach_untrusted_delimiters",
    "build_freud_stack",
    "create_mcp_server",
    "delimit_untrusted_content",
    "format_associative_response_dict",
    "format_evidence_response_dict",
    "parse_scope",
    "resolve_request_scopes",
    "run_stdio_server",
]
