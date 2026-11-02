"""Write-path normalization, secret detection, and sensitivity classification (M2.3).

Per `design.md` §6.5 and `Milestones.md` M2.3:
- Normalizes incoming content strings.
- Rejects credentials, private keys, API tokens, and inline secrets before any
  persistence occurs.
- Classifies sensitivity by combining caller-provided sensitivity with
  rule-based inspection, ensuring high-sensitivity records are never eligible
  for model training.
"""

from __future__ import annotations

import re

from somnium.domain import MemoryStatus, SensitivityLevel, VerificationLevel


class SecretDetectedError(ValueError):
    """Raised when incoming memory content contains credentials or private keys."""


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "PEM private key",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
    ),
    (
        "OpenAI / Anthropic API key",
        re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}\b"),
    ),
    (
        "GitHub personal access token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "AWS access key ID",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    (
        "Slack token",
        re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{16,}\b"),
    ),
    (
        "Inline credential assignment",
        re.compile(
            r"(?i)\b(?:[a-z0-9_]+_)?(?:api_key|secret_key|private_key|access_token|password|passwd|client_secret)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9/+_.~-]{12,}['\"]?"
        ),
    ),
    (
        "Database URI with embedded password",
        re.compile(r"\b(?:postgresql|mysql|mongodb|redis)://[^:\s]+:[^@\s]{4,}@"),
    ),
)

_HIGH_SENSITIVITY_KEYWORDS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bconfidential\b"),
    re.compile(r"(?i)\bembargoed\b"),
    re.compile(r"(?i)\bnda\b"),
    re.compile(r"(?i)\bstrictly\s+private\b"),
    re.compile(r"(?i)\bdo\s+not\s+train\b"),
    re.compile(r"(?i)\bm&a\s+memo\b"),
    re.compile(r"(?i)\bclassified\b"),
    re.compile(r"(?i)\.corp\.internal\b"),
    re.compile(r"\bCUST-[0-9]{4,}\b"),
)


def normalize_content(raw_content: str) -> str:
    """Normalize line endings, strip control characters, and collapse excess whitespace."""
    cleaned = raw_content.replace("\r\n", "\n").replace("\r", "\n")
    # Remove non-printable ASCII control characters except newline and tab
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    lines = [" ".join(line.split()) for line in cleaned.split("\n")]
    result = "\n".join(line for line in lines if line).strip()
    if not result:
        raise ValueError("Memory content cannot be empty after normalization.")
    return result


def assert_no_secrets(content: str) -> None:
    """Raise `SecretDetectedError` if `content` matches any credential or private key pattern."""
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(content):
            raise SecretDetectedError(
                f"Rejected memory write: detected potential secret ({label})."
            )


def classify_sensitivity(
    content: str,
    explicit_sensitivity: SensitivityLevel = SensitivityLevel.NORMAL,
) -> SensitivityLevel:
    """Determine record sensitivity from explicit input and content rules."""
    if explicit_sensitivity == SensitivityLevel.HIGH:
        return SensitivityLevel.HIGH
    for pattern in _HIGH_SENSITIVITY_KEYWORDS:
        if pattern.search(content):
            return SensitivityLevel.HIGH
    return SensitivityLevel.NORMAL


def compute_training_eligibility(
    *,
    sensitivity: SensitivityLevel,
    verification: VerificationLevel,
    status: MemoryStatus,
) -> bool:
    """Compute `training_eligible` per design §5.2 and §5.4."""
    if sensitivity == SensitivityLevel.HIGH:
        return False
    if verification == VerificationLevel.UNVERIFIED:
        return False
    return status == MemoryStatus.ACTIVE
