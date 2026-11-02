"""Architectural import-boundary tests (M0.3).

Verifies that ML framework libraries (`mlx`, `mlx_lm`, `torch`, `peft`, `vllm`,
`transformers`) are never imported outside `somnium/backends/<backend>/`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

FORBIDDEN_ML_ROOTS = frozenset(
    {
        "mlx",
        "mlx_lm",
        "torch",
        "peft",
        "vllm",
        "transformers",
    }
)


def find_forbidden_imports_in_source(
    source_code: str,
    rel_posix_path: str,
    forbidden_roots: frozenset[str] = FORBIDDEN_ML_ROOTS,
) -> list[str]:
    """Parse `source_code` and return human-readable violations for forbidden ML imports."""
    # Only concrete backend subpackages (`somnium/backends/mlx/...`, `somnium/backends/cuda/...`)
    # are allowed to import ML framework libraries.
    allowed_prefixes = (
        "somnium/backends/mlx/",
        "somnium/backends/cuda/",
    )
    if rel_posix_path.startswith(allowed_prefixes):
        return []

    tree = ast.parse(source_code, filename=rel_posix_path)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_pkg = alias.name.split(".")[0]
                if top_pkg in forbidden_roots:
                    violations.append(
                        f"{rel_posix_path}:{node.lineno}: forbidden import '{alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            top_pkg = node.module.split(".")[0]
            if top_pkg in forbidden_roots:
                violations.append(
                    f"{rel_posix_path}:{node.lineno}: forbidden from-import '{node.module}'"
                )

    return violations


def scan_package_import_boundaries(src_root: Path) -> list[str]:
    """Scan all Python files under `src_root` for forbidden ML imports."""
    all_violations: list[str] = []
    for py_file in sorted(src_root.rglob("*.py")):
        rel_posix = py_file.relative_to(src_root).as_posix()
        source = py_file.read_text(encoding="utf-8")
        all_violations.extend(find_forbidden_imports_in_source(source, rel_posix))
    return all_violations


def test_no_ml_imports_outside_backends() -> None:
    """Ensure no module outside `somnium/backends/<backend>/` imports ML libraries."""
    src_root = Path(__file__).resolve().parents[1] / "src"
    assert src_root.is_dir(), f"Expected src directory at {src_root}"
    violations = scan_package_import_boundaries(src_root)
    assert violations == [], "Import boundary violations detected:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    ("bad_code", "rel_path"),
    [
        ("import mlx.core as mx\n", "somnium/engram/pipeline.py"),
        ("from torch import nn\n", "somnium/dream/trainer.py"),
        ("import vllm\n", "somnium/freud/server.py"),
        ("from peft import LoraConfig\n", "somnium/backends/base.py"),
        ("import mlx_lm\n", "somnium/backends/fake.py"),
    ],
)
def test_boundary_checker_catches_deliberate_violation(bad_code: str, rel_path: str) -> None:
    """Verify the boundary checker fails when a forbidden import is introduced."""
    violations = find_forbidden_imports_in_source(bad_code, rel_path)
    assert len(violations) == 1


def test_boundary_checker_allows_backend_subpackages() -> None:
    """Verify concrete backend directories are permitted to import their ML frameworks."""
    mlx_code = "import mlx.core as mx\nimport mlx_lm\n"
    cuda_code = "import torch\nfrom peft import LoraConfig\nimport vllm\n"
    assert find_forbidden_imports_in_source(mlx_code, "somnium/backends/mlx/runner.py") == []
    assert find_forbidden_imports_in_source(cuda_code, "somnium/backends/cuda/runner.py") == []
