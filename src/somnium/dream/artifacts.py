"""Content-addressed filesystem artifact store for datasets and adapters (M5.4).

Implements the `ArtifactStore` Protocol (`somnium.interfaces`):
- Stores immutable blobs addressed by `(namespace, sha256_hex)`.
- Writes atomically via a temporary file and `os.replace` so concurrent readers
  never observe partial writes.
- Verifies SHA-256 content integrity on read (`get_bytes`).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


class LocalFilesystemArtifactStore:
    """Content-addressed local directory artifact store (M5.4)."""

    def __init__(self, root_dir: Path | str) -> None:
        self._root_dir = Path(root_dir).resolve()
        self._root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root_dir(self) -> Path:
        """Root directory on disk where artifacts are stored."""
        return self._root_dir

    def _validate_namespace(self, namespace: str) -> str:
        cleaned = namespace.strip().strip("/")
        if not cleaned or ".." in cleaned or "/" in cleaned or "\\" in cleaned:
            raise ValueError(f"Invalid artifact namespace '{namespace}'.")
        return cleaned

    def _validate_hash(self, content_hash: str) -> str:
        cleaned = content_hash.strip().lower()
        if len(cleaned) != 64 or any(c not in "0123456789abcdef" for c in cleaned):
            raise ValueError(f"Invalid SHA-256 content hash '{content_hash}'.")
        return cleaned

    def path_for(self, namespace: str, content_hash: str) -> Path:
        """Return the deterministic filesystem path for `(namespace, content_hash)`."""
        ns = self._validate_namespace(namespace)
        digest = self._validate_hash(content_hash)
        return self._root_dir / ns / digest[:2] / f"{digest}.bin"

    def put_bytes(self, namespace: str, content: bytes) -> str:
        """Atomically persist `content` under `namespace` and return its SHA-256 hex digest."""
        ns = self._validate_namespace(namespace)
        digest = hashlib.sha256(content).hexdigest()
        target_path = self.path_for(ns, digest)
        if target_path.is_file():
            return digest

        target_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target_path.parent),
            prefix=f".tmp-{digest[:8]}-",
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
            os.replace(tmp_name, target_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

        return digest

    def get_bytes(self, namespace: str, content_hash: str) -> bytes:
        """Read and verify bytes for `(namespace, content_hash)` or raise `KeyError`."""
        target_path = self.path_for(namespace, content_hash)
        if not target_path.is_file():
            raise KeyError(f"Artifact not found: {namespace}/{content_hash}")
        data = target_path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != content_hash.strip().lower():
            raise ValueError(
                f"Artifact integrity check failed for {namespace}/{content_hash}: "
                f"actual sha256={actual_hash}"
            )
        return data

    def exists(self, namespace: str, content_hash: str) -> bool:
        """Return True if `(namespace, content_hash)` exists on disk."""
        try:
            target_path = self.path_for(namespace, content_hash)
        except ValueError:
            return False
        return target_path.is_file()

    def uri_for(self, namespace: str, content_hash: str) -> str:
        """Return the canonical `file://` URI for `(namespace, content_hash)`."""
        return self.path_for(namespace, content_hash).as_uri()


__all__ = ["LocalFilesystemArtifactStore"]
