"""Content-integrity helpers for externally stored datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it whole."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise ``ValueError`` when a file does not match its pinned digest."""

    actual = sha256_file(path)
    normalized = expected.strip().lower()
    if actual != normalized:
        raise ValueError(
            f"SHA-256 mismatch for {Path(path).name}: expected {normalized}, got {actual}"
        )

