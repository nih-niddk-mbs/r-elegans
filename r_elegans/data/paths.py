"""Resolve and initialize the external scientific-data directory.

Datasets are intentionally excluded from the source repository. Every caller
must select a data root explicitly through ``R_ELEGANS_DATA_DIR`` or a function
argument. Repository-local roots are rejected as an additional safety check.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "R_ELEGANS_DATA_DIR"
DATA_DIRECTORIES = (
    "raw/connectome/cook2019",
    "raw/neurotransmitters",
    "raw/physiology",
    "raw/functional",
    "processed/connectome",
    "processed/parameters",
    "manifests",
    "cache",
    "results",
)


class DataRootNotConfigured(RuntimeError):
    """Raised when no external data root has been selected."""


def repository_root() -> Path:
    """Return the source checkout root containing ``pyproject.toml``."""

    return Path(__file__).resolve().parents[2]


def _reject_repository_local_path(path: Path) -> None:
    try:
        path.relative_to(repository_root())
    except ValueError:
        return
    raise ValueError(
        f"Scientific data root must be outside the Git repository: {path}"
    )


def get_data_root(
    path: str | os.PathLike[str] | None = None,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve the configured external data root without creating it."""

    configured = path if path is not None else os.environ.get(DATA_ROOT_ENV)
    if not configured:
        raise DataRootNotConfigured(
            f"Set {DATA_ROOT_ENV} to an external data directory"
        )
    resolved = Path(configured).expanduser().resolve(strict=False)
    _reject_repository_local_path(resolved)
    if must_exist and not resolved.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {resolved}")
    return resolved


def initialize_data_root(path: str | os.PathLike[str]) -> Path:
    """Create the standard external directory layout and return its root."""

    root = get_data_root(path, must_exist=False)
    for relative_path in DATA_DIRECTORIES:
        (root / relative_path).mkdir(parents=True, exist_ok=True)
    return root


def data_path(
    *parts: str,
    root: str | os.PathLike[str] | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a path beneath the configured root and prevent path escape."""

    data_root = get_data_root(root)
    candidate = data_root.joinpath(*parts).resolve(strict=False)
    try:
        candidate.relative_to(data_root)
    except ValueError as error:
        raise ValueError("Data path must remain beneath the data root") from error
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Data path does not exist: {candidate}")
    return candidate

