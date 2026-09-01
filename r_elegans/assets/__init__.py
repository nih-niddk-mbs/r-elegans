"""Small, versioned runtime assets distributed with the Python package."""

from __future__ import annotations

from importlib.resources import files
import json
from typing import Any


def load_asset_document(name: str) -> dict[str, Any]:
    """Read a bundled JSON asset without assuming a filesystem installation."""

    resource = files(__package__).joinpath(name)
    return json.loads(resource.read_text(encoding="utf-8"))


__all__ = ["load_asset_document"]
