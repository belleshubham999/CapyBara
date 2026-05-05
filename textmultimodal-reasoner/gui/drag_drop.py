"""Drag-and-drop helpers for TkinterDnD2 widgets."""

from __future__ import annotations

from pathlib import Path


def normalize_drop_path(raw: str) -> Path:
    """Convert TkinterDnD drop payload to a usable path."""
    cleaned = raw.strip().strip("{}")
    return Path(cleaned)
