"""Name normalization shared by Frigate face-library operations."""

from __future__ import annotations


def frigate_face_name(name: str) -> str:
    """Make an Immich name a valid Frigate face label."""
    return "_".join("".join(char for char in name if char.isalnum() or char in "'_- ").split())
