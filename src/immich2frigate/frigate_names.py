"""Name normalization shared by Frigate face-library operations."""

from __future__ import annotations


def frigate_face_name(name: str) -> str:
    """Match Frigate's face-folder rule: spaces in labels become underscores."""
    return name.replace(" ", "_")
