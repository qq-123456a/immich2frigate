"""Representative sampling from Immich's own vector spaces.

The primary selector deliberately does not classify pose, lighting, expression,
scene type, or invent an identity/quality confidence score. Immich has already
assigned each face to a person and already computed both face and smart-search
embeddings. This module only asks which small subset best covers those vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .immich_client import FaceCandidate, PersonRecord


@dataclass(frozen=True, slots=True)
class VectorFace:
    """One Immich face with the two vectors used for coverage selection."""

    source: FaceCandidate = field(repr=False)
    face_embedding: np.ndarray = field(repr=False)
    scene_embedding: np.ndarray = field(repr=False)


def select_representative_faces(
    person: PersonRecord,
    candidates: list[VectorFace],
    *,
    count: int = 30,
) -> list[VectorFace]:
    """Return up to count faces that maximize face+scene coverage.

    Face and scene vectors are L2-normalized independently and concatenated,
    giving both Immich vector spaces equal influence without assigning semantic
    labels. Selection then uses a deterministic greedy facility-location
    objective: each chosen image should improve coverage of the whole
    candidate population, which avoids preferentially selecting isolated
    outliers merely because they are unusual.

    The order is meaningful: earlier entries add more marginal coverage and
    should be attempted first when Frigate rejects some registration images.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not candidates:
        return []

    ordered = sorted(candidates, key=_candidate_key)
    face_vectors = _validated_matrix(person, ordered, attr="face_embedding")
    scene_vectors = _validated_matrix(person, ordered, attr="scene_embedding")

    # Normalize each space separately. Concatenating two unit vectors and then
    # scaling by sqrt(2) makes combined cosine similarity the arithmetic mean
    # of Immich's face-space and scene-space cosine similarities.
    face_unit = _row_normalize(face_vectors)
    scene_unit = _row_normalize(scene_vectors)
    combined = np.concatenate((face_unit, scene_unit), axis=1)
    combined *= np.float32(1.0 / np.sqrt(2.0))

    similarity = combined @ combined.T
    # Facility-location coverage is easier to reason about on [0, 1]. This is
    # a monotonic transform of cosine similarity and is not a confidence score.
    similarity = np.clip((similarity + np.float32(1.0)) * np.float32(0.5), 0.0, 1.0)

    limit = min(count, len(ordered))
    coverage = np.zeros(len(ordered), dtype=np.float32)
    selected: list[int] = []
    available = np.ones(len(ordered), dtype=bool)

    for _ in range(limit):
        gains = np.full(len(ordered), -np.inf, dtype=np.float64)
        for index in np.flatnonzero(available):
            improved = np.maximum(coverage, similarity[:, index])
            gains[index] = float(np.sum(improved - coverage, dtype=np.float64))

        # np.argmax is deterministic and ordered is deterministically sorted,
        # so exact ties have a stable result.
        best = int(np.argmax(gains))
        if not np.isfinite(gains[best]):
            raise ValueError("representative selection could not make progress")
        selected.append(best)
        coverage = np.maximum(coverage, similarity[:, best])
        available[best] = False

    return [ordered[index] for index in selected]


def _candidate_key(item: VectorFace) -> tuple[str, str, str]:
    return (
        item.source.taken_at,
        item.source.asset_id,
        item.source.face_id or "",
    )


def _validated_matrix(
    person: PersonRecord,
    candidates: list[VectorFace],
    *,
    attr: str,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    dimension: int | None = None
    seen_faces: set[str] = set()
    seen_assets: set[str] = set()

    for item in candidates:
        if item.source.person_id != person.person_id:
            raise ValueError("candidate belongs to a different Immich person")
        if item.source.face_id is None:
            raise ValueError("representative candidates require an Immich face ID")
        if item.source.face_id in seen_faces:
            raise ValueError("duplicate Immich face ID in candidate set")
        if item.source.asset_id in seen_assets:
            raise ValueError("multiple candidate faces from one asset are ambiguous")
        seen_faces.add(item.source.face_id)
        seen_assets.add(item.source.asset_id)

        vector = np.asarray(getattr(item, attr), dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(f"{attr} must be a finite one-dimensional vector")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"{attr} must have a finite, non-zero norm")
        if dimension is None:
            dimension = int(vector.size)
        elif vector.size != dimension:
            raise ValueError(f"all {attr} vectors must have the same dimension")
        vectors.append(np.ascontiguousarray(vector))

    return np.stack(vectors).astype(np.float32, copy=False)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("embedding matrix contains an invalid norm")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)
