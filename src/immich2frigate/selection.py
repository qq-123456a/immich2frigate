"""Representative sampling from Immich's own vector spaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .immich_client import FaceCandidate, PersonRecord

FOUNDATION_COUNT = 5
MAX_TRAINING_COUNT = 30
DEFAULT_NOVELTY_DISTANCE = 0.08


@dataclass(frozen=True, slots=True)
class VectorFace:
    """One Immich face with the two vectors used for representative sampling."""

    source: FaceCandidate = field(repr=False)
    face_embedding: np.ndarray = field(repr=False)
    scene_embedding: np.ndarray = field(repr=False)


def select_adaptive_faces(
    person: PersonRecord,
    foundation_candidates: list[VectorFace],
    expansion_candidates: list[VectorFace],
    *,
    foundation_count: int = FOUNDATION_COUNT,
    max_count: int = MAX_TRAINING_COUNT,
    novelty_distance: float = DEFAULT_NOVELTY_DISTANCE,
) -> list[VectorFace]:
    """Select a strong foundation, then add only meaningfully novel samples.

    Foundation candidates are expected to have passed the stricter image-quality
    gate. The first five are chosen by centrality in Immich's face-vector space,
    a conservative proxy for typical identity views. Expansion can use a wider
    quality-approved pool, but a candidate is added only when its combined face
    and scene vector is sufficiently different from every image already chosen.
    """

    if isinstance(foundation_count, bool) or not isinstance(foundation_count, int) or foundation_count < 1:
        raise ValueError("foundation_count must be a positive integer")
    if isinstance(max_count, bool) or not isinstance(max_count, int) or max_count < foundation_count:
        raise ValueError("max_count must be at least foundation_count")
    if (
        isinstance(novelty_distance, bool)
        or not isinstance(novelty_distance, (int, float))
        or not np.isfinite(novelty_distance)
        or not 0 < novelty_distance < 2
    ):
        raise ValueError("novelty_distance must be between 0 and 2")

    if len(foundation_candidates) < foundation_count:
        raise ValueError(
            f"at least {foundation_count} foundation-quality faces are required"
        )

    expansion_by_face = {
        item.source.face_id: item for item in expansion_candidates if item.source.face_id is not None
    }
    for item in foundation_candidates:
        if item.source.face_id is None or item.source.face_id not in expansion_by_face:
            raise ValueError("foundation candidates must be included in expansion candidates")

    expansion = sorted(expansion_candidates, key=_candidate_key)
    foundation = sorted(foundation_candidates, key=_candidate_key)
    expansion_face = _validated_matrix(person, expansion, attr="face_embedding")
    expansion_scene = _validated_matrix(person, expansion, attr="scene_embedding")
    foundation_face = _validated_matrix(person, foundation, attr="face_embedding")

    expansion_face_unit = _row_normalize(expansion_face)
    expansion_scene_unit = _row_normalize(expansion_scene)
    foundation_face_unit = _row_normalize(foundation_face)

    centroid = np.mean(foundation_face_unit, axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if not np.isfinite(centroid_norm) or centroid_norm == 0:
        centrality = np.zeros(len(foundation), dtype=np.float32)
    else:
        centroid = centroid / centroid_norm
        centrality = foundation_face_unit @ centroid

    foundation_indices = sorted(
        range(len(foundation)),
        key=lambda i: (-float(centrality[i]), _candidate_key(foundation[i])),
    )[:foundation_count]
    selected_faces = [foundation[index].source.face_id for index in foundation_indices]

    index_by_face = {item.source.face_id: index for index, item in enumerate(expansion)}
    selected = [index_by_face[face_id] for face_id in selected_faces]
    selected_set = set(selected)

    combined = np.concatenate((expansion_face_unit, expansion_scene_unit), axis=1)
    combined *= np.float32(1.0 / np.sqrt(2.0))

    while len(selected) < min(max_count, len(expansion)):
        selected_matrix = combined[selected]
        best_index: int | None = None
        best_distance = -1.0

        for index in range(len(expansion)):
            if index in selected_set:
                continue
            nearest_similarity = float(np.max(selected_matrix @ combined[index]))
            distance = 1.0 - nearest_similarity
            if distance > best_distance + 1e-12:
                best_index = index
                best_distance = distance
            elif abs(distance - best_distance) <= 1e-12 and best_index is not None:
                if _candidate_key(expansion[index]) < _candidate_key(expansion[best_index]):
                    best_index = index

        if best_index is None or best_distance < novelty_distance:
            break
        selected.append(best_index)
        selected_set.add(best_index)

    return [expansion[index] for index in selected]


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
