"""Representative sampling from Immich's own vector spaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .immich_client import FaceCandidate, PersonRecord

FOUNDATION_COUNT = 5
MAX_TRAINING_COUNT = 30
MAX_SELECTION_CANDIDATES = 2_000
DEFAULT_NOVELTY_DISTANCE = 0.08
FOUNDATION_DUPLICATE_DISTANCE = 0.025
_IDENTITY_MEDOID_MARGIN = 0.35
_IDENTITY_NEIGHBOR_MARGIN = 0.20
_ROBUST_SIGMA = 6.0


@dataclass(frozen=True, slots=True)
class VectorFace:
    """One Immich face with the vectors used for representative sampling."""

    source: FaceCandidate = field(repr=False)
    face_embedding: np.ndarray = field(repr=False)
    scene_embedding: np.ndarray | None = field(default=None, repr=False)


def select_adaptive_faces(
    person: PersonRecord,
    foundation_candidates: list[VectorFace],
    expansion_candidates: list[VectorFace],
    *,
    foundation_count: int = FOUNDATION_COUNT,
    max_count: int = MAX_TRAINING_COUNT,
    novelty_distance: float = DEFAULT_NOVELTY_DISTANCE,
) -> list[VectorFace]:
    """Select a safe identity core, then add only useful diversity.

    Identity safety is derived only from Immich's persisted face embeddings. A
    conservative robust envelope removes isolated or clearly off-cluster faces
    before novelty is rewarded. Foundation faces come from that safe core and
    prefer central samples while avoiding near-duplicates when alternatives
    exist. Expansion uses face-vector novelty plus scene-vector novelty when a
    Smart Search embedding is available; missing scene vectors fall back to
    face-only distance rather than excluding the candidate.
    """

    if isinstance(foundation_count, bool) or not isinstance(foundation_count, int) or foundation_count < 1:
        raise ValueError("foundation_count must be a positive integer")
    if isinstance(max_count, bool) or not isinstance(max_count, int) or max_count < foundation_count:
        raise ValueError("max_count must be at least foundation_count")
    if max_count > MAX_TRAINING_COUNT:
        raise ValueError(f"max_count must not exceed {MAX_TRAINING_COUNT}")
    if (
        isinstance(novelty_distance, bool)
        or not isinstance(novelty_distance, (int, float))
        or not np.isfinite(novelty_distance)
        or not 0 < novelty_distance < 2
    ):
        raise ValueError("novelty_distance must be between 0 and 2")

    if len(foundation_candidates) < foundation_count:
        raise ValueError(f"at least {foundation_count} foundation-quality faces are required")

    expansion_by_face = {
        item.source.face_id: item for item in expansion_candidates if item.source.face_id is not None
    }
    for item in foundation_candidates:
        match = expansion_by_face.get(item.source.face_id)
        if item.source.face_id is None or match is None:
            raise ValueError("foundation candidates must be included in expansion candidates")
        if match.source.asset_id != item.source.asset_id:
            raise ValueError("foundation and expansion candidates disagree on the asset ID")

    expansion = sorted(expansion_candidates, key=_candidate_key)
    foundation = sorted(foundation_candidates, key=_candidate_key)
    if len(expansion) > MAX_SELECTION_CANDIDATES:
        foundation_ids = {item.source.face_id for item in foundation}
        if len(foundation) >= MAX_SELECTION_CANDIDATES:
            foundation = _sample_evenly(foundation, MAX_SELECTION_CANDIDATES)
            foundation_ids = {item.source.face_id for item in foundation}
        remaining = [item for item in expansion if item.source.face_id not in foundation_ids]
        room = MAX_SELECTION_CANDIDATES - len(foundation_ids)
        selected_ids = foundation_ids | {
            item.source.face_id for item in _sample_evenly(remaining, room)
        }
        expansion = [item for item in expansion if item.source.face_id in selected_ids]
        foundation = [item for item in foundation if item.source.face_id in selected_ids]
    expansion_face = _validated_matrix(person, expansion, attr="face_embedding")
    foundation_face = _validated_matrix(person, foundation, attr="face_embedding")
    expansion_face_unit = _row_normalize(expansion_face)
    foundation_face_unit = _row_normalize(foundation_face)

    expansion_safe = _identity_safe_mask(expansion_face_unit, minimum_keep=foundation_count)
    safe_face_ids = {
        expansion[index].source.face_id for index in range(len(expansion)) if expansion_safe[index]
    }
    safe_foundation = [item for item in foundation if item.source.face_id in safe_face_ids]
    if len(safe_foundation) < foundation_count:
        raise ValueError(
            f"only {len(safe_foundation)} foundation-quality faces remain inside the Immich identity core; "
            f"{foundation_count} are required"
        )

    safe_foundation_face = np.stack(
        [foundation_face_unit[index] for index, item in enumerate(foundation) if item.source.face_id in safe_face_ids]
    ).astype(np.float32, copy=False)
    foundation_pair_distance = _combined_pair_distance(
        person, safe_foundation, include_scene=False
    )
    centrality_distance = _medoid_distances(safe_foundation_face)
    density_distance = _local_density_distances(safe_foundation_face)
    core_score = centrality_distance + density_distance
    foundation_indices = _select_foundation_indices(
        safe_foundation,
        core_score,
        foundation_pair_distance,
        foundation_count,
    )
    selected_face_ids = [safe_foundation[index].source.face_id for index in foundation_indices]

    safe_expansion = [item for index, item in enumerate(expansion) if expansion_safe[index]]
    index_by_face = {item.source.face_id: index for index, item in enumerate(safe_expansion)}
    selected = [index_by_face[face_id] for face_id in selected_face_ids]
    selected_set = set(selected)
    pair_distance = _combined_pair_distance(person, safe_expansion)

    while len(selected) < min(max_count, len(safe_expansion)):
        best_index: int | None = None
        best_distance = -1.0

        for index in range(len(safe_expansion)):
            if index in selected_set:
                continue
            nearest_distance = float(np.min(pair_distance[index, selected]))
            if nearest_distance > best_distance + 1e-12:
                best_index = index
                best_distance = nearest_distance
            elif abs(nearest_distance - best_distance) <= 1e-12 and best_index is not None:
                if _candidate_key(safe_expansion[index]) < _candidate_key(safe_expansion[best_index]):
                    best_index = index

        if best_index is None or best_distance < novelty_distance:
            break
        selected.append(best_index)
        selected_set.add(best_index)

    return [safe_expansion[index] for index in selected]


def _select_foundation_indices(
    candidates: list[VectorFace],
    centrality_distance: np.ndarray,
    pair_distance: np.ndarray,
    count: int,
) -> list[int]:
    order = sorted(
        range(len(candidates)),
        key=lambda i: (float(centrality_distance[i]), _candidate_key(candidates[i])),
    )
    selected: list[int] = []
    for index in order:
        if not selected or float(np.min(pair_distance[index, selected])) >= FOUNDATION_DUPLICATE_DISTANCE:
            selected.append(index)
            if len(selected) == count:
                return selected

    # A person may genuinely have only near-identical source photos. Keep the
    # five-image safety floor by filling remaining slots with the most central
    # candidates rather than failing or reaching for an outlier.
    for index in order:
        if index not in selected:
            selected.append(index)
            if len(selected) == count:
                return selected
    return selected


def _identity_safe_mask(face_unit: np.ndarray, *, minimum_keep: int) -> np.ndarray:
    size = face_unit.shape[0]
    similarity = np.clip(face_unit @ face_unit.T, -1.0, 1.0)
    medoid = int(np.argmax(np.mean(similarity, axis=1)))
    medoid_distance = 1.0 - similarity[:, medoid]

    without_self = similarity.copy()
    np.fill_diagonal(without_self, -np.inf)
    nearest_neighbor_distance = 1.0 - np.max(without_self, axis=1)

    medoid_limit = _robust_upper_fence(medoid_distance, _IDENTITY_MEDOID_MARGIN)
    neighbor_limit = _robust_upper_fence(nearest_neighbor_distance, _IDENTITY_NEIGHBOR_MARGIN)
    safe = (medoid_distance <= medoid_limit) & (nearest_neighbor_distance <= neighbor_limit)

    # Never manufacture confidence by relaxing the robust gate. If it leaves too
    # few samples for the required foundation, the caller fails closed before a
    # destructive Frigate rebuild.
    return safe


def _robust_upper_fence(values: np.ndarray, minimum_margin: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    return median + max(_ROBUST_SIGMA * robust_sigma, minimum_margin)


def _medoid_distances(face_unit: np.ndarray) -> np.ndarray:
    similarity = np.clip(face_unit @ face_unit.T, -1.0, 1.0)
    medoid = int(np.argmax(np.mean(similarity, axis=1)))
    return np.ascontiguousarray(1.0 - similarity[:, medoid], dtype=np.float32)


def _local_density_distances(face_unit: np.ndarray) -> np.ndarray:
    similarity = np.clip(face_unit @ face_unit.T, -1.0, 1.0)
    np.fill_diagonal(similarity, -np.inf)
    neighbor_count = min(5, max(1, face_unit.shape[0] - 1))
    nearest = np.partition(similarity, -neighbor_count, axis=1)[:, -neighbor_count:]
    return np.ascontiguousarray(1.0 - np.mean(nearest, axis=1), dtype=np.float32)


def _combined_pair_distance(
    person: PersonRecord,
    candidates: list[VectorFace],
    *,
    include_scene: bool = True,
) -> np.ndarray:
    face = _row_normalize(_validated_matrix(person, candidates, attr="face_embedding"))
    face_similarity = np.clip(face @ face.T, -1.0, 1.0)
    similarity = face_similarity.copy()

    scene_rows: list[np.ndarray] = []
    scene_indices: list[int] = []
    scene_dimension: int | None = None
    if include_scene:
        for index, item in enumerate(candidates):
            if item.scene_embedding is None:
                continue
            vector = np.asarray(item.scene_embedding, dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError("scene_embedding must be a finite one-dimensional vector")
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("scene_embedding must have a finite, non-zero norm")
            if scene_dimension is None:
                scene_dimension = int(vector.size)
            elif vector.size != scene_dimension:
                raise ValueError("all available scene_embedding vectors must have the same dimension")
            scene_rows.append(np.ascontiguousarray(vector))
            scene_indices.append(index)

    if scene_rows:
        scene = _row_normalize(np.stack(scene_rows).astype(np.float32, copy=False))
        scene_similarity = np.clip(scene @ scene.T, -1.0, 1.0)
        indices = np.asarray(scene_indices, dtype=np.intp)
        pairs = np.ix_(indices, indices)
        similarity[pairs] = (face_similarity[pairs] + scene_similarity) * 0.5

    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return np.ascontiguousarray(distance, dtype=np.float32)


def _candidate_key(item: VectorFace) -> tuple[str, str, str]:
    return (
        item.source.taken_at,
        item.source.asset_id,
        item.source.face_id or "",
    )


def _sample_evenly(candidates: list[VectorFace], count: int) -> list[VectorFace]:
    if len(candidates) <= count:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, count, dtype=np.intp)
    return [candidates[index] for index in indices]


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
