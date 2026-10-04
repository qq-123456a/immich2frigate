"""In-memory adapter for if-curator's Frigate-independent selection step."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .immich_client import FaceCandidate, PersonRecord


@dataclass(frozen=True, slots=True)
class EmbeddedFace:
    """A source candidate, its BGR preview, and verified ArcFace vector."""

    source: FaceCandidate = field(repr=False)
    image_bgr: np.ndarray = field(repr=False)
    embedding: np.ndarray = field(repr=False)


def select_diverse_faces(
    person: PersonRecord,
    candidates: list[EmbeddedFace],
    *,
    count: int = 5,
    threshold: float = 0.3,
) -> list[EmbeddedFace]:
    """Select diverse, identity-consistent faces entirely in memory.

    The pinned selector's `recognized` score uses Frigate 0.17 math; this
    wrapper intentionally ignores it. Embeddings must come from the separately
    validated Frigate 0.18 model pipeline.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("threshold must be a finite number between 0 and 1")
    if not candidates:
        return []

    vectors: list[np.ndarray] = []
    dimension: int | None = None
    asset_ids: set[str] = set()
    for item in candidates:
        if item.source.person_id != person.person_id:
            raise ValueError("candidate belongs to a different Immich person")
        if item.source.asset_id in asset_ids:
            raise ValueError("multiple candidate faces from one asset are ambiguous")
        asset_ids.add(item.source.asset_id)
        vector = np.asarray(item.embedding)
        if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.floating):
            raise ValueError("each embedding must be a one-dimensional floating-point vector")
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError("embeddings must be non-empty and finite")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("embeddings must have a finite, non-zero norm")
        if dimension is None:
            dimension = vector.size
        elif vector.size != dimension:
            raise ValueError("all embeddings must have the same dimension")
        vectors.append(np.ascontiguousarray(vector))

    try:
        from if_curator.selection import Candidate, Job, select
    except ImportError as error:
        raise RuntimeError(
            "Install the optional pinned if-curator dependency to select candidates"
        ) from error

    upstream_candidates = []
    source_by_identity: dict[int, EmbeddedFace] = {}
    for item, vector in zip(candidates, vectors, strict=True):
        candidate = Candidate(
            item.source.asset_id,
            taken=item.source.taken_at,
            checksum=item.source.checksum,
            face_id=item.source.face_id,
            box=item.source.box,
            frame=item.source.frame,
            embedding=vector,
        )
        upstream_candidates.append(candidate)
        source_by_identity[id(candidate)] = item

    job = Job(
        {"id": person.person_id, "name": person.name},
        count,
        candidates=upstream_candidates,
    )
    select([job], threshold)
    return [
        source_by_identity[id(candidate)]
        for candidate in job.selected
        if id(candidate) in source_by_identity
    ]
