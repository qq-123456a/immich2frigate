from __future__ import annotations

import numpy as np
import pytest

from immich2frigate.immich_client import FaceCandidate, PersonRecord
from immich2frigate.selection import VectorFace, select_adaptive_faces

PERSON = "00000000-0000-4000-8000-000000000001"


def vector_face(index: int, face, scene=None) -> VectorFace:
    source = FaceCandidate(
        person_id=PERSON,
        face_id=f"00000000-0000-4000-8000-{index:012d}",
        asset_id=f"10000000-0000-4000-8000-{index:012d}",
        taken_at=f"2025-01-{(index % 28) + 1:02d}T00:00:00Z",
        checksum=f"checksum-{index}",
        box=(0.0, 0.0, 100.0, 100.0),
        frame=(100, 100),
    )
    return VectorFace(
        source=source,
        face_embedding=np.asarray(face, dtype=np.float32),
        scene_embedding=None if scene is None else np.asarray(scene, dtype=np.float32),
    )


def test_same_distribution_stops_at_five_foundation_faces():
    person = PersonRecord(PERSON, "Synthetic Person")
    candidates = [
        vector_face(i, [1.0, i * 0.001], [1.0, i * 0.001])
        for i in range(1, 11)
    ]

    result = select_adaptive_faces(person, candidates[:7], candidates, novelty_distance=0.08)

    assert len(result) == 5
    assert len({item.source.face_id for item in result}) == 5


def test_diverse_distribution_expands_beyond_foundation_without_exceeding_cap():
    person = PersonRecord(PERSON, "Synthetic Person")
    foundation = [
        vector_face(i, [1.0, i * 0.001], [1.0, i * 0.001])
        for i in range(1, 6)
    ]
    expansion = foundation + [
        vector_face(6, [0.7, 0.7], [1.0, 0.0]),
        vector_face(7, [0.6, 0.8], [0.0, 1.0]),
        vector_face(8, [0.8, 0.6], [-1.0, 0.0]),
    ]

    result = select_adaptive_faces(person, foundation, expansion, max_count=7)

    assert len(result) == 7
    assert len({item.source.face_id for item in result}) == 7


def test_selector_requires_five_foundation_quality_faces():
    person = PersonRecord(PERSON, "Synthetic Person")
    candidates = [vector_face(i, [1.0, i * 0.01], [1.0, 0.0]) for i in range(1, 5)]

    with pytest.raises(ValueError, match="at least 5"):
        select_adaptive_faces(person, candidates, candidates)


def test_selector_rejects_invalid_vectors_and_duplicate_assets():
    person = PersonRecord(PERSON, "Synthetic Person")
    valid = [vector_face(i, [1.0, i], [1.0, i]) for i in range(1, 6)]
    bad = vector_face(6, [0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="non-zero"):
        select_adaptive_faces(person, valid, valid + [bad])

    duplicate_source = FaceCandidate(
        person_id=PERSON,
        face_id="00000000-0000-4000-8000-000000000099",
        asset_id=valid[0].source.asset_id,
        taken_at="2025-02-01T00:00:00Z",
        checksum="duplicate",
        box=(0.0, 0.0, 100.0, 100.0),
        frame=(100, 100),
    )
    duplicate = VectorFace(
        duplicate_source,
        np.array([0.0, 1.0], np.float32),
        np.array([0.0, 1.0], np.float32),
    )
    with pytest.raises(ValueError, match="one asset"):
        select_adaptive_faces(person, valid, valid + [duplicate])


def test_isolated_face_outlier_is_not_rewarded_for_novelty():
    person = PersonRecord(PERSON, "Synthetic Person")
    core = [
        vector_face(i, [1.0, (i - 4) * 0.01], [1.0, i * 0.01])
        for i in range(1, 9)
    ]
    outlier = vector_face(99, [-1.0, 0.0], [0.0, -1.0])

    result = select_adaptive_faces(person, core[:6], core + [outlier], max_count=9, novelty_distance=0.001)

    assert outlier.source.face_id not in {item.source.face_id for item in result}


def test_foundation_avoids_near_duplicates_when_core_alternatives_exist():
    person = PersonRecord(PERSON, "Synthetic Person")
    duplicates = [
        vector_face(i, [1.0, i * 0.0001], [1.0, i * 0.0001])
        for i in range(1, 6)
    ]
    alternatives = [
        vector_face(10, [0.99, 0.12], [0.9, 0.4]),
        vector_face(11, [0.98, -0.15], [0.8, -0.5]),
        vector_face(12, [0.97, 0.20], [0.7, 0.7]),
    ]
    candidates = duplicates + alternatives

    result = select_adaptive_faces(person, candidates, candidates, max_count=5)
    selected = {item.source.face_id for item in result}

    assert len(result) == 5
    assert any(item.source.face_id in selected for item in alternatives)


def test_missing_scene_embeddings_fall_back_to_face_novelty():
    person = PersonRecord(PERSON, "Synthetic Person")
    foundation = [vector_face(i, [1.0, i * 0.001], None) for i in range(1, 6)]
    extra = vector_face(6, [0.8, 0.6], None)

    result = select_adaptive_faces(person, foundation, foundation + [extra], max_count=6, novelty_distance=0.01)

    assert extra.source.face_id in {item.source.face_id for item in result}
