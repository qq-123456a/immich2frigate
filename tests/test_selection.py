from __future__ import annotations

import numpy as np
import pytest

from immich2frigate.immich_client import FaceCandidate, PersonRecord
from immich2frigate.selection import VectorFace, select_representative_faces

PERSON = "00000000-0000-4000-8000-000000000001"


def vector_face(index: int, face, scene) -> VectorFace:
    source = FaceCandidate(
        person_id=PERSON,
        face_id=f"00000000-0000-4000-8000-{index:012d}",
        asset_id=f"10000000-0000-4000-8000-{index:012d}",
        taken_at=f"2025-01-{index:02d}T00:00:00Z",
        checksum=f"checksum-{index}",
        box=(0.0, 0.0, 10.0, 10.0),
        frame=(10, 10),
    )
    return VectorFace(
        source=source,
        face_embedding=np.asarray(face, dtype=np.float32),
        scene_embedding=np.asarray(scene, dtype=np.float32),
    )


def test_selector_returns_requested_count_and_is_deterministic():
    person = PersonRecord(PERSON, "Synthetic Person")
    candidates = [
        vector_face(1, [1.0, 0.0], [1.0, 0.0]),
        vector_face(2, [0.9, 0.1], [1.0, 0.0]),
        vector_face(3, [0.0, 1.0], [0.0, 1.0]),
        vector_face(4, [-1.0, 0.0], [0.0, -1.0]),
    ]

    first = select_representative_faces(person, list(reversed(candidates)), count=3)
    second = select_representative_faces(person, candidates, count=3)

    assert [item.source.face_id for item in first] == [item.source.face_id for item in second]
    assert len(first) == 3
    assert len({item.source.face_id for item in first}) == 3


def test_selector_caps_at_available_candidates():
    person = PersonRecord(PERSON, "Synthetic Person")
    candidates = [
        vector_face(1, [1.0, 0.0], [1.0, 0.0]),
        vector_face(2, [0.0, 1.0], [0.0, 1.0]),
    ]

    assert len(select_representative_faces(person, candidates, count=30)) == 2


def test_selector_rejects_invalid_count_and_vectors():
    person = PersonRecord(PERSON, "Synthetic Person")
    valid = vector_face(1, [1.0, 0.0], [1.0, 0.0])

    with pytest.raises(ValueError, match="positive"):
        select_representative_faces(person, [valid], count=0)

    bad = vector_face(2, [0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="non-zero"):
        select_representative_faces(person, [bad])

    mismatch = vector_face(3, [1.0, 0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="same dimension"):
        select_representative_faces(person, [valid, mismatch])


def test_selector_rejects_duplicate_assets_and_wrong_person():
    person = PersonRecord(PERSON, "Synthetic Person")
    one = vector_face(1, [1.0, 0.0], [1.0, 0.0])
    two = vector_face(2, [0.0, 1.0], [0.0, 1.0])
    duplicate_asset_source = FaceCandidate(
        person_id=PERSON,
        face_id=two.source.face_id,
        asset_id=one.source.asset_id,
        taken_at=two.source.taken_at,
        checksum=two.source.checksum,
        box=two.source.box,
        frame=two.source.frame,
    )
    duplicate_asset = VectorFace(
        duplicate_asset_source,
        two.face_embedding,
        two.scene_embedding,
    )
    with pytest.raises(ValueError, match="one asset"):
        select_representative_faces(person, [one, duplicate_asset])

    wrong_source = FaceCandidate(
        person_id="00000000-0000-4000-8000-000000000099",
        face_id="00000000-0000-4000-8000-000000000098",
        asset_id="10000000-0000-4000-8000-000000000098",
        taken_at="2025-02-01T00:00:00Z",
        checksum="wrong",
        box=(0.0, 0.0, 10.0, 10.0),
        frame=(10, 10),
    )
    wrong = VectorFace(wrong_source, np.ones(2, np.float32), np.ones(2, np.float32))
    with pytest.raises(ValueError, match="different Immich person"):
        select_representative_faces(person, [wrong])
