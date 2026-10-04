from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from immich2frigate.enrollment import build_rebuild_plan, reset_registered_library
from immich2frigate.immich_client import FaceCandidate, PersonRecord
from immich2frigate.selection import VectorFace

PERSON = "00000000-0000-4000-8000-000000000001"


class FakeImmich:
    def __init__(self, count: int):
        self.person = PersonRecord(PERSON, "Synthetic Person")
        self.items = [
            FaceCandidate(
                person_id=PERSON,
                face_id=f"00000000-0000-4000-8000-{i:012d}",
                asset_id=f"10000000-0000-4000-8000-{i:012d}",
                taken_at=f"2025-01-{(i % 28) + 1:02d}T00:00:00Z",
                checksum=str(i),
                box=(0, 0, 10, 10),
                frame=(10, 10),
            )
            for i in range(1, count + 1)
        ]

    def people(self):
        return [self.person]

    def candidates(self, person_id):
        assert person_id == PERSON
        return self.items


class FakeVectors:
    def vectors_for_person(self, person, candidates):
        return [
            VectorFace(
                source=item,
                face_embedding=np.array([1.0, float(i + 1)], np.float32),
                scene_embedding=np.array([float(i + 1), 1.0], np.float32),
            )
            for i, item in enumerate(candidates)
        ]


def test_rebuild_plan_requires_full_30_before_destructive_phase():
    with pytest.raises(ValueError, match="30 are required"):
        build_rebuild_plan(FakeImmich(29), FakeVectors())

    plan = build_rebuild_plan(FakeImmich(35), FakeVectors())
    assert plan.target_per_person == 30
    assert plan.total_images == 30
    assert len(plan.people[0].candidates) == 35


class FakeFrigate:
    def __init__(self):
        self.state = {"Amy": ("a.webp", "b.webp"), "Bob": ("c.webp",)}
        self.deleted = []

    def inventory(self):
        return dict(self.state)

    def delete_faces(self, name, filenames):
        self.deleted.append((name, tuple(filenames)))
        self.state.pop(name, None)


def test_reset_deletes_explicit_inventory_and_verifies_empty():
    frigate = FakeFrigate()
    assert reset_registered_library(frigate) == 3
    assert frigate.deleted == [
        ("Amy", ("a.webp", "b.webp")),
        ("Bob", ("c.webp",)),
    ]
    assert frigate.inventory() == {}


def test_reset_fails_if_remote_inventory_remains():
    frigate = FakeFrigate()

    def bad_delete(name, filenames):
        pass

    frigate.delete_faces = bad_delete
    with pytest.raises(RuntimeError, match="empty"):
        reset_registered_library(frigate)
