from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from immich2frigate.enrollment import (
    apply_rebuild_plan,
    backup_registered_library,
    build_incremental_plan,
    build_rebuild_plan,
    RebuildPlan,
    reset_registered_library,
)
from immich2frigate.identity_registry import PersonIdentityRegistry
from immich2frigate.immich_client import FaceCandidate, PersonRecord
from immich2frigate.selection import VectorFace

PERSON = "00000000-0000-4000-8000-000000000001"


def clear_color_image() -> np.ndarray:
    image = np.empty((100, 100, 3), dtype=np.uint8)
    image[::2, ::2] = (30, 180, 240)
    image[1::2, ::2] = (220, 40, 80)
    image[::2, 1::2] = (220, 40, 80)
    image[1::2, 1::2] = (30, 180, 240)
    return image


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
                box=(0, 0, 100, 100),
                frame=(100, 100),
            )
            for i in range(1, count + 1)
        ]
        self.image = clear_color_image()

    def people(self):
        return [self.person]

    def candidates(self, person_id, years=100):
        assert person_id == PERSON
        assert 1 <= years <= 100
        return self.items

    def preview(self, asset_id):
        return self.image.copy()


class FakeVectors:
    def __init__(self, diverse: bool = False):
        self.diverse = diverse

    def vectors_for_person(self, person, candidates):
        rows = []
        for i, item in enumerate(candidates):
            if self.diverse and i >= 5:
                # Add meaningful variation while remaining inside the same
                # synthetic identity core. Extreme opposite vectors are identity
                # outliers and should no longer be treated as useful diversity.
                face_angle = (i - 4) * 0.18
                scene_angle = (i - 4) * 0.65
                face = np.array([np.cos(face_angle), np.sin(face_angle)], np.float32)
                scene = np.array([np.cos(scene_angle), np.sin(scene_angle)], np.float32)
            else:
                face = np.array([1.0, (i + 1) * 0.001], np.float32)
                scene = np.array([1.0, (i + 1) * 0.001], np.float32)
            rows.append(VectorFace(item, face, scene))
        return rows


def test_rebuild_plan_requires_only_five_foundation_quality_faces():
    with pytest.raises(ValueError, match="5 are required"):
        build_rebuild_plan(FakeImmich(4), FakeVectors())

    plan = build_rebuild_plan(FakeImmich(20), FakeVectors())
    assert plan.total_images == 5
    assert plan.people[0].target_count == 5
    assert len(plan.people[0].examined_face_ids) == 20


def test_rebuild_plan_expands_when_distribution_is_genuinely_diverse():
    plan = build_rebuild_plan(FakeImmich(12), FakeVectors(diverse=True))
    assert 5 < plan.people[0].target_count <= 30


def test_incremental_plan_returns_examined_ids_after_preparation():
    immich = FakeImmich(10)
    known = [item.face_id for item in immich.items[:5]]

    plan, examined = build_incremental_plan(
        immich, FakeVectors(), immich.person, known, known
    )

    assert plan.examined_face_ids == examined
    assert set(known) <= set(examined)


def test_incremental_plan_keeps_one_largest_face_per_asset():
    immich = FakeImmich(12)
    duplicate = immich.items[6]
    immich.items[6] = replace(duplicate, asset_id=immich.items[5].asset_id)
    known = [item.face_id for item in immich.items[:5]]

    plan, examined = build_incremental_plan(
        immich, FakeVectors(diverse=True), immich.person, known, known
    )

    selected_assets = [item.source.asset_id for item in plan.candidates]
    assert len(selected_assets) == len(set(selected_assets))
    assert duplicate.face_id in examined


def test_apply_uses_preflighted_uploads_without_refetching_immich(tmp_path):
    immich = FakeImmich(5)
    plan = build_rebuild_plan(immich, FakeVectors())
    immich.preview = lambda asset_id: (_ for _ in ()).throw(AssertionError("unexpected preview refetch"))
    frigate = FakeFrigate()
    registry = PersonIdentityRegistry(
        tmp_path / "identities.json",
        immich_origin="http://immich:2283/api",
        frigate_origin="http://frigate:5000",
    )

    result = apply_rebuild_plan(plan, immich, frigate, registry=registry)

    assert result.registered_images == 5
    assert registry.binding(PERSON).frigate_name == "Synthetic_Person"


def test_apply_rejects_fewer_than_five_faces_before_touching_frigate():
    plan = build_rebuild_plan(FakeImmich(5), FakeVectors())
    too_small = replace(
        plan.people[0],
        candidates=plan.people[0].candidates[:4],
        uploads=plan.people[0].uploads[:4],
    )
    frigate = FakeFrigate()

    with pytest.raises(ValueError, match="invalid preflighted uploads"):
        apply_rebuild_plan(RebuildPlan((too_small,)), FakeImmich(5), frigate)

    assert frigate.deleted == []


class FakeFrigate:
    def __init__(self):
        self.state = {"Amy": ("a.webp", "b.webp"), "Bob": ("c.webp",)}
        self.deleted = []

    def inventory(self):
        return dict(self.state)

    def delete_faces(self, name, filenames):
        self.deleted.append((name, tuple(filenames)))
        self.state.pop(name, None)

    def face_image_bytes(self, name, filename):
        return f"{name}/{filename}".encode()

    def create_face(self, name):
        self.state.setdefault(name, ())

    def register_face(self, name, image_bytes):
        assert image_bytes.startswith(b"RIFF")
        self.state[name] = (*self.state.get(name, ()), f"{len(image_bytes)}.webp")
        return {"success": True}


def test_backup_inventory_must_match_before_reset(tmp_path):
    frigate = FakeFrigate()
    manifest = backup_registered_library(frigate, tmp_path)
    frigate.state["New person"] = ("new.webp",)

    with pytest.raises(RuntimeError, match="changed after backup"):
        reset_registered_library(frigate, expected_inventory=manifest["inventory"])

    assert frigate.deleted == []


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
