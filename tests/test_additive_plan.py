from __future__ import annotations

import hashlib
import pytest
import numpy as np
from types import SimpleNamespace

from immich2frigate.additive_plan import (
    _append_locked_ledger,
    _select_additions,
    additions_needed,
    apply_additive_plan,
    prepare_additive,
)
from immich2frigate.frozen_plan import _external_identities


@pytest.mark.parametrize(
    ("current", "limit", "expected"),
    [(0, 20, 20), (4, 20, 16), (20, 20, 0), (23, 20, 0)],
)
def test_additions_needed_respects_requested_total(current, limit, expected):
    assert additions_needed(current, limit) == expected


@pytest.mark.parametrize(("current", "limit"), [(-1, 20), (0, 0), (True, 20), (0, 31)])
def test_additions_needed_rejects_invalid_counts(current, limit):
    with pytest.raises(ValueError):
        additions_needed(current, limit)


def test_select_additions_uses_four_batches_and_drops_cross_batch_duplicates(monkeypatch):
    from immich2frigate import additive_plan

    candidates = []
    for index in range(22):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[index if index != 5 else 0] = 1
        source = SimpleNamespace(source=SimpleNamespace(face_id=str(index)))
        candidates.append(SimpleNamespace(source=source, face=SimpleNamespace(embedding=embedding)))
    calls = []

    def select_five(_person, remaining, **_kwargs):
        calls.append(len(remaining))
        return remaining[:5]

    monkeypatch.setattr(additive_plan, "select_foundation", select_five)
    selected = _select_additions(SimpleNamespace(name="Example Person A"), candidates, 20, None, {})

    assert len(selected) == 20
    assert len({item.source.source.face_id for item in selected}) == 20
    assert len(calls) == 4


def test_append_locked_ledger_tracks_new_files_for_the_existing_identity():
    row = {
        "person_id": "00000000-0000-4000-8000-000000000001",
        "face_ids": ["00000000-0000-4000-8000-000000000011"],
        "examined_face_ids": [],
        "images": [{
            "face_id": "00000000-0000-4000-8000-000000000011",
            "asset_id": "00000000-0000-4000-8000-000000000021",
            "checksum": "old", "filename": "old.webp", "sha256": "old",
        }],
    }

    class State:
        roster = [row]

        def save(self):
            self.saved = True

    state = State()
    record = {
        "person_id": row["person_id"], "name": "Example Person",
        "face_id": "00000000-0000-4000-8000-000000000012",
        "asset_id": "00000000-0000-4000-8000-000000000022",
        "checksum": "new", "filename": "new.webp", "sha256": "new",
    }

    _append_locked_ledger(state, [record])

    assert state.saved
    assert state.roster[0]["face_ids"] == [row["face_ids"][0], record["face_id"]]
    assert state.roster[0]["images"][-1] == {
        key: record[key] for key in ("face_id", "asset_id", "checksum", "filename", "sha256")
    }


def test_append_locked_ledger_ignores_supplemental_identity():
    state = SimpleNamespace(roster=[{"person_id": "locked", "face_ids": ["old"], "images": []}])
    state.saved = False
    state.save = lambda: setattr(state, "saved", True)

    _append_locked_ledger(state, [{
        "person_id": "supplemental", "face_id": "new", "asset_id": "asset",
        "checksum": "sum", "filename": "new.webp", "sha256": "hash",
    }])

    assert not state.saved
    assert state.roster == [{"person_id": "locked", "face_ids": ["old"], "images": []}]


def test_external_identities_preserves_only_bound_supplemental_labels():
    locked_id = "00000000-0000-4000-8000-000000000001"
    external_id = "00000000-0000-4000-8000-000000000002"
    people = [SimpleNamespace(person_id=external_id, name="Example Person")]
    bindings = {
        locked_id: SimpleNamespace(frigate_name="Locked_Person"),
        external_id: SimpleNamespace(frigate_name="Example_Person"),
    }
    state = SimpleNamespace(roster=[{"person_id": locked_id}])
    registry = SimpleNamespace(binding=bindings.get)
    frigate = SimpleNamespace(face_image_bytes=lambda _name, _filename: b"stored face")

    preserved, external, hashes = _external_identities(
        state,
        registry,
        people,
        {"Locked_Person": ("locked.webp",), "Example_Person": ("face.webp",)},
        frigate,
    )

    assert preserved == {"Example_Person": ["face.webp"]}
    assert external == {external_id: people[0]}
    assert hashes["Example_Person"]["face.webp"] == hashlib.sha256(b"stored face").hexdigest()


def test_external_identities_rejects_unbound_supplemental_label():
    state = SimpleNamespace(roster=[{"person_id": "locked"}])
    registry = SimpleNamespace(binding=lambda _person_id: None)
    person = SimpleNamespace(person_id="outside", name="Example Person")
    frigate = SimpleNamespace(face_image_bytes=lambda _name, _filename: b"stored face")

    with pytest.raises(RuntimeError, match="verified Immich identity binding"):
        _external_identities(state, registry, [person], {"Example_Person": ("face.webp",)}, frigate)


@pytest.mark.parametrize(
    ("face_ids", "images", "message"),
    [
        (["locked-face"], [{"face_id": "different-face", "filename": "current.webp"}], "source ledger is incomplete"),
        (["locked-face"], [{"face_id": "locked-face", "filename": "stale.webp"}], "filenames differ"),
    ],
)
def test_prepare_additive_rejects_locked_ledger_mismatch_before_scanning(
    tmp_path, monkeypatch, face_ids, images, message
):
    from immich2frigate import additive_plan
    from immich2frigate.frigate_client import FrigateTarget

    person_id = "00000000-0000-4000-8000-000000000001"
    person = SimpleNamespace(person_id=person_id, name="Example Person")
    state_path = tmp_path / "state.json"
    registry_path = tmp_path / "registry.json"
    state_path.write_text("{}")
    registry_path.write_text("{}")
    state = SimpleNamespace(
        path=state_path,
        pending=None,
        roster=[{"person_id": person_id, "face_ids": face_ids, "images": images}],
        immich_origin="http://immich",
        years=100,
    )
    binding = SimpleNamespace(frigate_name="Example_Person")
    registry = SimpleNamespace(
        path=registry_path,
        binding=lambda _person_id: binding,
        reconcile=lambda *_args: [{"person_id": person_id, "status": "MATCHED"}],
        preflight_bindings=lambda _bindings: None,
    )

    class Frigate:
        def verify_target(self):
            return FrigateTarget("http://frigate", "0.18.0", "large")

        def face_recognition_profile(self):
            return {"enabled": True}

        def inventory(self):
            return {"Example_Person": ("current.webp",)}

    immich = SimpleNamespace(people=lambda: [person])
    monkeypatch.setattr(
        additive_plan,
        "scan_person",
        lambda *_args, **_kwargs: pytest.fail("preflight must reject before scanning"),
    )

    with pytest.raises(RuntimeError, match=message):
        prepare_additive(
            tmp_path / "plan",
            target_names=[person.name],
            max_images=2,
            state=state,
            registry=registry,
            frigate=Frigate(),
            immich=immich,
            vectors=object(),
            pipeline=SimpleNamespace(profile=SimpleNamespace(metadata=lambda: {})),
            teacher=SimpleNamespace(profile=lambda: {}),
        )

    assert not (tmp_path / "plan").exists()


@pytest.mark.parametrize("operation", ["prepare", "apply"])
def test_additive_enrollment_waits_for_pending_expansion_validation(
    tmp_path, monkeypatch, operation
):
    from immich2frigate import additive_plan

    state = SimpleNamespace(
        path=tmp_path / "state.json",
        pending=None,
        roster=[{"last_batch": {"status": "pending"}}],
    )
    state.path.write_text("{}")

    if operation == "prepare":
        with pytest.raises(RuntimeError, match="expansion batch is still awaiting validation"):
            prepare_additive(
                tmp_path / "plan",
                target_names=["Example Person"],
                max_images=20,
                state=state,
                registry=None,
                frigate=None,
                immich=None,
                vectors=None,
                pipeline=None,
                teacher=None,
            )
    else:
        monkeypatch.setattr(additive_plan, "load_additive_plan", lambda root: (root, {}))
        with pytest.raises(RuntimeError, match="expansion batch is still awaiting validation"):
            apply_additive_plan(
                tmp_path / "plan",
                backup_dir=tmp_path / "backup",
                state=state,
                registry=None,
                frigate=None,
                immich=None,
                vectors=None,
                pipeline=None,
                teacher=None,
            )

    assert not (tmp_path / "backup").exists()
