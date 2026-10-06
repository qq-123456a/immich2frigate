from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from immich2frigate.frozen_plan import (
    apply_plan,
    digest,
    file_hash,
    load_plan,
    reconcile_expansion,
    save_plan,
)
from immich2frigate.frigate_client import FrigateTarget
from immich2frigate.frigate_names import frigate_face_name
from immich2frigate.identity_registry import PersonIdentityRegistry
from immich2frigate.sync_state import SyncState, write_json


def _webp(label: str) -> bytes:
    return b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + label.encode("ascii")


class FakePipeline:
    def __init__(self):
        self.profile = SimpleNamespace(metadata=lambda: {"pipeline": "synthetic-v1"})
        self.registered_by_upload = {}

    def simulate_upload(self, upload_bytes: bytes):
        return SimpleNamespace(
            stored_webp=self.registered_by_upload[upload_bytes],
            accepted_detection_count=1,
            box=(0, 0, 1, 1),
        )

    def evaluate_crop(self, _image: bytes):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1.0
        return SimpleNamespace(eligible=True, embedding=embedding)


class FakeTeacher:
    def __init__(self, asset_to_person: dict[str, str]):
        self.asset_to_person = asset_to_person
        self.reject_all = False

    def profile(self):
        return {"teacher": "synthetic-v1"}

    def confirm(self, _image: bytes, *, exclude_asset_id=None, exclude_checksum=None):
        if self.reject_all:
            return SimpleNamespace(confirmed=False, person_id=None)
        return SimpleNamespace(
            confirmed=True,
            person_id=self.asset_to_person[exclude_asset_id],
        )


class FakeImmich:
    def __init__(self, people):
        self._people = people

    def people(self):
        return list(self._people)


class FakeVectors:
    def __init__(self, sources):
        self.sources = sources

    def candidates_for_person(self, person, *, years, include_face_ids):
        assert years == 100
        return [
            SimpleNamespace(source=self.sources[face_id])
            for face_id in include_face_ids
            if self.sources[face_id].person_id == person.person_id
        ]


class FakeFrigate:
    def __init__(self, target, initial_inventory, registered_by_upload, *, failure=None):
        self._target = target
        self.state = {name: list(files) for name, files in initial_inventory.items()}
        self.blobs = {
            (name, filename): _webp(f"old-{name}-{index}")
            for name, files in self.state.items()
            for index, filename in enumerate(files)
        }
        self.registered_by_upload = registered_by_upload
        self.failure = failure
        self.delete_calls = []
        self.register_calls = []
        self._next_filename = 0

    def verify_target(self):
        return self._target

    def face_recognition_profile(self):
        return {
            "enabled": True,
            "model_size": "large",
            "recognition_threshold": 0.9,
            "min_faces": 3,
        }

    def inventory(self):
        return {name: tuple(sorted(files)) for name, files in sorted(self.state.items())}

    def face_image_bytes(self, name, filename):
        return self.blobs[(name, filename)]

    def raw_config_bytes(self):
        return b"synthetic-frigate-config"

    def delete_faces(self, name, filenames):
        self.delete_calls.append((name, tuple(filenames)))
        self.state[name] = [filename for filename in self.state.get(name, []) if filename not in filenames]
        for filename in filenames:
            self.blobs.pop((name, filename), None)
        if not self.state[name]:
            self.state.pop(name, None)
        if self.failure == "delete-timeout" and len(self.delete_calls) == 1:
            raise TimeoutError("synthetic delete timeout after commit")
        return {"success": True}

    def create_face(self, name):
        self.state.setdefault(name, [])
        # Frigate 0.18 reports false even though the folder was created.
        return {"success": False, "message": "Successfully created face folder."}

    def register_face(self, name, upload):
        self.register_calls.append((name, upload))
        self._next_filename += 1
        filename = f"{self._next_filename:032x}.webp"
        # If a hostile/stale plan file was swapped after load, model Frigate
        # storing those changed bytes so the test can observe the destructive
        # reset that would precede a later readback rejection.
        stored = self.registered_by_upload.get(upload, upload)
        if self.failure == "readback" and self._next_filename == 1:
            stored += b"changed"
        self.state.setdefault(name, []).append(filename)
        self.blobs[(name, filename)] = stored
        if self.failure == "timeout" and self._next_filename == 1:
            # Model an ambiguous write: the server committed it, then the
            # response was lost. The caller must leave the journal and stop.
            raise TimeoutError("synthetic response timeout after commit")
        return {"success": True}


def _make_case(tmp_path: Path, *, failure=None):
    target = FrigateTarget("http://frigate:5000", "0.18.0", "large")
    state = SyncState(
        tmp_path / "state.json",
        immich_origin="http://immich:2283",
        frigate_origin=target.origin,
        years=100,
    )
    registry = PersonIdentityRegistry(
        tmp_path / "registry.json",
        immich_origin="http://immich:2283/api",
        frigate_origin=target.origin,
    )
    pipeline = FakePipeline()
    person_rows = []
    people = []
    sources = {}
    asset_to_person = {}
    registered_by_upload = {}
    initial_inventory = {}
    manifest_people = []
    plan_files = {}

    for person_index in range(3):
        person_id = str(uuid4())
        person_name = f"Synthetic Person {person_index}"
        face_name = frigate_face_name(person_name)
        old_face_ids = [str(uuid4()) for _ in range(5)]
        old_files = [f"old-{person_index}-{index}.webp" for index in range(5)]
        initial_inventory[face_name] = old_files
        registry.bind(person_id, face_name)
        person_rows.append({
            "person_id": person_id,
            "name": person_name,
            "face_ids": old_face_ids,
            "examined_face_ids": old_face_ids.copy(),
            "images": [],
            "profile": {"pipeline": "synthetic-v1"},
            "enrolled_at": 1,
        })
        people.append(SimpleNamespace(person_id=person_id, name=person_name))
        image_rows = []
        for image_index in range(5):
            face_id, asset_id = str(uuid4()), str(uuid4())
            upload = _webp(f"upload-{person_index}-{image_index}")
            stored = _webp(f"registered-{person_index}-{image_index}")
            upload_file = f"{face_id}.upload.webp"
            registered_file = f"{face_id}.registered.webp"
            plan_files[upload_file] = upload
            plan_files[registered_file] = stored
            registered_by_upload[upload] = stored
            checksum = hashlib.sha256(f"source-{person_index}-{image_index}".encode()).hexdigest()
            image_rows.append({
                "person_id": person_id,
                "face_id": face_id,
                "asset_id": asset_id,
                "checksum": checksum,
                "upload_file": upload_file,
                "upload_sha256": hashlib.sha256(upload).hexdigest(),
                "registered_file": registered_file,
                "registered_sha256": hashlib.sha256(stored).hexdigest(),
                "embedding": [1.0] + [0.0] * 511,
                "metrics": {},
            })
            source = SimpleNamespace(
                person_id=person_id,
                face_id=face_id,
                asset_id=asset_id,
                checksum=checksum,
            )
            sources[face_id] = source
            asset_to_person[asset_id] = person_id
        manifest_people.append({
            "person_id": person_id,
            "name": person_name,
            "frigate_name": face_name,
            "images": image_rows,
            "examined_face_ids": [item["face_id"] for item in image_rows],
        })

    pipeline.registered_by_upload = registered_by_upload

    state.roster = person_rows
    state.save()
    teacher = FakeTeacher(asset_to_person)
    frigate = FakeFrigate(target, initial_inventory, registered_by_upload, failure=failure)
    immich = FakeImmich(people)
    vectors = FakeVectors(sources)
    manifest = {
        "schema": 1,
        "kind": "rebuild",
        "created_at": time.time(),
        "target": asdict(target),
        "immich_origin": state.immich_origin,
        "years": state.years,
        "profile": pipeline.profile.metadata(),
        "teacher": teacher.profile(),
        "frigate_face_recognition": frigate.face_recognition_profile(),
        "inventory": {name: sorted(files) for name, files in initial_inventory.items()},
        "state_sha256": file_hash(state.path),
        "registry_sha256": file_hash(registry.path),
        "people": manifest_people,
    }
    plan_dir = tmp_path / "frozen-plan"
    save_plan(plan_dir, manifest, plan_files)
    root, plan = load_plan(plan_dir)
    return {
        "root": root,
        "plan": plan,
        "state": state,
        "registry": registry,
        "frigate": frigate,
        "pipeline": pipeline,
        "teacher": teacher,
        "immich": immich,
        "vectors": vectors,
        "backup_dir": tmp_path / "backup",
        "journal": state.path.with_name(state.path.name + ".rebuild-in-progress.json"),
    }


def _apply(case):
    return apply_plan(
        case["root"],
        case["plan"],
        state=case["state"],
        registry=case["registry"],
        frigate=case["frigate"],
        pipeline=case["pipeline"],
        teacher=case["teacher"],
        immich=case["immich"],
        vectors=case["vectors"],
        backup_dir=case["backup_dir"],
    )


def _empty_expansion_case(tmp_path: Path, *, with_one_image=False):
    case = _make_case(tmp_path)
    original = case["plan"]
    original_root = case["root"]
    evidence = {
        "calibration-event": {
            "partition": "calibration",
            "category": "agree",
            "teacher_person_id": original["people"][0]["person_id"],
        }
    }
    manifest = {
        key: original[key]
        for key in (
            "schema", "target", "immich_origin", "years", "profile", "teacher",
            "frigate_face_recognition",
            "inventory", "state_sha256", "registry_sha256",
        )
    }
    manifest.update({
        "kind": "expansion",
        "validation_epoch": "frozen-epoch",
        "validation_events": evidence,
        "validation_digest": digest(evidence),
        "people": [
            {
                **person,
                "images": person["images"][:1] if with_one_image and index == 0 else [],
                "examined_face_ids": [],
            }
            for index, person in enumerate(original["people"])
        ],
    })
    files = {}
    if with_one_image:
        for kind in ("upload", "registered"):
            filename = original["people"][0]["images"][0][kind + "_file"]
            files[filename] = (original_root / filename).read_bytes()
    expansion_root = tmp_path / "empty-expansion"
    save_plan(expansion_root, manifest, files)
    case["root"], case["plan"] = load_plan(expansion_root)
    case["backup_dir"] = expansion_root / "before-apply"
    case["evidence"] = evidence
    return case


def _pending_expansion_case(tmp_path: Path):
    case = _make_case(tmp_path)
    state, frigate = case["state"], case["frigate"]
    batch_id = "b" * 64
    current_profile = {"pipeline": "after-expansion"}
    for person_index, row in enumerate(state.roster):
        name = frigate_face_name(row["name"])
        old_ids = list(row["face_ids"])
        old_records = []
        for image_index, face_id in enumerate(old_ids):
            filename = f"old-{person_index}-{image_index}.webp"
            content = frigate.face_image_bytes(name, filename)
            old_records.append({
                "face_id": face_id,
                "asset_id": str(uuid4()),
                "checksum": hashlib.sha256(f"old-source-{person_index}-{image_index}".encode()).hexdigest(),
                "filename": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
            })
        added_id, added_asset = str(uuid4()), str(uuid4())
        added_filename = f"added-{person_index}.webp"
        added_content = _webp(f"added-{person_index}")
        frigate.state[name].append(added_filename)
        frigate.blobs[(name, added_filename)] = added_content
        added = {
            "face_id": added_id,
            "asset_id": added_asset,
            "checksum": hashlib.sha256(f"new-source-{person_index}".encode()).hexdigest(),
            "filename": added_filename,
            "sha256": hashlib.sha256(added_content).hexdigest(),
        }
        previous_enrolled_at = row["enrolled_at"]
        previous_profile = row["profile"]
        previous_plan_id = row.get("plan_id", "prior-plan")
        row["plan_id"] = previous_plan_id
        row.update({
            "face_ids": [*old_ids, added_id],
            "images": [*old_records, added],
            "enrolled_at": time.time(),
            "profile": current_profile,
            "plan_id": batch_id,
            "last_batch": {
                "plan_id": batch_id,
                "status": "pending",
                "added": [added],
                "previous_enrolled_at": previous_enrolled_at,
                "previous_profile": previous_profile,
                "previous_plan_id": previous_plan_id,
            },
        })
    state.save()
    case["batch_id"] = batch_id
    return case


def _validation_report(case, events):
    from immich2frigate.expansion import _inventory_digest

    state = case["state"]
    profile = {
        "faces": {row["person_id"]: row.get("profile") for row in state.roster},
        "pipeline": case["pipeline"].profile.metadata(),
        "teacher": case["teacher"].profile(),
        "frigate_face_recognition": case["frigate"].face_recognition_profile(),
        "frigate_target": asdict(case["frigate"].verify_target()),
    }
    inventory = {name: list(files) for name, files in case["frigate"].inventory().items()}
    inventory_digest = _inventory_digest(inventory)
    enrolled = sorted(str(row.get("enrolled_at")) for row in state.roster)
    epoch_seed = json.dumps(profile, sort_keys=True, default=str, separators=(",", ":"))
    epoch = hashlib.sha256(f"{epoch_seed}|{'|'.join(enrolled)}|{inventory_digest}".encode()).hexdigest()
    return {
        "fresh": True,
        "updated_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "inventory_digest": inventory_digest,
        "epoch": epoch,
        "events": events,
    }


def test_rebuild_registers_and_verifies_exactly_fifteen_images(tmp_path, monkeypatch):
    case = _make_case(tmp_path)
    from immich2frigate import frozen_plan

    phases = []
    write_json_original = frozen_plan.write_json

    def record_journal(path, value):
        if Path(path) == case["journal"]:
            phases.append(value.get("phase"))
        return write_json_original(path, value)

    monkeypatch.setattr(frozen_plan, "write_json", record_journal)

    result = _apply(case)

    assert result["verified_uploads"] == 15
    assert result["registered_counts"] == [5, 5, 5]
    assert len(case["frigate"].register_calls) == 15
    assert case["frigate"].inventory() == {
        person["frigate_name"]: tuple(sorted(row["filename"] for row in roster_person["images"]))
        for person, roster_person in zip(case["plan"]["people"], case["state"].roster)
    }
    assert len((case["backup_dir"] / "manifest.json").read_text()) > 0
    assert not case["journal"].exists()
    assert phases.count("deleting") == phases.count("deleted") == 3
    assert phases.index("deleting") < phases.index("deleted") < phases.index("creating")


def test_frozen_hash_change_is_rejected_at_load(tmp_path):
    case = _make_case(tmp_path)
    first = case["plan"]["people"][0]["images"][0]
    (case["root"] / first["registered_file"]).write_bytes(_webp("tampered"))

    with pytest.raises(ValueError, match="frozen image bytes have changed"):
        load_plan(case["root"])


def test_unsafe_frozen_relative_path_is_rejected_even_with_recomputed_plan_id(tmp_path):
    case = _make_case(tmp_path)
    manifest_path = case["root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("plan_id")
    manifest["people"][0]["images"][0]["upload_file"] = "../../outside.webp"
    manifest["plan_id"] = digest(manifest)
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="unsafe frozen image path"):
        load_plan(case["root"])


def test_plan_image_changed_after_load_is_rejected_before_any_delete(tmp_path):
    case = _make_case(tmp_path)
    first = case["plan"]["people"][0]["images"][0]
    (case["root"] / first["upload_file"]).write_bytes(_webp("tampered-after-load"))

    with pytest.raises((RuntimeError, ValueError), match="frozen|changed"):
        _apply(case)

    assert case["frigate"].delete_calls == []
    assert case["frigate"].register_calls == []
    assert not case["journal"].exists()


def test_registered_readback_hash_mismatch_keeps_journal_and_stops(tmp_path):
    case = _make_case(tmp_path, failure="readback")

    with pytest.raises(RuntimeError, match="different crop"):
        _apply(case)

    assert len(case["frigate"].register_calls) == 1
    journal = json.loads(case["journal"].read_text())
    assert journal["phase"] == "registering"
    assert journal["pending"]["face_id"] == case["plan"]["people"][0]["images"][0]["face_id"]
    assert journal["completed"] == []


def test_ambiguous_upload_timeout_keeps_journal_without_retry(tmp_path):
    case = _make_case(tmp_path, failure="timeout")

    with pytest.raises(TimeoutError, match="after commit"):
        _apply(case)

    assert len(case["frigate"].register_calls) == 1
    journal = json.loads(case["journal"].read_text())
    assert journal["phase"] == "registering"
    assert journal["pending"]["face_id"] == case["plan"]["people"][0]["images"][0]["face_id"]
    assert journal["completed"] == []


def test_ambiguous_delete_timeout_keeps_prewrite_journal(tmp_path):
    case = _make_case(tmp_path, failure="delete-timeout")

    with pytest.raises(TimeoutError, match="delete timeout after commit"):
        _apply(case)

    assert len(case["frigate"].delete_calls) == 1
    assert case["frigate"].register_calls == []
    journal = json.loads(case["journal"].read_text())
    assert journal["phase"] == "deleting"
    assert journal["delete"]["files"] == list(case["frigate"].delete_calls[0][1])


def test_changed_state_context_fails_before_backup_or_deletion(tmp_path):
    case = _make_case(tmp_path)
    case["state"].roster[0]["name"] = "Changed Person"
    case["state"].save()

    with pytest.raises(RuntimeError, match="changed after preparation"):
        _apply(case)

    assert case["frigate"].delete_calls == []
    assert not case["backup_dir"].exists()
    assert not case["journal"].exists()


def test_changed_frigate_recognition_profile_fails_before_backup_or_deletion(tmp_path):
    case = _make_case(tmp_path)
    case["frigate"].face_recognition_profile = lambda: {
        "enabled": True,
        "model_size": "large",
        "recognition_threshold": 0.95,
        "min_faces": 3,
    }

    with pytest.raises(RuntimeError, match="recognition settings changed"):
        _apply(case)

    assert case["frigate"].delete_calls == case["frigate"].register_calls == []
    assert not case["backup_dir"].exists()
    assert not case["journal"].exists()


def test_teacher_preflight_rejection_fails_before_backup_or_remote_writes(tmp_path):
    case = _make_case(tmp_path)
    case["teacher"].reject_all = True

    with pytest.raises(RuntimeError, match="cannot confirm a frozen image"):
        _apply(case)

    assert case["frigate"].delete_calls == case["frigate"].register_calls == []
    assert not case["backup_dir"].exists()
    assert not case["journal"].exists()


def test_empty_expansion_ignores_report_timestamp_and_unrelated_events(tmp_path, monkeypatch):
    from immich2frigate import expansion

    case = _empty_expansion_case(tmp_path)
    monkeypatch.setattr(
        expansion,
        "check_validation_context",
        lambda *_args: {"epoch": "frozen-epoch", "digest": "synthetic"},
    )
    report = {
        "epoch": "frozen-epoch",
        "updated_at": "newer-report-timestamp",
        "events": {
            **case["evidence"],
            "unrelated-new-event": {"category": "agree", "partition": "calibration"},
        },
    }
    prior_state = case["state"].path.read_bytes()
    prior_roster = [dict(row) for row in case["state"].roster]

    result = apply_plan(
        case["root"], case["plan"], state=case["state"], registry=case["registry"],
        frigate=case["frigate"], pipeline=case["pipeline"], teacher=case["teacher"],
        immich=case["immich"], vectors=case["vectors"], validation_report=report,
    )

    assert result["status"] == "no_improving_candidates"
    assert case["state"].path.read_bytes() == prior_state
    assert case["state"].roster == prior_roster
    assert not case["backup_dir"].exists()
    assert not case["journal"].exists()
    assert case["frigate"].delete_calls == case["frigate"].register_calls == []


def test_changed_frozen_expansion_event_is_rejected_before_writes(tmp_path, monkeypatch):
    from immich2frigate import expansion
    from immich2frigate import validation

    case = _empty_expansion_case(tmp_path, with_one_image=True)
    monkeypatch.setattr(
        expansion,
        "check_validation_context",
        lambda *_args: {"epoch": "frozen-epoch", "digest": "synthetic"},
    )
    monkeypatch.setattr(validation, "calibration_gate", lambda *_args: {"passed": True})
    changed = dict(case["evidence"]["calibration-event"], category="conflict")
    report = {
        "epoch": "frozen-epoch",
        "updated_at": datetime.now(UTC).isoformat(),
        "events": {"calibration-event": changed},
    }
    prior_state = case["state"].path.read_bytes()

    with pytest.raises(RuntimeError, match="validation evidence changed"):
        apply_plan(
            case["root"], case["plan"], state=case["state"], registry=case["registry"],
            frigate=case["frigate"], pipeline=case["pipeline"], teacher=case["teacher"],
            immich=case["immich"], vectors=case["vectors"], validation_report=report,
        )

    assert case["state"].path.read_bytes() == prior_state
    assert not case["backup_dir"].exists()
    assert not case["journal"].exists()
    assert case["frigate"].delete_calls == case["frigate"].register_calls == []


def test_fresh_conflict_withdraws_only_added_images_and_restores_classifier_metadata(tmp_path):
    case = _pending_expansion_case(tmp_path)
    before = case["frigate"].inventory()
    report = _validation_report(case, {
        "fresh-conflict": {
            "partition": "holdout",
            "category": "conflict",
            "teacher_person_id": case["state"].roster[0]["person_id"],
        }
    })

    result = reconcile_expansion(
        state=case["state"], registry=case["registry"], frigate=case["frigate"],
        pipeline=case["pipeline"], teacher=case["teacher"], report=report,
    )

    assert result["batch_status"] == "withdrawn"
    assert case["frigate"].inventory() == {
        name: tuple(filename for filename in files if not filename.startswith("added-"))
        for name, files in before.items()
    }
    assert len(case["frigate"].delete_calls) == 3
    assert all(len(files) == 1 and files[0].startswith("added-") for _, files in case["frigate"].delete_calls)
    for row in case["state"].roster:
        assert len(row["face_ids"]) == len(row["images"]) == 5
        assert row["enrolled_at"] > 1
        assert len(row["rejected_face_ids"]) == 1
        assert row["profile"] == {"pipeline": "synthetic-v1"}
        assert row["plan_id"] == "prior-plan"
        assert "last_batch" not in row
    persisted = json.loads(case["state"].path.read_text())
    assert all("last_batch" not in row for row in persisted["roster"])
    assert not case["journal"].exists()


def test_holdout_waits_without_writes_then_accepts_only_after_all_three_gates(tmp_path):
    case = _pending_expansion_case(tmp_path)
    insufficient = {
        f"few-{index}": {
            "partition": "holdout",
            "category": "agree",
            "teacher_person_id": row["person_id"],
        }
        for index, row in enumerate(case["state"].roster)
    }
    before_state = case["state"].path.read_bytes()
    before_inventory = case["frigate"].inventory()
    waiting = reconcile_expansion(
        state=case["state"], registry=case["registry"], frigate=case["frigate"],
        pipeline=case["pipeline"], teacher=case["teacher"], report=_validation_report(case, insufficient),
    )
    assert waiting["batch_status"] == "awaiting_new_holdout"
    assert case["state"].path.read_bytes() == before_state
    assert case["frigate"].inventory() == before_inventory
    assert case["frigate"].delete_calls == []

    sufficient = {
        f"holdout-{person_index}-{event_index}": {
            "partition": "holdout",
            "category": "agree",
            "teacher_person_id": row["person_id"],
        }
        for person_index, row in enumerate(case["state"].roster)
        for event_index in range(20)
    }
    accepted = reconcile_expansion(
        state=case["state"], registry=case["registry"], frigate=case["frigate"],
        pipeline=case["pipeline"], teacher=case["teacher"], report=_validation_report(case, sufficient),
    )

    assert accepted["batch_status"] == "accepted"
    assert all(row["last_batch"]["status"] == "accepted" for row in case["state"].roster)
    assert case["frigate"].inventory() == before_inventory
    assert case["frigate"].delete_calls == []
