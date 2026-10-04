from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from immich2frigate.frigate_client import FrigateTarget
from immich2frigate.identity_registry import PersonIdentityRegistry
from immich2frigate.identity_sync import apply_person_name_sync, plan_person_name_sync

PERSON = "00000000-0000-4000-8000-000000000001"
TARGET = FrigateTarget("http://frigate:5000", "0.18.0", "large")


@dataclass
class Person:
    person_id: str
    name: str


class FakeFrigate:
    def __init__(self, faces):
        self.faces = dict(faces)
        self.rename_calls = []
        self.rename_response = {"success": True}
        self.changed_after_plan = False
        self.mutate_during_rename = False

    def verify_target(self):
        return TARGET

    def inventory(self):
        if self.changed_after_plan:
            self.changed_after_plan = False
            self.faces["Unexpected"] = ()
        return dict(self.faces)

    def rename_face(self, old_name, new_name):
        self.rename_calls.append((old_name, new_name))
        response = self.rename_response
        if response.get("success") is True:
            self.faces[new_name] = self.faces.pop(old_name)
            if self.mutate_during_rename:
                self.faces["Unexpected"] = ("other.jpg",)
        return response


def registry(path):
    value = PersonIdentityRegistry(
        path,
        immich_origin="http://immich:2283/api",
        frigate_origin=TARGET.origin,
    )
    value.bind(PERSON, "Old_Name")
    return value


def test_dry_run_and_apply_rename_preserves_files_then_updates_binding(tmp_path):
    store = registry(tmp_path / "identities.json")
    client = FakeFrigate({"Old_Name": ("1.jpg", "2.webp")})
    people = [Person(PERSON, "New Name")]
    plan = plan_person_name_sync(
        TARGET, "http://immich:2283/api", people, client.inventory(), store
    )
    assert plan.actions[0]["status"] == "RENAME_REQUIRED"
    assert plan.as_dict()["writes_enabled"] is False
    assert client.rename_calls == []

    result = apply_person_name_sync(
        plan,
        immich_origin="http://immich:2283/api",
        people=people,
        frigate_client=client,
        registry=store,
        journal_path=tmp_path / "private" / "operations.jsonl",
        backup_verified=lambda: True,
    )

    assert result.renamed_person_ids == (PERSON,)
    assert client.faces == {"New_Name": ("1.jpg", "2.webp")}
    assert store.binding(PERSON).frigate_name == "New_Name"
    journal = [
        json.loads(line)
        for line in (tmp_path / "private" / "operations.jsonl").read_text().splitlines()
    ]
    assert [row["state"] for row in journal] == ["PENDING", "VERIFIED"]


def test_apply_refuses_destination_collision_without_mutation(tmp_path):
    store = registry(tmp_path / "identities.json")
    client = FakeFrigate({"Old_Name": ("1.jpg",), "New_Name": ()})
    people = [Person(PERSON, "New Name")]
    plan = plan_person_name_sync(TARGET, "http://immich:2283", people, client.inventory(), store)

    with pytest.raises(ValueError, match="manual review|destination"):
        apply_person_name_sync(
            plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=tmp_path / "operations.jsonl",
            backup_verified=lambda: True,
        )

    assert client.rename_calls == []
    assert store.binding(PERSON).frigate_name == "Old_Name"


def test_apply_refuses_stale_inventory_and_requires_verified_backup(tmp_path):
    store = registry(tmp_path / "identities.json")
    people = [Person(PERSON, "New Name")]
    client = FakeFrigate({"Old_Name": ("1.jpg",)})
    plan = plan_person_name_sync(TARGET, "http://immich:2283", people, client.inventory(), store)
    client.changed_after_plan = True

    with pytest.raises(ValueError, match="inventory changed"):
        apply_person_name_sync(
            plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=tmp_path / "operations.jsonl",
            backup_verified=lambda: True,
        )
    assert client.rename_calls == []

    client = FakeFrigate({"Old_Name": ("1.jpg",)})
    plan = plan_person_name_sync(TARGET, "http://immich:2283", people, client.inventory(), store)
    with pytest.raises(ValueError, match="backup"):
        apply_person_name_sync(
            plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=tmp_path / "operations.jsonl",
            backup_verified=lambda: False,
        )
    assert client.rename_calls == []


def test_failed_remote_response_does_not_update_registry(tmp_path):
    store = registry(tmp_path / "identities.json")
    client = FakeFrigate({"Old_Name": ("1.jpg",)})
    client.rename_response = {"success": False}
    people = [Person(PERSON, "New Name")]
    plan = plan_person_name_sync(TARGET, "http://immich:2283", people, client.inventory(), store)

    with pytest.raises(RuntimeError, match="did not confirm"):
        apply_person_name_sync(
            plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=tmp_path / "operations.jsonl",
            backup_verified=lambda: True,
        )

    assert store.binding(PERSON).frigate_name == "Old_Name"

    recovery_plan = plan_person_name_sync(
        TARGET, "http://immich:2283", people, client.inventory(), store
    )
    with pytest.raises(ValueError, match="manual review"):
        apply_person_name_sync(
            recovery_plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=tmp_path / "operations.jsonl",
            backup_verified=lambda: True,
        )
    assert store.binding(PERSON).frigate_name == "Old_Name"
    assert client.faces == {"Old_Name": ("1.jpg",)}


def test_planner_rejects_unsupported_frigate_profile(tmp_path):
    store = registry(tmp_path / "identities.json")
    with pytest.raises(ValueError, match="requires Frigate"):
        plan_person_name_sync(
            FrigateTarget("http://frigate:5000", "0.17.0", "large"),
            "http://immich:2283",
            [Person(PERSON, "New Name")],
            {"Old_Name": ("1.jpg",)},
            store,
        )


def test_apply_requires_exact_expected_post_rename_inventory(tmp_path):
    store = registry(tmp_path / "identities.json")
    client = FakeFrigate({"Old_Name": ("1.jpg",)})
    client.mutate_during_rename = True
    people = [Person(PERSON, "New Name")]
    plan = plan_person_name_sync(TARGET, "http://immich:2283", people, client.inventory(), store)

    with pytest.raises(ValueError, match="unexpected labels"):
        apply_person_name_sync(
            plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=tmp_path / "operations.jsonl",
            backup_verified=lambda: True,
        )

    assert store.binding(PERSON).frigate_name == "Old_Name"
    recovery_plan = plan_person_name_sync(
        TARGET, "http://immich:2283", people, client.inventory(), store
    )
    with pytest.raises(ValueError, match="manual review"):
        apply_person_name_sync(
            recovery_plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=tmp_path / "operations.jsonl",
            backup_verified=lambda: True,
        )
    assert store.binding(PERSON).frigate_name == "Old_Name"


def test_recovers_when_remote_and_registry_succeeded_but_verified_log_failed(
    tmp_path, monkeypatch
):
    import immich2frigate.identity_sync as identity_sync

    store = registry(tmp_path / "identities.json")
    client = FakeFrigate({"Old_Name": ("1.jpg",)})
    people = [Person(PERSON, "New Name")]
    plan = plan_person_name_sync(TARGET, "http://immich:2283", people, client.inventory(), store)
    journal = tmp_path / "operations.jsonl"
    append = identity_sync._append_journal
    fail_once = True

    def fail_verified_once(path, entry):
        nonlocal fail_once
        if entry.get("state") == "VERIFIED" and fail_once:
            fail_once = False
            raise OSError("journal disk unavailable")
        append(path, entry)

    monkeypatch.setattr(identity_sync, "_append_journal", fail_verified_once)
    with pytest.raises(OSError, match="journal disk unavailable"):
        apply_person_name_sync(
            plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=journal,
            backup_verified=lambda: True,
        )

    recovery_plan = plan_person_name_sync(
        TARGET, "http://immich:2283", people, client.inventory(), store
    )
    result = apply_person_name_sync(
        recovery_plan,
        immich_origin="http://immich:2283",
        people=people,
        frigate_client=client,
        registry=store,
        journal_path=journal,
        backup_verified=lambda: pytest.fail("recovery must not write to Frigate"),
    )

    assert result.recovered_person_ids == (PERSON,)
    assert client.rename_calls == [("Old_Name", "New_Name")]
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert entries[-1]["state"] == "RECOVERED"


def test_sync_lock_prevents_two_writers_using_same_registry(tmp_path):
    store = registry(tmp_path / "identities.json")
    client = FakeFrigate({"Old_Name": ("1.jpg",)})
    people = [Person(PERSON, "New Name")]
    plan = plan_person_name_sync(TARGET, "http://immich:2283", people, client.inventory(), store)
    (tmp_path / "identities.json.lock").write_text("active")

    with pytest.raises(ValueError, match="another name-sync run"):
        apply_person_name_sync(
            plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=tmp_path / "operations.jsonl",
            backup_verified=lambda: True,
        )

    assert client.rename_calls == []


def test_ambiguous_write_is_journaled_and_never_retried(tmp_path):
    store = registry(tmp_path / "identities.json")
    client = FakeFrigate({"Old_Name": ("1.jpg",)})
    people = [Person(PERSON, "New Name")]
    plan = plan_person_name_sync(TARGET, "http://immich:2283", people, client.inventory(), store)
    journal = tmp_path / "operations.jsonl"

    def fail_ambiguous(old_name, new_name):
        client.rename_calls.append((old_name, new_name))
        raise TimeoutError("uncertain result")

    client.rename_face = fail_ambiguous
    with pytest.raises(TimeoutError):
        apply_person_name_sync(
            plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=journal,
            backup_verified=lambda: True,
        )

    with pytest.raises(ValueError, match="automatic retry is disabled"):
        apply_person_name_sync(
            plan,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=journal,
            backup_verified=lambda: True,
        )
    assert len(client.rename_calls) == 1


def test_reconciles_verified_remote_rename_after_registry_write_failure(tmp_path):
    store = registry(tmp_path / "identities.json")
    client = FakeFrigate({"Old_Name": ("1.jpg",)})
    people = [Person(PERSON, "New Name")]
    initial = plan_person_name_sync(TARGET, "http://immich:2283", people, client.inventory(), store)
    journal = tmp_path / "operations.jsonl"
    original_bind = store.bind

    def fail_registry_write(person_id, frigate_name):
        raise OSError("disk unavailable")

    store.bind = fail_registry_write
    with pytest.raises(OSError):
        apply_person_name_sync(
            initial,
            immich_origin="http://immich:2283",
            people=people,
            frigate_client=client,
            registry=store,
            journal_path=journal,
            backup_verified=lambda: True,
        )
    store.bind = original_bind

    recovery_plan = plan_person_name_sync(
        TARGET, "http://immich:2283", people, client.inventory(), store
    )
    result = apply_person_name_sync(
        recovery_plan,
        immich_origin="http://immich:2283",
        people=people,
        frigate_client=client,
        registry=store,
        journal_path=journal,
        backup_verified=lambda: pytest.fail("recovery must not write to Frigate"),
    )

    assert result.recovered_person_ids == (PERSON,)
    assert client.rename_calls == [("Old_Name", "New_Name")]
    assert store.binding(PERSON).frigate_name == "New_Name"
