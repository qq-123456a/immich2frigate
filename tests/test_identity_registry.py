from __future__ import annotations

from dataclasses import dataclass

import pytest

from immich2frigate.identity_registry import PersonIdentityRegistry

PERSON = "00000000-0000-4000-8000-000000000001"
OTHER = "00000000-0000-4000-8000-000000000002"


@dataclass
class Person:
    person_id: str
    name: str


def registry(path):
    return PersonIdentityRegistry(
        path,
        immich_origin="http://immich:2283/api",
        frigate_origin="http://frigate:5000",
    )


def test_registry_persists_binding_and_keeps_sync_id_when_name_changes(tmp_path):
    store = tmp_path / "private" / "identities.json"
    first = registry(store)
    binding = first.bind(PERSON, "Old_Name")
    second = registry(store)

    assert second.immich_origin == "http://immich:2283"
    assert second.binding(PERSON) == binding
    renamed = second.bind(PERSON, "New_Name")
    assert renamed.sync_person_id == binding.sync_person_id
    assert renamed.frigate_name == "New_Name"


def test_reconcile_uses_person_id_and_bootstraps_exact_name_labels(tmp_path):
    store = registry(tmp_path / "identities.json")
    store.bind(PERSON, "Old_Name")

    actions = store.reconcile(
        [Person(PERSON, "New Name"), Person(OTHER, "Manual Label")],
        {"Old_Name": ("old.jpg",), "Manual_Label": ("manual.jpg",)},
    )

    assert actions[0]["status"] == "RENAME_REQUIRED"
    assert actions[0]["old_name"] == "Old_Name"
    assert actions[0]["name"] == "New_Name"
    assert actions[1]["status"] == "AUTO_BIND_REQUIRED"


def test_reconcile_blocks_when_both_old_and_new_labels_exist(tmp_path):
    store = registry(tmp_path / "identities.json")
    store.bind(PERSON, "Old_Name")

    actions = store.reconcile(
        [Person(PERSON, "New Name")],
        {"Old_Name": ("old.jpg",), "New_Name": ("other.jpg",)},
    )

    assert actions[0]["status"] == "RENAME_CONFLICT"


def test_registry_rejects_different_service_instances_and_invalid_names(tmp_path):
    store = registry(tmp_path / "identities.json")
    store.bind(PERSON, "Old_Name")
    with pytest.raises(ValueError, match="different service instances"):
        PersonIdentityRegistry(
            tmp_path / "identities.json",
            immich_origin="http://other-immich:2283",
            frigate_origin="http://frigate:5000",
        )
    with pytest.raises(ValueError, match="unsafe"):
        store.bind(OTHER, "../unsafe")
    with pytest.raises(ValueError, match="unsafe"):
        store.bind(OTHER, "x" * 51)


def test_reconcile_does_not_bootstrap_label_without_registered_faces(tmp_path):
    store = registry(tmp_path / "identities.json")
    actions = store.reconcile([Person(PERSON, "Existing Label")], {"Existing_Label": ()})
    assert actions[0]["status"] == "UNBOUND_PERSON"
    assert store.binding(PERSON) is None


def test_auto_bind_action_preserves_the_actual_frigate_label(tmp_path):
    store = registry(tmp_path / "identities.json")
    actions = store.reconcile([Person(PERSON, "Alex Smith")], {"alex_smith": ("face.jpg",)})
    assert actions[0]["status"] == "AUTO_BIND_REQUIRED"
    assert actions[0]["name"] == "Alex_Smith"
    assert actions[0]["frigate_name"] == "alex_smith"
