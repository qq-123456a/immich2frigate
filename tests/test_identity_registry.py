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


def test_reconcile_uses_person_id_and_does_not_claim_unmanaged_labels(tmp_path):
    store = registry(tmp_path / "identities.json")
    store.bind(PERSON, "Old_Name")

    actions = store.reconcile(
        [Person(PERSON, "New Name"), Person(OTHER, "Manual Label")],
        {"Old_Name": ("old.jpg",), "Manual_Label": ("manual.jpg",)},
    )

    assert actions[0]["status"] == "RENAME_REQUIRED"
    assert actions[0]["old_name"] == "Old_Name"
    assert actions[0]["name"] == "New_Name"
    assert actions[1]["status"] == "UNMANAGED_LABEL_EXISTS"


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


def test_adopt_requires_an_existing_reviewed_label(tmp_path):
    store = registry(tmp_path / "identities.json")
    with pytest.raises(ValueError, match="must exist"):
        store.adopt(PERSON, "Existing_Label", {})

    binding = store.adopt(PERSON, "Existing_Label", {"Existing_Label": ("face.jpg",)})
    assert binding.frigate_name == "Existing_Label"
