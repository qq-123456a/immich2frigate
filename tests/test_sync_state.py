import pytest

from immich2frigate.sync_state import SyncState
from immich2frigate.cli import _recover_pending_sync


PERSON_IDS = [
    "00000000-0000-4000-8000-000000000001",
    "00000000-0000-4000-8000-000000000002",
    "00000000-0000-4000-8000-000000000003",
]
FACE_IDS = [
    [f"10000000-0000-4000-8000-{index:012d}" for index in range(start, start + 5)]
    for start in (1, 6, 11)
]


def test_sync_state_round_trips_locked_roster_and_pending_upload(tmp_path):
    path = tmp_path / "sync.json"
    state = SyncState(path, immich_origin="http://immich:2283", frigate_origin="http://frigate:5000")
    state.roster = [
        {"person_id": person_id, "name": f"Person {index}", "face_ids": faces, "examined_face_ids": faces.copy()}
        for index, (person_id, faces) in enumerate(zip(PERSON_IDS, FACE_IDS), start=1)
    ]
    state.pending = {"person_id": PERSON_IDS[0], "face_id": "10000000-0000-4000-8000-000000000099", "before_count": 5}
    state.save()

    loaded = SyncState(path, immich_origin="http://immich:2283", frigate_origin="http://frigate:5000")

    assert loaded.roster == state.roster
    assert loaded.pending == state.pending


def test_sync_state_rejects_different_service_origins(tmp_path):
    path = tmp_path / "sync.json"
    state = SyncState(path, immich_origin="http://immich:2283", frigate_origin="http://frigate:5000")
    state.roster = [
        {"person_id": person_id, "name": f"Person {index}", "face_ids": faces, "examined_face_ids": faces.copy()}
        for index, (person_id, faces) in enumerate(zip(PERSON_IDS, FACE_IDS), start=1)
    ]
    state.save()

    with pytest.raises(ValueError, match="different service instances"):
        SyncState(path, immich_origin="http://other-immich:2283", frigate_origin="http://frigate:5000")


def test_sync_state_rejects_changed_year_window(tmp_path):
    path = tmp_path / "sync.json"
    state = SyncState(path, immich_origin="http://immich:2283", frigate_origin="http://frigate:5000")
    state.roster = [
        {"person_id": person_id, "name": f"Person {index}", "face_ids": faces, "examined_face_ids": faces.copy()}
        for index, (person_id, faces) in enumerate(zip(PERSON_IDS, FACE_IDS), start=1)
    ]
    state.save()

    with pytest.raises(ValueError, match="different service instances"):
        SyncState(
            path,
            immich_origin="http://immich:2283",
            frigate_origin="http://frigate:5000",
            years=20,
        )


def test_pending_upload_fails_closed_without_clearing_journal(tmp_path):
    state = SyncState(
        tmp_path / "sync.json",
        immich_origin="http://immich:2283",
        frigate_origin="http://frigate:5000",
    )
    state.pending = {
        "person_id": PERSON_IDS[0],
        "face_id": "10000000-0000-4000-8000-000000000099",
        "before_count": 5,
    }

    with pytest.raises(ValueError, match="cannot be verified from image counts"):
        _recover_pending_sync(state)

    assert state.pending is not None
