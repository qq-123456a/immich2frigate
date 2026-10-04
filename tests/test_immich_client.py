from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from immich2frigate.immich_client import ImmichReadOnlyClient
from immich2frigate.settings import ImmichSettings

PERSON = "00000000-0000-4000-8000-000000000001"
ASSET = "00000000-0000-4000-8000-000000000002"
FACE = "00000000-0000-4000-8000-000000000003"
UPPERCASE_UUID = "abcdefab-cdef-4abc-8def-abcdefabcdef"


class FakeApi:
    def __init__(self):
        self.person_rows = []
        self.asset_rows = []
        self.faces = []
        self.photo_calls = []
        self.closed = False
        self.session = SimpleNamespace(close=self.close)

    def close(self):
        self.closed = True

    def people(self):
        return self.person_rows

    def photos(self, person_id, years):
        self.photo_calls.append((person_id, years))
        return self.asset_rows

    def target_faces(self, asset, person_id):
        return self.faces

    def image(self, asset_id, original=False):
        return np.zeros((2, 3, 3), dtype=np.uint8)


def make_client(api):
    return ImmichReadOnlyClient(
        ImmichSettings("http://immich/api", "test-only-placeholder"), api=api
    )


def test_people_returns_only_minimal_sorted_records_and_closes_session():
    api = FakeApi()
    api.person_rows = [
        {"id": "00000000-0000-4000-8000-000000000009", "name": "Zed", "assets": ["private"]},
        {"id": PERSON, "name": " Amy ", "faceCount": 12},
        {"name": "Unnamed"},
    ]
    client = make_client(api)

    people = client.people()

    assert [(row.person_id, row.name) for row in people] == [
        (PERSON, "Amy"),
        ("00000000-0000-4000-8000-000000000009", "Zed"),
    ]
    assert "private" not in repr(people)
    assert api.session.max_redirects == 0
    client.close()
    assert api.closed


def test_duplicate_person_id_fails_closed():
    api = FakeApi()
    api.person_rows = [{"id": PERSON, "name": "Amy"}, {"id": PERSON, "name": "A"}]
    with pytest.raises(ValueError, match="duplicate"):
        make_client(api).people()


@pytest.mark.parametrize("name", ["train", "../Amy", "Amy\\Bob", "Amy\nBob"])
def test_person_names_unsafe_for_frigate_fail_closed(name):
    api = FakeApi()
    api.person_rows = [{"id": PERSON, "name": name}]
    with pytest.raises(ValueError, match="Frigate face name"):
        make_client(api).people()


def test_duplicate_person_names_fail_closed_case_insensitively():
    api = FakeApi()
    api.person_rows = [
        {"id": PERSON, "name": "Amy"},
        {"id": "00000000-0000-4000-8000-000000000004", "name": "amy"},
    ]
    with pytest.raises(ValueError, match="collide"):
        make_client(api).people()


def test_person_names_that_normalize_to_same_frigate_label_fail_closed():
    api = FakeApi()
    api.person_rows = [
        {"id": PERSON, "name": "Synthetic Person"},
        {"id": "00000000-0000-4000-8000-000000000004", "name": "Synthetic_Person"},
    ]
    with pytest.raises(ValueError, match="collide"):
        make_client(api).people()


def test_candidates_keeps_valid_minimal_face_metadata_and_preview_is_memory_only():
    api = FakeApi()
    api.asset_rows = [{
        "id": ASSET,
        "fileCreatedAt": "2025-01-02T00:00:00Z",
        "checksum": "abc123",
        "originalFileName": "private.jpg",
    }]
    api.faces = [
        {"id": FACE, "boundingBoxX1": 2, "boundingBoxY1": 3,
         "boundingBoxX2": 8, "boundingBoxY2": 9, "imageWidth": 10,
         "imageHeight": 12, "personId": PERSON},
        {"id": FACE, "boundingBoxX1": 1, "boundingBoxY1": 1,
         "boundingBoxX2": 5, "boundingBoxY2": 5, "imageWidth": 10,
         "imageHeight": 12, "personId": "00000000-0000-4000-8000-000000000004"},
        {"id": FACE, "boundingBoxX1": 2, "boundingBoxY1": 3,
         "boundingBoxX2": 8, "boundingBoxY2": 9, "imageWidth": 10,
         "imageHeight": 12, "personId": PERSON},
        {"boundingBoxX1": -1, "boundingBoxY1": 0, "boundingBoxX2": 5,
         "boundingBoxY2": 5, "imageWidth": 10, "imageHeight": 10},
    ]
    client = make_client(api)

    candidates = client.candidates(PERSON, years=4)
    preview = client.preview(ASSET)

    assert len(candidates) == 1
    assert candidates[0].box == (2.0, 3.0, 8.0, 9.0)
    assert candidates[0].frame == (10, 12)
    assert "private.jpg" not in repr(candidates[0])
    assert api.photo_calls == [(PERSON, 4)]
    assert preview.shape == (2, 3, 3)
    assert preview.dtype == np.uint8
    assert preview.flags.c_contiguous


def test_invalid_ids_and_years_are_rejected_before_read():
    api = FakeApi()
    client = make_client(api)

    with pytest.raises(ValueError):
        client.candidates("../../face")
    with pytest.raises(ValueError):
        client.candidates(PERSON, years=0)
    with pytest.raises(ValueError):
        client.preview("not-a-uuid")
    assert api.photo_calls == []


def test_noncanonical_immich_ids_are_skipped_or_rejected():
    api = FakeApi()
    api.person_rows = [{"id": UPPERCASE_UUID.upper(), "name": "Amy"}]
    with pytest.raises(ValueError, match="canonical"):
        make_client(api).people()


@pytest.mark.parametrize(
    "url",
    ["file:///tmp/immich", "http://user:secret@immich", "http://immich/?key=secret", "http:///bad"],
)
def test_invalid_immich_origins_are_rejected(url):
    with pytest.raises(ValueError):
        ImmichReadOnlyClient(ImmichSettings(url, "test-only-placeholder"), api=FakeApi())
