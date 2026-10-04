from __future__ import annotations

import json
from email.message import Message
from io import BytesIO

import pytest

from immich2frigate.immich_face_reader import ImmichApiError, ImmichFaceReader
from immich2frigate.settings import ImmichSettings


class Response(BytesIO):
    def __init__(self, body: bytes, url: str, content_type: str = "application/json"):
        super().__init__(body)
        self._url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def geturl(self) -> str:
        return self._url


class Opener:
    def __init__(self, responses: list[tuple[bytes, str, str]]):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout: float):
        self.requests.append(request)
        body, url, content_type = self.responses.pop(0)
        return Response(body, url, content_type)


def test_people_reads_immich_v3_paginated_response_and_sends_key() -> None:
    origin = "http://immich:2283"
    people = {
        "people": [
            {"id": "00000000-0000-4000-8000-000000000001", "name": "Roddy"},
            {"id": "00000000-0000-4000-8000-000000000002", "name": ""},
        ],
        "hasNextPage": False,
        "total": 2,
        "hidden": 0,
    }
    opener = Opener([(json.dumps(people).encode(), origin + "/api/people?page=1&size=500", "application/json")])
    reader = ImmichFaceReader(ImmichSettings(origin, "test-key"), opener=opener)

    result = reader.people()

    assert [person.name for person in result] == ["Roddy"]
    assert opener.requests[0].full_url.endswith("/api/people?page=1&size=500")
    assert opener.requests[0].get_header("X-api-key") == "test-key"


def test_people_paginates_until_v3_has_next_page_is_false() -> None:
    origin = "http://immich:2283"
    page1 = {"people": [{"id": "00000000-0000-4000-8000-000000000001", "name": "A"}], "hasNextPage": True}
    page2 = {"people": [{"id": "00000000-0000-4000-8000-000000000002", "name": "B"}], "hasNextPage": False}
    opener = Opener([
        (json.dumps(page1).encode(), origin + "/api/people?page=1&size=500", "application/json"),
        (json.dumps(page2).encode(), origin + "/api/people?page=2&size=500", "application/json"),
    ])

    result = ImmichFaceReader(ImmichSettings(origin, "test-key"), opener=opener).people()

    assert [person.name for person in result] == ["A", "B"]
    assert len(opener.requests) == 2


def test_people_rejects_names_that_collapse_to_same_frigate_label() -> None:
    origin = "http://immich:2283"
    people = {"people": [
        {"id": "00000000-0000-4000-8000-000000000001", "name": "A B"},
        {"id": "00000000-0000-4000-8000-000000000002", "name": "A_B"},
    ], "hasNextPage": False}
    opener = Opener([(json.dumps(people).encode(), origin + "/api/people?page=1&size=500", "application/json")])

    with pytest.raises(ImmichApiError, match="duplicate"):
        ImmichFaceReader(ImmichSettings(origin, "test-key"), opener=opener).people()


def test_person_thumbnail_is_bounded_and_signature_checked() -> None:
    origin = "http://immich:2283"
    person_id = "00000000-0000-4000-8000-000000000001"
    image = b"\xff\xd8\xffthumbnail"
    opener = Opener([(image, origin + f"/api/people/{person_id}/thumbnail", "image/jpeg")])

    result = ImmichFaceReader(ImmichSettings(origin, "test-key"), opener=opener).person_thumbnail(person_id)

    assert result == image
    assert opener.requests[0].full_url.endswith(f"/api/people/{person_id}/thumbnail")
    assert opener.requests[0].get_header("X-api-key") == "test-key"


def test_person_thumbnail_rejects_unexpected_content() -> None:
    origin = "http://immich:2283"
    person_id = "00000000-0000-4000-8000-000000000001"
    opener = Opener([(b"not an image", origin + f"/api/people/{person_id}/thumbnail", "text/plain")])

    with pytest.raises(ImmichApiError, match="thumbnail"):
        ImmichFaceReader(ImmichSettings(origin, "test-key"), opener=opener).person_thumbnail(person_id)


def test_face_reader_rejects_cross_origin_redirects() -> None:
    origin = "http://immich:2283"
    people = {"people": [], "hasNextPage": False}
    opener = Opener([(json.dumps(people).encode(), "http://other:2283/api/people?page=1&size=500", "application/json")])

    with pytest.raises(ImmichApiError, match="redirected"):
        ImmichFaceReader(ImmichSettings(origin, "test-key"), opener=opener).people()
