from __future__ import annotations

import json
from io import BytesIO

import pytest

from immich2frigate.frigate_client import FrigateApiError
from immich2frigate.frigate_write import FrigateWriteClient
from immich2frigate.settings import FrigateSettings


class FakeResponse(BytesIO):
    def __init__(self, body: bytes, url: str):
        super().__init__(body)
        self._url = url

    def geturl(self) -> str:
        return self._url


class FakeOpener:
    def __init__(self, responses: list[bytes | Exception], urls: list[str] | None = None):
        self.responses = list(responses)
        self.urls = list(urls or [])
        self.requests = []

    def open(self, request, timeout: float):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        url = self.urls.pop(0) if self.urls else request.full_url
        return FakeResponse(response, url)


def client(opener: FakeOpener) -> FrigateWriteClient:
    return FrigateWriteClient(FrigateSettings("http://frigate:5000"), opener=opener)


def test_inventory_uses_the_read_only_allow_list_and_omits_train() -> None:
    opener = FakeOpener([b'{"Zed":["b.webp","a.jpg"],"train":["tmp.jpg"]}'])

    assert client(opener).inventory() == {"Zed": ("a.jpg", "b.webp")}
    assert [(request.method, request.full_url) for request in opener.requests] == [
        ("GET", "http://frigate:5000/api/faces")
    ]


def test_create_face_is_fixed_post_with_quoted_name() -> None:
    opener = FakeOpener([b'{"success":false,"message":"Successfully created face folder."}'])

    result = client(opener).create_face("A B")

    request = opener.requests[0]
    assert result["success"] is False  # Frigate 0.18 has this response quirk.
    assert request.method == "POST"
    assert request.full_url == "http://frigate:5000/api/faces/A%20B/create"
    assert request.data == b""


def test_register_face_sends_bounded_multipart_image() -> None:
    jpeg = b"\xff\xd8\xff" + b"pixels"
    opener = FakeOpener([b'{"success":true,"message":"ok"}'])

    result = client(opener).register_face("Amy", jpeg)

    request = opener.requests[0]
    content_type = request.headers["Content-type"]
    assert result["success"] is True
    assert request.method == "POST"
    assert request.full_url == "http://frigate:5000/api/faces/Amy/register"
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="file"; filename="face.jpg"' in request.data
    assert jpeg in request.data


def test_delete_faces_serializes_only_valid_explicit_ids() -> None:
    opener = FakeOpener([b'{"success":true}'])

    client(opener).delete_faces("Amy", ["a.jpg", "b.webp"])

    request = opener.requests[0]
    assert request.method == "POST"
    assert request.full_url == "http://frigate:5000/api/faces/Amy/delete"
    assert json.loads(request.data) == {"ids": ["a.jpg", "b.webp"]}
    assert request.headers["Content-type"] == "application/json"


def test_recognize_uses_the_same_fixed_multipart_contract() -> None:
    png = b"\x89PNG\r\n\x1a\nimage"
    opener = FakeOpener([b'{"success":true,"face_name":"Amy"}'])

    result = client(opener).recognize(png)

    request = opener.requests[0]
    assert result["face_name"] == "Amy"
    assert request.method == "POST"
    assert request.full_url == "http://frigate:5000/api/faces/recognize"
    assert b'filename="face.png"' in request.data


@pytest.mark.parametrize(
    "operation",
    [
        lambda c: c.create_face("train"),
        lambda c: c.create_face("../Amy"),
        lambda c: c.register_face("train", b"\xff\xd8\xffx"),
        lambda c: c.delete_faces("Amy", []),
        lambda c: c.delete_faces("Amy", ["../face.jpg"]),
        lambda c: c.delete_faces("Amy", ["face.txt"]),
        lambda c: c.register_face("Amy", b"not an image"),
    ],
)
def test_write_paths_reject_unsafe_or_unbounded_inputs(operation) -> None:
    with pytest.raises((ValueError, TypeError)):
        operation(client(FakeOpener([])))


def test_write_client_rejects_redirects_and_oversized_responses() -> None:
    opener = FakeOpener([b"{}"], urls=["http://other-host:5000/api/faces/Amy/create"])
    with pytest.raises(FrigateApiError, match="redirected"):
        client(opener).create_face("Amy")

    opener = FakeOpener([b"x" * (2 * 1024 * 1024 + 1)])
    with pytest.raises(FrigateApiError, match="size limit"):
        client(opener).create_face("Amy")
