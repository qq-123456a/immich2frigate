from io import BytesIO

import pytest

from immich2frigate.frigate_client import FrigateApiError, FrigateReadOnlyClient
from immich2frigate.settings import FrigateSettings


class FakeResponse(BytesIO):
    def __init__(self, body: bytes, url: str):
        super().__init__(body)
        self._url = url

    def geturl(self) -> str:
        return self._url


class FakeOpener:
    def __init__(self, responses: list[bytes]):
        self.responses = list(responses)
        self.requests: list[tuple[str, str]] = []

    def open(self, request, timeout: float):
        self.requests.append((request.method, request.full_url))
        return FakeResponse(self.responses.pop(0), request.full_url)


def client(opener: FakeOpener) -> FrigateReadOnlyClient:
    return FrigateReadOnlyClient(FrigateSettings("http://frigate:5000"), opener=opener)


def test_target_preflight_checks_version_and_model_with_get_only_requests() -> None:
    opener = FakeOpener(
        [
            b"0.18.0\n",
            b'{"username":"anonymous","role":"admin","allowed_cameras":[]}',
            b'{"face_recognition":{"enabled":true,"model_size":"large"},"environment_vars":{"SECRET":"never retain"}}',
        ]
    )

    client(opener).verify_target()

    assert opener.requests == [
        ("GET", "http://frigate:5000/api/version"),
        ("GET", "http://frigate:5000/api/profile"),
        ("GET", "http://frigate:5000/api/config"),
    ]


def test_target_preflight_rejects_non_admin_and_disabled_recognition() -> None:
    with pytest.raises(FrigateApiError, match="admin role"):
        client(
            FakeOpener(
                [b"0.18.0", b'{"username":"viewer","role":"viewer"}']
            )
        ).verify_target()
    with pytest.raises(FrigateApiError, match="not enabled"):
        client(
            FakeOpener(
                [
                    b"0.18.0",
                    b'{"username":"anonymous","role":"admin"}',
                    b'{"face_recognition":{"model_size":"large","enabled":false}}',
                ]
            )
        ).verify_target()


def test_face_listing_sorts_files_and_names() -> None:
    opener = FakeOpener([b'{"Zed":["b.webp","a.jpg"],"train":["tmp.jpg"],"Amy":["face.png"]}'])

    result = client(opener).faces()

    assert result == {"Amy": ("face.png",), "Zed": ("a.jpg", "b.webp")}


def test_version_response_requires_utf8():
    with pytest.raises(FrigateApiError, match="UTF-8"):
        client(FakeOpener([b"\xff"])).version()


@pytest.mark.parametrize(
    "payload",
    [b'{"../secret":["face.jpg"]}', b'{"Roddy":["../face.jpg"]}', b'{"Roddy":["face.txt"]}'],
)
def test_face_listing_rejects_unsafe_names_and_files(payload: bytes) -> None:
    with pytest.raises(FrigateApiError):
        client(FakeOpener([payload])).faces()


def test_client_rejects_base_paths_and_embedded_credentials() -> None:
    with pytest.raises(ValueError):
        FrigateReadOnlyClient(FrigateSettings("http://frigate:5000/prefix"))
    with pytest.raises(ValueError):
        FrigateReadOnlyClient(FrigateSettings("http://user:secret@frigate:5000"))


def test_registered_face_image_is_quoted_and_signature_checked() -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"synthetic"
    opener = FakeOpener([png])

    result = client(opener).face_image_bytes("A B", "face 1.png")

    assert result == png
    assert opener.requests == [
        ("GET", "http://frigate:5000/clips/faces/A%20B/face%201.png")
    ]
    with pytest.raises(FrigateApiError, match="format"):
        client(FakeOpener([b"not an image"])).face_image_bytes("Amy", "face.jpg")
    with pytest.raises(ValueError):
        client(FakeOpener([])).face_image_bytes("../Amy", "face.jpg")
