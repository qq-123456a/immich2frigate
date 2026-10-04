"""Small, allow-listed write client for Frigate 0.18 face registration.

The read-only adapter remains the source of truth for inventory and target
preflight.  This module adds only the five operations needed by the enrollment
workflow: inventory, create a face name, register one image, delete explicitly
listed images, and recognize one image.  There is deliberately no arbitrary
request method and no operation for Frigate's ``train`` staging directory.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request

from .frigate_client import (
    _MAX_RESPONSE_BYTES,
    _matches_image_signature,
    _require_face_filename,
    _require_face_segment,
    FrigateApiError,
    FrigateReadOnlyClient,
)
from .frigate_names import frigate_face_name


# Face crops produced by the enrollment pipeline are normally a few hundred KB.
# Keep a generous cap for a caller-provided preview while bounding memory and
# request size if an upstream service ever returns an unexpected object.
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024
_MAX_FACE_NAME_LENGTH = 128
_MAX_FACE_ID_LENGTH = 255
_MAX_DELETE_IDS = 512


class FrigateWriteClient(FrigateReadOnlyClient):
    """Allow-listed Frigate face-library writes on the configured origin.

    Frigate's internal API grants administrator-equivalent access.  Callers
    must preflight with :meth:`verify_target` and keep this client on the
    trusted internal API port.  Redirects are disabled and responses are
    bounded; request/response data is never logged here.
    """

    def inventory(self) -> dict[str, tuple[str, ...]]:
        """Return current registered names and image filenames.

        The inherited read path intentionally omits Frigate's ``train``
        staging directory, so this method cannot accidentally expose it as a
        writable target.
        """

        return self.faces()

    def create_face(self, name: str) -> dict[str, object]:
        """Create a face folder for ``name`` and return Frigate's response."""

        name = _require_write_name(name)
        body = self._post(
            f"/api/faces/{quote(name, safe='')}/create",
            b"",
            headers={"Accept": "application/json"},
        )
        return _json_object(body, "create face response")

    def register_face(self, name: str, image_bytes: bytes) -> dict[str, object]:
        """Upload one bounded image to an existing Frigate face name."""

        name = _require_write_name(name)
        payload, extension, content_type = _require_upload(image_bytes)
        body, request_headers = _multipart_body(payload, extension, content_type)
        response = self._post(
            f"/api/faces/{quote(name, safe='')}/register",
            body,
            headers={"Accept": "application/json", **request_headers},
        )
        return _json_object(response, "register face response")

    def delete_faces(
        self,
        name: str,
        image_ids: Sequence[str],
    ) -> dict[str, object]:
        """Delete exactly the supplied image IDs from one registered name.

        ``train`` is rejected as a name, IDs must be supported face image
        filenames, and an empty list is rejected so a caller cannot mistake a
        no-op for a verified deletion.
        """

        name = _require_write_name(name)
        if isinstance(image_ids, (str, bytes, bytearray)) or not isinstance(
            image_ids, Sequence
        ):
            raise ValueError("image_ids must be a sequence of filenames")
        if not image_ids:
            raise ValueError("image_ids must not be empty")
        if len(image_ids) > _MAX_DELETE_IDS:
            raise ValueError("too many face image IDs")

        validated: list[str] = []
        seen: set[str] = set()
        for image_id in image_ids:
            _require_face_filename(image_id)
            if len(image_id) > _MAX_FACE_ID_LENGTH:
                raise ValueError("face image ID is too long")
            if image_id in seen:
                raise ValueError("image_ids must not contain duplicates")
            seen.add(image_id)
            validated.append(image_id)

        body = json.dumps(
            {"ids": validated},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        if len(body) > _MAX_RESPONSE_BYTES:
            # This is unreachable with the count/length limits above, but it
            # keeps the request-size invariant local to the write operation.
            raise ValueError("delete request exceeded the size limit")
        response = self._post(
            f"/api/faces/{quote(name, safe='')}/delete",
            body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        return _json_object(response, "delete faces response")

    def recognize(self, image_bytes: bytes) -> dict[str, object]:
        """Ask Frigate to recognize one uploaded image."""

        payload, extension, content_type = _require_upload(image_bytes)
        body, request_headers = _multipart_body(payload, extension, content_type)
        response = self._post(
            "/api/faces/recognize",
            body,
            headers={"Accept": "application/json", **request_headers},
        )
        return _json_object(response, "recognize response")

    def _post(self, path: str, body: bytes, *, headers: Mapping[str, str]) -> bytes:
        """Send one fixed POST route and return a bounded response body."""

        request = Request(
            self._origin + path,
            data=body,
            headers={"X-Cache-Bypass": "1", **headers},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                final = response.geturl()
                from urllib.parse import urlsplit

                final_parts = urlsplit(final)
                origin_parts = urlsplit(self._origin)
                if (final_parts.scheme, final_parts.netloc) != (
                    origin_parts.scheme,
                    origin_parts.netloc,
                ):
                    raise FrigateApiError("Frigate redirected outside its configured origin")
                response_body = response.read(_MAX_RESPONSE_BYTES + 1)
        except FrigateApiError:
            raise
        except ValueError:
            raise FrigateApiError("Frigate returned an invalid response URL") from None
        except HTTPError as error:
            raise FrigateApiError(f"Frigate returned HTTP {error.code}") from None
        except (URLError, TimeoutError, OSError) as error:
            # Keep the failure surface non-sensitive; never include a request
            # URL, body, or server response in this message.
            raise FrigateApiError(
                f"Frigate request failed ({type(error).__name__})"
            ) from None
        if len(response_body) > _MAX_RESPONSE_BYTES:
            raise FrigateApiError("Frigate response exceeded the size limit")
        return response_body


def _require_write_name(name: str) -> str:
    """Validate one remote face label and exclude the staging directory."""

    _require_face_segment(name, "face name")
    if len(name) > _MAX_FACE_NAME_LENGTH:
        raise ValueError("face name is too long")
    if name.strip() != name or not name.strip():
        raise ValueError("face name must not have surrounding or only whitespace")
    if name == "train":
        raise ValueError("Frigate's train staging directory is not writable")
    return frigate_face_name(name)


def _require_upload(image_bytes: bytes) -> tuple[bytes, str, str]:
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("image_bytes must be bytes-like")
    payload = bytes(image_bytes)
    if not payload:
        raise ValueError("image_bytes must not be empty")
    if len(payload) > _MAX_UPLOAD_BYTES:
        raise ValueError("image upload exceeded the size limit")

    if _matches_image_signature("face.png", payload):
        return payload, "png", "image/png"
    if _matches_image_signature("face.jpg", payload):
        return payload, "jpg", "image/jpeg"
    if _matches_image_signature("face.webp", payload):
        return payload, "webp", "image/webp"
    raise ValueError("image_bytes must be a PNG, JPEG, or WebP image")


def _multipart_body(
    payload: bytes,
    extension: str,
    content_type: str,
) -> tuple[bytes, dict[str, str]]:
    """Build a bounded multipart body with a fixed, non-user-controlled name."""

    boundary = f"----immich2frigate-{uuid.uuid4().hex}"
    filename = f"face.{extension}"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("ascii")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    body = prefix + payload + suffix
    if len(body) > _MAX_UPLOAD_BYTES + 2048:
        raise ValueError("multipart image upload exceeded the size limit")
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def _json_object(body: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise FrigateApiError(f"Frigate {label} was not valid JSON") from None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FrigateApiError(f"Frigate {label} was not an object")
    return value
