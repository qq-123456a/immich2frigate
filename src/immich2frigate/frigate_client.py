"""Strictly read-only client for the Frigate 0.18 internal API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .frigate018 import require_target
from .settings import FrigateSettings

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_FACE_EXTENSIONS = (".webp", ".png", ".jpg", ".jpeg")


class FrigateApiError(RuntimeError):
    """A bounded, non-sensitive Frigate API failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        api_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.api_message = api_message


@dataclass(frozen=True, slots=True)
class FrigateTarget:
    """The non-secret identity fields verified during a Frigate preflight."""

    origin: str = field(repr=False)
    version: str
    model_size: str


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class FrigateReadOnlyClient:
    """Read version, redacted runtime profile, and existing face filenames.

    Frigate's internal API port grants administrator-equivalent access. Only
    point this client at a trusted, private network endpoint. This class exposes
    fixed GET operations only; it has no arbitrary request method.
    """

    def __init__(
        self,
        settings: FrigateSettings,
        timeout: float = 10,
        *,
        opener=None,
    ):
        parts = urlsplit(settings.frigate_url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.path not in {"", "/"}
            or parts.query
            or parts.fragment
        ):
            raise ValueError("FRIGATE_URL must be an origin URL without credentials or path")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 60:
            raise ValueError("timeout must be between 0 and 60 seconds")
        self._origin = f"{parts.scheme}://{parts.netloc}"
        self._timeout = timeout
        self._opener = opener or build_opener(_NoRedirect())

    def _get(self, path: str) -> bytes:
        request = Request(
            self._origin + path,
            headers={"Accept": "application/json, text/plain", "X-Cache-Bypass": "1"},
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                final = urlsplit(response.geturl())
                if (final.scheme, final.netloc) != tuple(urlsplit(self._origin)[i] for i in (0, 1)):
                    raise FrigateApiError("Frigate redirected outside its configured origin")
                body = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            raise FrigateApiError(f"Frigate returned HTTP {error.code}") from None
        except (URLError, TimeoutError, OSError) as error:
            raise FrigateApiError(f"Frigate request failed ({type(error).__name__})") from None
        except ValueError:
            raise FrigateApiError("Frigate returned an invalid response URL") from None
        if len(body) > _MAX_RESPONSE_BYTES:
            raise FrigateApiError("Frigate response exceeded the size limit")
        return body

    def version(self) -> str:
        """Read the public version endpoint; it does not contain credentials."""
        try:
            return self._get("/api/version").decode("utf-8").strip()
        except UnicodeDecodeError:
            raise FrigateApiError("Frigate version response was not valid UTF-8") from None

    def _require_admin(self) -> None:
        try:
            profile = json.loads(self._get("/api/profile"))
            role = profile["role"]
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            raise FrigateApiError("Frigate profile did not contain an authorization role") from None
        if role != "admin":
            raise FrigateApiError("Frigate internal API did not grant the required admin role")

    def verify_target(self) -> FrigateTarget:
        """Require the inspected version, enabled recognition, large model, and admin role."""
        version = self.version()
        self._require_admin()
        try:
            config = json.loads(self._get("/api/config"))
            face_config = config["face_recognition"]
            model_size = face_config["model_size"]
            enabled = face_config["enabled"]
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            raise FrigateApiError("Frigate config did not contain face-recognition settings") from None
        if not isinstance(model_size, str):
            raise FrigateApiError("Frigate face model size was not a string")
        if enabled is not True:
            raise FrigateApiError("Frigate face recognition is not enabled")
        try:
            require_target(version, model_size)
        except ValueError:
            raise FrigateApiError("Frigate does not match the verified compatibility target") from None
        return FrigateTarget(self._origin, version, model_size)

    def faces(self) -> dict[str, tuple[str, ...]]:
        """Return validated names and supported image filenames, never image bytes."""
        try:
            value = json.loads(self._get("/api/faces"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise FrigateApiError("Frigate face library response was not valid JSON") from None
        if not isinstance(value, Mapping):
            raise FrigateApiError("Frigate face library response was not an object")
        result: dict[str, tuple[str, ...]] = {}
        for name, filenames in value.items():
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or any(c in name for c in "/\\")
                or any(ord(c) < 32 or ord(c) == 127 for c in name)
            ):
                raise FrigateApiError("Frigate returned an unsafe face name")
            if name == "train":
                continue
            if not isinstance(filenames, list) or not all(isinstance(file, str) for file in filenames):
                raise FrigateApiError("Frigate returned an invalid face filename list")
            if any(
                not file
                or file in {".", ".."}
                or "/" in file
                or "\\" in file
                or any(ord(c) < 32 or ord(c) == 127 for c in file)
                or not file.lower().endswith(_FACE_EXTENSIONS)
                for file in filenames
            ):
                raise FrigateApiError("Frigate returned an unsafe face filename")
            result[name] = tuple(sorted(filenames))
        return dict(sorted(result.items()))

    def face_image_bytes(self, name: str, filename: str) -> bytes:
        """Read one registered image with path quoting and format validation."""
        _require_face_segment(name, "face name")
        if name == "train":
            raise ValueError("Frigate's train staging directory is not a registered face")
        _require_face_filename(filename)
        body = self._get(
            f"/clips/faces/{quote(name, safe='')}/{quote(filename, safe='')}"
        )
        if not _matches_image_signature(filename, body):
            raise FrigateApiError("Frigate face image did not match its supported format")
        return body


def _require_face_segment(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(char in value for char in "/\\")
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"{label} is unsafe")


def _require_face_filename(value: str) -> None:
    _require_face_segment(value, "face filename")
    if not value.lower().endswith(_FACE_EXTENSIONS):
        raise ValueError("face filename extension is unsupported")


def _matches_image_signature(filename: str, body: bytes) -> bool:
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix == "png":
        return body.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {"jpg", "jpeg"}:
        return body.startswith(b"\xff\xd8\xff")
    if suffix == "webp":
        return len(body) >= 12 and body[:4] == b"RIFF" and body[8:12] == b"WEBP"
    return False
