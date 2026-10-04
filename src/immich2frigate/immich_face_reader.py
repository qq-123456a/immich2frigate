"""Minimal read-only Immich v3 people and face-thumbnail API client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID

from .frigate_names import frigate_face_name
from .settings import ImmichSettings

_MAX_RESPONSE_BYTES = 20 * 1024 * 1024
_PAGE_SIZE = 500
_MAX_PAGES = 1000
_IMAGE_SIGNATURES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
)


class ImmichApiError(RuntimeError):
    """A bounded, non-sensitive Immich API failure."""


@dataclass(frozen=True, slots=True)
class ImmichPerson:
    person_id: str = field(repr=False)
    name: str = field(repr=False)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True, slots=True)
class ImmichFaceReader:
    """Read named people and their feature-face thumbnails without caching."""

    settings: ImmichSettings = field(repr=False)
    timeout: float = 15.0
    opener: object = field(default=None, repr=False, compare=False)
    _api: str = field(init=False, repr=False, compare=False)
    _opener: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        parts = urlsplit(self.settings.immich_url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or parts.path not in {"", "/", "/api"}
        ):
            raise ValueError("IMMICH_URL must be an origin URL, optionally ending in /api")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or not 0 < self.timeout <= 60:
            raise ValueError("timeout must be between 0 and 60 seconds")
        object.__setattr__(self, "_api", f"{parts.scheme}://{parts.netloc}/api")
        object.__setattr__(self, "_opener", self.opener or build_opener(_NoRedirect()))

    def people(self) -> list[ImmichPerson]:
        """Return all non-empty named people from Immich's paginated v3 API."""

        records: list[ImmichPerson] = []
        ids: set[str] = set()
        names: set[str] = set()
        for page_number in range(1, _MAX_PAGES + 1):
            payload = self._get_json(f"/people?page={page_number}&size={_PAGE_SIZE}")
            if isinstance(payload, list):
                # Older Immich releases returned a flat array.
                items = payload
                has_next = False
            elif isinstance(payload, dict) and isinstance(payload.get("people"), list):
                items = payload["people"]
                has_next = payload.get("hasNextPage") is True
            else:
                raise ImmichApiError("Immich people response had an unexpected shape")

            for item in items:
                if not isinstance(item, dict):
                    continue
                person_id, name = item.get("id"), item.get("name")
                if not isinstance(person_id, str) or not isinstance(name, str) or not name.strip():
                    continue
                _require_uuid(person_id)
                name = name.strip()
                if name in {".", "..", "train"} or any(c in name for c in "/\\") or any(ord(c) < 32 or ord(c) == 127 for c in name):
                    raise ImmichApiError("Immich returned a person name unsafe for Frigate")
                normalized_name = frigate_face_name(name).casefold()
                if person_id in ids or normalized_name in names:
                    raise ImmichApiError("Immich returned duplicate person IDs or names")
                ids.add(person_id)
                names.add(normalized_name)
                records.append(ImmichPerson(person_id=person_id, name=name))

            if not has_next:
                return sorted(records, key=lambda p: (p.name.casefold(), p.person_id))
        raise ImmichApiError("Immich people pagination exceeded its safety limit")

    def person_thumbnail(self, person_id: str) -> bytes:
        """Read one person's feature-face thumbnail, not the full source asset."""

        _require_uuid(person_id)
        body, content_type = self._get_bytes(
            f"/people/{quote(person_id, safe='')}/thumbnail"
        )
        if not any(body.startswith(signature) and content_type.startswith(mime) for signature, mime in _IMAGE_SIGNATURES):
            raise ImmichApiError("Immich face thumbnail is not a supported JPEG or PNG image")
        return body

    def _get_json(self, path: str) -> object:
        body, _ = self._get_bytes(path, accept="application/json")
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ImmichApiError("Immich returned invalid JSON") from None

    def _get_bytes(self, path: str, *, accept: str = "image/jpeg, image/png") -> tuple[bytes, str]:
        request = Request(
            self._api + path,
            headers={"Accept": accept, "x-api-key": self.settings.immich_api_key},
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                final = urlsplit(response.geturl())
                origin = urlsplit(self._api)
                if (final.scheme, final.netloc) != (origin.scheme, origin.netloc):
                    raise ImmichApiError("Immich redirected outside its configured origin")
                body = response.read(_MAX_RESPONSE_BYTES + 1)
                content_type = response.headers.get("Content-Type", "")
        except ImmichApiError:
            raise
        except HTTPError as error:
            raise ImmichApiError(f"Immich returned HTTP {error.code}") from None
        except ValueError:
            raise ImmichApiError("Immich returned an invalid response URL") from None
        except (URLError, TimeoutError, OSError) as error:
            raise ImmichApiError(f"Immich request failed ({type(error).__name__})") from None
        if len(body) > _MAX_RESPONSE_BYTES:
            raise ImmichApiError("Immich response exceeded the size limit")
        return body, content_type


def _require_uuid(value: str) -> None:
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise ValueError("person ID must be a UUID") from None
    if str(parsed) != value:
        raise ValueError("person ID must be a canonical UUID")
