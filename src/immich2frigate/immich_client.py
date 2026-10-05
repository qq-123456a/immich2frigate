"""Read-only Immich access using the pinned if-curator HTTP client."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from io import BytesIO
from uuid import UUID

import numpy as np

from .settings import ImmichSettings
from .frigate_names import frigate_face_name

_MAX_CANDIDATE_ASSETS = 2_000


@dataclass(frozen=True, slots=True)
class PersonRecord:
    person_id: str = field(repr=False)
    name: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class FaceCandidate:
    person_id: str = field(repr=False)
    face_id: str | None = field(repr=False)
    asset_id: str = field(repr=False)
    taken_at: str = field(repr=False)
    checksum: str = field(repr=False)
    box: tuple[float, float, float, float] = field(repr=False)
    frame: tuple[int, int] = field(repr=False)


class ImmichReadOnlyClient:
    """Fetch named people, their face boxes, and previews without local caching."""

    def __init__(self, settings: ImmichSettings, *, api=None):
        from urllib.parse import urlsplit

        parts = urlsplit(settings.immich_url)
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
        if api is None:
            api = _ImmichHttpApi(settings.immich_url, settings.immich_api_key)
        # Never forward x-api-key beyond the configured host.
        session = getattr(api, "session", None)
        if session is None or not hasattr(session, "close"):
            raise ValueError("Immich API client must expose a closable session")
        session.max_redirects = 0
        self._api = api

    def people(self) -> list[PersonRecord]:
        """Return only stable IDs and non-empty names; discard full API payloads."""
        records: list[PersonRecord] = []
        ids: set[str] = set()
        names: set[str] = set()
        for item in self._api.people():
            person_id = item.get("id") if isinstance(item, dict) else None
            name = item.get("name") if isinstance(item, dict) else None
            if not isinstance(person_id, str) or not isinstance(name, str) or not name.strip():
                continue
            _require_uuid(person_id, "person ID")
            name = name.strip()
            if (
                name in {".", "..", "train"}
                or any(char in name for char in "/\\")
                or any(ord(char) < 32 or ord(char) == 127 for char in name)
            ):
                raise ValueError("Immich person name cannot be used as a Frigate face name")
            if person_id in ids:
                raise ValueError("Immich returned a duplicate person ID")
            normalized_name = frigate_face_name(name).casefold()
            if normalized_name in names:
                raise ValueError("Immich returned names that collide in Frigate")
            ids.add(person_id)
            names.add(normalized_name)
            records.append(PersonRecord(person_id=person_id, name=name))
        return sorted(records, key=lambda p: (p.name.casefold(), p.person_id))

    def candidates(self, person_id: str, years: int = 100) -> list[FaceCandidate]:
        """Return only this person's valid face boxes and minimal asset metadata."""
        _require_uuid(person_id, "person ID")
        if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= 100:
            raise ValueError("years must be between 1 and 100")
        result: list[FaceCandidate] = []
        seen: set[tuple[str, str | None, tuple[float, float, float, float]]] = set()
        assets = self._api.photos(person_id, years)
        if len(assets) > _MAX_CANDIDATE_ASSETS:
            assets.sort(key=lambda item: (
                str(item.get("fileCreatedAt") or item.get("localDateTime") or ""),
                str(item.get("id") or ""),
            ))
            indexes = {
                round(index * (len(assets) - 1) / (_MAX_CANDIDATE_ASSETS - 1))
                for index in range(_MAX_CANDIDATE_ASSETS)
            }
            assets = [asset for index, asset in enumerate(assets) if index in indexes]
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            asset_id = asset.get("id")
            if not isinstance(asset_id, str):
                continue
            _require_uuid(asset_id, "asset ID")
            timestamp = asset.get("fileCreatedAt") or asset.get("localDateTime") or ""
            checksum = asset.get("checksum") or ""
            if not isinstance(timestamp, str) or not isinstance(checksum, str):
                continue
            if not timestamp.strip() or not checksum.strip():
                continue
            for face in self._api.target_faces(asset, person_id):
                if not isinstance(face, dict):
                    continue
                candidate = self._candidate(person_id, asset_id, timestamp, checksum, face)
                key = (candidate.asset_id, candidate.face_id, candidate.box) if candidate else None
                if candidate is not None and key not in seen:
                    result.append(candidate)
                    seen.add(key)
        result.sort(key=lambda c: (c.taken_at, c.asset_id, c.face_id or ""))
        return result

    def photo_count(self, person_id: str, years: int = 100) -> int:
        """Count distinct images for one person without fetching face details."""
        _require_uuid(person_id, "person ID")
        if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= 100:
            raise ValueError("years must be between 1 and 100")
        return len({
            asset.get("id")
            for asset in self._api.photos(person_id, years)
            if isinstance(asset, dict) and isinstance(asset.get("id"), str)
        })

    @staticmethod
    def _candidate(
        person_id: str,
        asset_id: str,
        taken_at: str,
        checksum: str,
        face: dict,
    ) -> FaceCandidate | None:
        keys = ("boundingBoxX1", "boundingBoxY1", "boundingBoxX2", "boundingBoxY2")
        values = [face.get(key) for key in keys]
        width, height = face.get("imageWidth"), face.get("imageHeight")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not _finite_coordinate(value)
                for value in values
            )
            or isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            return None
        x1, y1, x2, y2 = (float(value) for value in values)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            return None
        face_id = face.get("id")
        returned_person_id = face.get("personId")
        if returned_person_id is not None and returned_person_id != person_id:
            return None
        if face_id is not None and not isinstance(face_id, str):
            return None
        if face_id is not None:
            try:
                _require_uuid(face_id, "face ID")
            except ValueError:
                return None
        return FaceCandidate(
            person_id=person_id,
            face_id=face_id,
            asset_id=asset_id,
            taken_at=taken_at,
            checksum=checksum,
            box=(x1, y1, x2, y2),
            frame=(width, height),
        )

    def preview(self, asset_id: str) -> np.ndarray:
        """Load one upright BGR preview into memory; this method does not cache it."""
        _require_uuid(asset_id, "asset ID")
        image = self._api.image(asset_id, original=False)
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
            or image.dtype != np.uint8
        ):
            raise ValueError("Immich preview was not a non-empty 8-bit BGR image")
        return np.ascontiguousarray(image)

    def close(self) -> None:
        self._api.session.close()

    def __enter__(self) -> ImmichReadOnlyClient:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _require_uuid(value: str, label: str) -> None:
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise ValueError(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise ValueError(f"{label} must be a canonical UUID")


def _finite_coordinate(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


class _ImmichHttpApi:
    """Small, bounded Immich API adapter for the read-only operations used here."""

    def __init__(self, origin: str, api_key: str, *, session=None):
        import requests

        self.origin = origin.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update({"x-api-key": api_key, "Accept": "application/json"})
        self.session.max_redirects = 0

    def people(self) -> list[dict]:
        rows: list[dict] = []
        page = 1
        while True:
            data = self._request("GET", "/people", params={"page": page, "size": 1000})
            batch = data.get("people") if isinstance(data, dict) else None
            if not isinstance(batch, list):
                raise ValueError("Immich people response had an unexpected shape")
            rows.extend(item for item in batch if isinstance(item, dict))
            if data.get("hasNextPage") is not True:
                return rows
            page += 1

    def photos(self, person_id: str, years: int) -> list[dict]:
        if not 1 <= years <= 100:
            raise ValueError("years must be between 1 and 100")
        taken_after = datetime.now(UTC) - timedelta(days=round(365.25 * years))
        query = {
            "personIds": [person_id],
            "type": "IMAGE",
            "takenAfter": taken_after.isoformat(),
            "withPeople": True,
            "size": 1000,
        }
        assets: list[dict] = []
        page = 1
        while page:
            data = self._request("POST", "/search/metadata", json={**query, "page": page})
            result = data.get("assets") if isinstance(data, dict) else None
            items = result.get("items") if isinstance(result, dict) else None
            if not isinstance(items, list):
                raise ValueError("Immich asset search response had an unexpected shape")
            assets.extend(item for item in items if isinstance(item, dict))
            next_page = result.get("nextPage")
            page = int(next_page) if next_page else 0
        return assets

    def target_faces(self, asset: dict, person_id: str) -> list[dict]:
        faces = [
            face
            for person in asset.get("people") or []
            if person.get("id") == person_id
            for face in person.get("faces") or []
        ]
        if faces and all(
            key in face
            for face in faces
            for key in ("boundingBoxX1", "boundingBoxY1", "boundingBoxX2", "boundingBoxY2")
        ):
            return faces
        rows = self._request("GET", "/faces", params={"id": asset["id"]})
        if not isinstance(rows, list):
            raise ValueError("Immich face response had an unexpected shape")
        return [face for face in rows if (face.get("person") or {}).get("id") == person_id]

    def image(self, asset_id: str, original: bool = False) -> np.ndarray:
        from PIL import Image, ImageOps

        suffix = "original" if original else "thumbnail?size=preview"
        route = f"/assets/{asset_id}/{suffix}"
        body = self._request("GET", route, max_bytes=32 * 1024 * 1024, raw=True)
        with Image.open(BytesIO(body)) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def _request(self, method: str, route: str, *, max_bytes: int = 16 * 1024 * 1024, raw: bool = False, **kwargs):
        try:
            response = self.session.request(
                method,
                f"{self.origin}/api{route}",
                timeout=(10, 60),
                allow_redirects=False,
                stream=True,
                **kwargs,
            )
        except Exception as error:
            raise RuntimeError(f"Immich request failed ({type(error).__name__})") from None
        try:
            if 300 <= response.status_code < 400:
                raise RuntimeError("Immich redirected a credentialed request")
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"Immich returned HTTP {response.status_code}")
            content = bytearray()
            for chunk in response.iter_content(64 * 1024):
                if not chunk:
                    continue
                content.extend(chunk)
                if len(content) > max_bytes:
                    raise RuntimeError("Immich response exceeded the size limit")
        finally:
            response.close()
        if raw:
            return bytes(content)
        try:
            import json

            return json.loads(content)
        except (UnicodeDecodeError, ValueError):
            raise RuntimeError("Immich returned invalid JSON") from None
