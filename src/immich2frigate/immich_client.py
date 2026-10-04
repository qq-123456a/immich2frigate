"""Read-only Immich access using the pinned if-curator HTTP client."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np

from .settings import ImmichSettings


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
            # Import lazily so compatibility-only installs do not need if-curator.
            from if_curator.immich import Immich

            api = Immich(settings.immich_url, settings.immich_api_key)
        # The upstream wrapper uses requests.Session, whose default behavior
        # follows redirects. Never forward x-api-key beyond the configured host.
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
            normalized_name = name.casefold()
            if normalized_name in names:
                raise ValueError("Immich returned names that collide in Frigate")
            ids.add(person_id)
            names.add(normalized_name)
            records.append(PersonRecord(person_id=person_id, name=name))
        return sorted(records, key=lambda p: (p.name.casefold(), p.person_id))

    def candidates(self, person_id: str, years: int = 10) -> list[FaceCandidate]:
        """Return only this person's valid face boxes and minimal asset metadata."""
        _require_uuid(person_id, "person ID")
        if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= 100:
            raise ValueError("years must be between 1 and 100")
        result: list[FaceCandidate] = []
        seen: set[tuple[str, str | None, tuple[float, float, float, float]]] = set()
        for asset in self._api.photos(person_id, years):
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
