"""Private, durable source-face state for additive synchronization."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

_VERSION = 2


def write_json(path: str | Path, value: dict) -> None:
    """Commit private state or a journal atomically, including a durable flush."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            os.chmod(temporary, 0o600)
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


class SyncState:
    def __init__(self, path: str | Path, *, immich_origin: str, frigate_origin: str, years: int = 100):
        if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= 100:
            raise ValueError("sync state years must be between 1 and 100")
        self.path = Path(path)
        self.immich_origin = _origin(immich_origin)
        self.frigate_origin = _origin(frigate_origin)
        self.years = years
        self.roster: list[dict] = []
        self.pending: dict | None = None
        if self.path.exists():
            self._load()

    def preflight(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent) as stream:
            stream.flush()
            os.fsync(stream.fileno())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _VERSION,
            "immich_origin": self.immich_origin,
            "frigate_origin": self.frigate_origin,
            "years": self.years,
            "roster": self.roster,
            "pending": self.pending,
        }
        write_json(self.path, payload)

    def _load(self) -> None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                value.get("version") != _VERSION
                or value.get("immich_origin") != self.immich_origin
                or value.get("frigate_origin") != self.frigate_origin
                or value.get("years") != self.years
                or not isinstance(value.get("roster"), list)
                or len(value["roster"]) != 3
            ):
                raise ValueError
            seen_ids: set[str] = set()
            for person in value["roster"]:
                person_id = _uuid(person["person_id"])
                name = person["name"]
                faces = person["face_ids"]
                examined = person.get("examined_face_ids", faces)
                if (
                    person_id in seen_ids
                    or not isinstance(name, str)
                    or not name.strip()
                    or not isinstance(faces, list)
                    or not 5 <= len(faces) <= 30
                    or any(not isinstance(face_id, str) for face_id in faces)
                    or len(set(faces)) != len(faces)
                    or not isinstance(examined, list)
                    or len(examined) > 50_000
                    or any(not isinstance(face_id, str) for face_id in examined)
                    or len(set(examined)) != len(examined)
                ):
                    raise ValueError
                seen_ids.add(person_id)
                for face_id in faces:
                    _uuid(face_id)
                for face_id in examined:
                    _uuid(face_id)
                person["examined_face_ids"] = examined
            pending = value.get("pending")
            if pending is not None:
                if not isinstance(pending, dict):
                    raise ValueError
                _uuid(pending["person_id"])
                pending_face_id = _uuid(pending["face_id"])
                before_count = pending["before_count"]
                if (
                    isinstance(before_count, bool)
                    or not isinstance(before_count, int)
                    or not 0 <= before_count < 30
                    or any(pending_face_id in person["face_ids"] for person in value["roster"])
                ):
                    raise ValueError
            self.roster = value["roster"]
            self.pending = pending
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            raise ValueError("sync state is invalid or belongs to different service instances") from None


def _origin(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("sync state origin is invalid")
    return f"{parts.scheme}://{parts.netloc}"


def _uuid(value: str) -> str:
    try:
        parsed = UUID(value)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("sync state ID is invalid") from None
    if str(parsed) != value:
        raise ValueError("sync state ID is invalid")
    return value
