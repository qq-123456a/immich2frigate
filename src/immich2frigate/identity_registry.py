"""Private, local bindings between Immich people and Frigate face labels."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from .frigate_names import frigate_face_name

_VERSION = 1


@dataclass(frozen=True, slots=True)
class PersonIdentityBinding:
    """A sync-owned identity; Frigate itself exposes labels, not person IDs."""

    immich_person_id: str
    sync_person_id: str
    frigate_name: str


class PersonIdentityRegistry:
    """Persist verified Immich-to-Frigate bindings outside the source tree.

    ``sync_person_id`` is generated and owned by this project. Frigate 0.18's
    face-library API has no stable person ID, so the registry keeps its current
    Frigate label alongside the Immich UUID. Call :meth:`bind` only after the
    remote enrollment or rename has been verified.
    """

    def __init__(self, path: str | Path, *, immich_origin: str, frigate_origin: str):
        self.path = Path(path)
        self.immich_origin = normalize_origin(immich_origin, allow_api=True)
        self.frigate_origin = normalize_origin(frigate_origin)
        self._bindings: dict[str, PersonIdentityBinding] = {}
        if self.path.exists():
            self._load()

    def binding(self, immich_person_id: str) -> PersonIdentityBinding | None:
        return self._bindings.get(_person_id(immich_person_id))

    def bind(self, immich_person_id: str, frigate_name: str) -> PersonIdentityBinding:
        """Create/update one binding after the Frigate result is confirmed."""

        person_id = _person_id(immich_person_id)
        name = validate_frigate_name(frigate_name)
        for other_id, other in self._bindings.items():
            if (
                other_id != person_id
                and frigate_face_name(other.frigate_name).casefold()
                == frigate_face_name(name).casefold()
            ):
                raise ValueError("Frigate label is already bound to another Immich person")
        previous = self._bindings.get(person_id)
        item = PersonIdentityBinding(
            immich_person_id=person_id,
            sync_person_id=previous.sync_person_id if previous else str(uuid4()),
            frigate_name=name,
        )
        self._bindings[person_id] = item
        try:
            self._save()
        except Exception:
            if previous is None:
                self._bindings.pop(person_id, None)
            else:
                self._bindings[person_id] = previous
            raise
        return item

    def reconcile(self, people, frigate_faces: dict[str, tuple[str, ...]]) -> tuple[dict[str, str], ...]:
        """Describe changes and bootstrap same-name labels in a managed library."""

        remote: dict[str, str] = {}
        for name in frigate_faces:
            key = frigate_face_name(name).casefold()
            if key in remote:
                raise ValueError("Frigate has colliding face labels")
            remote[key] = name
        actions: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for person in people:
            person_id = _person_id(person.person_id)
            if person_id in seen_ids:
                raise ValueError("people contains duplicate Immich person IDs")
            seen_ids.add(person_id)
            new_name = _face_name(person.name)
            normalized_name = new_name.casefold()
            if normalized_name in seen_names:
                raise ValueError("Immich people have colliding Frigate labels")
            seen_names.add(normalized_name)
            old = self._bindings.get(person_id)
            current_remote = remote.get(normalized_name)
            if old is None:
                # The Frigate library is owned by this integration. A same-name
                # label with registered faces is therefore a deterministic first
                # sync match, rather than a manual adoption decision.
                status = (
                    "AUTO_BIND_REQUIRED"
                    if current_remote and frigate_faces[current_remote]
                    else "UNBOUND_PERSON"
                )
                action = {"person_id": person_id, "name": new_name, "status": status}
                if status == "AUTO_BIND_REQUIRED":
                    action["frigate_name"] = current_remote
                actions.append(action)
                continue

            old_remote = frigate_faces.get(old.frigate_name)
            if old.frigate_name == new_name:
                status = "MATCHED" if old_remote is not None else "BOUND_LABEL_MISSING"
            elif old_remote is not None and (
                current_remote is None or current_remote == old.frigate_name
            ):
                status = "RENAME_REQUIRED"
            elif old_remote is None and current_remote is not None:
                status = "REMOTE_ALREADY_RENAMED"
            elif old_remote is not None:
                status = "RENAME_CONFLICT"
            else:
                status = "BOUND_LABELS_MISSING"
            actions.append(
                {
                    "person_id": person_id,
                    "sync_person_id": old.sync_person_id,
                    "old_name": old.frigate_name,
                    "name": new_name,
                    "status": status,
                }
            )
        return tuple(actions)

    def _load(self) -> None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                value.get("version") != _VERSION
                or value.get("immich_origin") != self.immich_origin
                or value.get("frigate_origin") != self.frigate_origin
                or not isinstance(value.get("bindings"), list)
            ):
                raise ValueError
            for row in value["bindings"]:
                if not isinstance(row, dict):
                    raise ValueError
                person_id = _person_id(row["immich_person_id"])
                sync_id = _person_id(row["sync_person_id"])
                name = validate_frigate_name(row["frigate_name"])
                if person_id in self._bindings or any(
                    item.sync_person_id == sync_id
                    or frigate_face_name(item.frigate_name).casefold()
                    == frigate_face_name(name).casefold()
                    for item in self._bindings.values()
                ):
                    raise ValueError
                self._bindings[person_id] = PersonIdentityBinding(person_id, sync_id, name)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            raise ValueError("identity registry is invalid or belongs to different service instances") from None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "version": _VERSION,
            "immich_origin": self.immich_origin,
            "frigate_origin": self.frigate_origin,
            "bindings": [
                {
                    "immich_person_id": row.immich_person_id,
                    "sync_person_id": row.sync_person_id,
                    "frigate_name": row.frigate_name,
                }
                for row in sorted(self._bindings.values(), key=lambda item: item.immich_person_id)
            ],
        }
        content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as temporary:
                temporary_path = temporary.name
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)


def normalize_origin(value: str, *, allow_api: bool = False) -> str:
    parts = urlsplit(value)
    try:
        port = parts.port
    except ValueError:
        raise ValueError("service origin must be an absolute URL without credentials or path") from None
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ({"", "/", "/api"} if allow_api else {"", "/"})
        or parts.query
        or parts.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("service origin must be an absolute URL without credentials or path")
    return f"{parts.scheme}://{parts.netloc}"


def validate_frigate_name(value: str) -> str:
    """Validate an exact label returned by Frigate without normalizing it."""

    return _face_name(value, normalize=False)


def _person_id(value: str) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("person ID must be a UUID") from None
    canonical = str(parsed)
    if value != canonical:
        raise ValueError("person ID must be a canonical UUID")
    return canonical


def _face_name(value: str, *, normalize: bool = True) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value in {".", "..", "train"}
        or any(char in value for char in "/\\")
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ) or len(value) > 50 or any(
        not (char.isalpha() or char.isdigit() or char in "'_- ")
        for char in value
    ):
        raise ValueError("Frigate face name is unsafe")
    return frigate_face_name(value) if normalize else value
