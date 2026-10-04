"""Build a deterministic, in-memory review plan; this module never writes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from urllib.parse import urlsplit
from uuid import UUID

from .frigate018 import require_target
from .frigate_client import FrigateTarget
from .frigate_registration import prepare_registered_upload
from .immich_client import FaceCandidate, PersonRecord


@dataclass(frozen=True, slots=True)
class DryRunCandidate:
    person: PersonRecord = field(repr=False)
    source: FaceCandidate = field(repr=False)
    upload_bytes: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class DryRunPlan:
    plan_id: str
    target: str = field(repr=False)
    version: str
    model_size: str
    entries: tuple[Mapping[str, object], ...]
    dry_run: bool = True
    writes_enabled: bool = False
    successful_sync_recorded: bool = False

    def as_dict(self) -> dict:
        """Return an explicit JSON-safe view; face bytes are never serialized."""
        return {
            "plan_id": self.plan_id,
            "state": "DRY_RUN_ONLY",
            "compatibility_status": "MODEL_ASSETS_NOT_VERIFIED",
            "target": self.target,
            "version": self.version,
            "model_size": self.model_size,
            "dry_run": True,
            "writes_enabled": False,
            "successful_sync_recorded": False,
            "entries": [dict(entry) for entry in self.entries],
        }

    def to_json(self) -> str:
        """Serialize locally at the caller's request; no file is created here."""
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, indent=2)


def build_dry_run_plan(
    target: FrigateTarget,
    people: list[PersonRecord],
    candidates: list[DryRunCandidate],
    existing_faces: Mapping[str, tuple[str, ...]],
    *,
    detector,
) -> DryRunPlan:
    """Describe proposed additions and expected Frigate re-encoded bytes.

    Each `upload_bytes` value must be the exact image request body content the
    caller intends to send. The function only decodes and simulates it locally.
    Existing labels are treated as manually owned and are never proposed for
    modification. The returned plan is never an authorization to upload.
    """
    require_target(target.version, target.model_size)
    _require_safe_origin(target.origin)
    people_by_id = {person.person_id: person for person in people}
    if len(people_by_id) != len(people):
        raise ValueError("people contains duplicate IDs")
    name_owners: dict[str, str] = {}
    for person in people:
        _require_uuid(person.person_id, "person ID")
        _require_safe_name(person.name)
        normalized = person.name.casefold()
        if normalized in name_owners:
            raise ValueError("people contains Frigate name collisions")
        name_owners[normalized] = person.person_id

    for name in existing_faces:
        _require_safe_name(name)
    existing_by_name: dict[str, str] = {}
    for name in existing_faces:
        normalized = name.casefold()
        if normalized in existing_by_name:
            raise ValueError("Frigate has case-colliding face labels")
        existing_by_name[normalized] = name
    seen_sources: set[tuple[str, str]] = set()
    entries: list[dict] = []
    for candidate in candidates:
        person = people_by_id.get(candidate.person.person_id)
        if person is None or person.name != candidate.person.name:
            raise ValueError("candidate person is not in the supplied people inventory")
        if candidate.source.person_id != person.person_id:
            raise ValueError("candidate belongs to a different Immich person")
        _require_uuid(candidate.source.asset_id, "asset ID")
        if candidate.source.face_id is not None:
            _require_uuid(candidate.source.face_id, "face ID")
        source_key = (person.person_id, candidate.source.asset_id)
        if source_key in seen_sources:
            raise ValueError("plan contains duplicate candidates from one person and asset")
        seen_sources.add(source_key)
        if not isinstance(candidate.upload_bytes, bytes) or not candidate.upload_bytes:
            raise ValueError("candidate upload must be non-empty bytes")

        entry = {
            "person_id": person.person_id,
            "person_name": person.name,
            "asset_id": candidate.source.asset_id,
            "face_id": candidate.source.face_id,
            "upload_bytes": len(candidate.upload_bytes),
            "upload_sha256": hashlib.sha256(candidate.upload_bytes).hexdigest(),
            "status": "PROPOSED_ADD_MODELS_UNVERIFIED",
            "existing_label": None,
            "simulated_registered_image_sha256": None,
            "registered_box": None,
            "manual_review_required": True,
        }
        existing_name = existing_by_name.get(person.name.casefold())
        if existing_name is not None:
            entry["status"] = "SKIPPED_EXISTING_LABEL"
            entry["existing_label"] = existing_name
        else:
            registered = prepare_registered_upload(candidate.upload_bytes, detector)
            if registered is None:
                entry["status"] = "BLOCKED_NO_FACE_DETECTED"
            else:
                entry["simulated_registered_image_sha256"] = hashlib.sha256(
                    registered.stored_webp
                ).hexdigest()
                entry["registered_box"] = registered.box
        entries.append(entry)

    entries.sort(
        key=lambda item: (
            item["person_name"].casefold(),
            item["person_id"],
            item["asset_id"],
            item["face_id"] or "",
        )
    )
    identity = {
        "target": target.origin,
        "version": target.version,
        "model_size": target.model_size,
        "entries": entries,
    }
    plan_id = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return DryRunPlan(
        plan_id=plan_id,
        target=target.origin,
        version=target.version,
        model_size=target.model_size,
        entries=tuple(MappingProxyType(dict(entry)) for entry in entries),
    )


def _require_safe_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", "..", "train"}
        or any(char in name for char in "/\\")
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        raise ValueError("person name cannot be used as a Frigate face name")


def _require_safe_origin(origin: str) -> None:
    try:
        parts = urlsplit(origin)
        valid_port = parts.port is None or 1 <= parts.port <= 65535
    except (TypeError, ValueError):
        valid_port = False
        parts = None
    if (
        parts is None
        or parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
        or not valid_port
    ):
        raise ValueError("dry-run target must be a credential-free origin URL")


def _require_uuid(value: str, label: str) -> None:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError):
        raise ValueError(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise ValueError(f"{label} must be a canonical UUID")
