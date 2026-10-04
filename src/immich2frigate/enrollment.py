"""One-shot rebuild of the Frigate face library from Immich representatives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .frigate_names import frigate_face_name
from .identity_registry import PersonIdentityRegistry
from .immich_client import ImmichReadOnlyClient, PersonRecord
from .immich_vectors import ImmichVectorStore
from .selection import VectorFace, select_representative_faces
from .upload_image import prepare_candidate_upload

TARGET_FACES_PER_PERSON = 30
DEFAULT_SELECTION_POOL = 60


@dataclass(frozen=True, slots=True)
class PersonTrainingPlan:
    person: PersonRecord = field(repr=False)
    candidates: tuple[VectorFace, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class RebuildPlan:
    people: tuple[PersonTrainingPlan, ...]
    target_per_person: int = TARGET_FACES_PER_PERSON

    @property
    def total_images(self) -> int:
        return len(self.people) * self.target_per_person


@dataclass(frozen=True, slots=True)
class RebuildResult:
    people: int
    registered_images: int
    target_per_person: int


def build_rebuild_plan(
    immich: ImmichReadOnlyClient,
    vectors: ImmichVectorStore,
    *,
    target_per_person: int = TARGET_FACES_PER_PERSON,
    selection_pool: int = DEFAULT_SELECTION_POOL,
) -> RebuildPlan:
    """Preflight every named Immich person before any Frigate deletion occurs."""

    if target_per_person < 1:
        raise ValueError("target_per_person must be positive")
    if selection_pool < target_per_person:
        raise ValueError("selection_pool must be at least target_per_person")

    plans: list[PersonTrainingPlan] = []
    for person in immich.people():
        raw = immich.candidates(person.person_id)
        embedded = vectors.vectors_for_person(person, raw)
        if len(embedded) < target_per_person:
            raise ValueError(
                f"Immich person {person.name!r} has only {len(embedded)} usable vector-backed faces; "
                f"{target_per_person} are required before a destructive rebuild"
            )
        ordered = select_representative_faces(
            person,
            embedded,
            count=min(selection_pool, len(embedded)),
        )
        plans.append(PersonTrainingPlan(person=person, candidates=tuple(ordered)))

    if not plans:
        raise ValueError("Immich returned no named people")
    return RebuildPlan(tuple(plans), target_per_person)


def backup_registered_library(frigate, backup_dir: str | Path) -> dict[str, object]:
    """Back up registered Frigate face images before the one-time reset."""

    root = Path(backup_dir)
    root.mkdir(parents=True, exist_ok=True)
    inventory = frigate.inventory()
    manifest: dict[str, object] = {"faces": {}}
    for name, filenames in inventory.items():
        face_dir = root / _safe_backup_component(name)
        face_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for filename in filenames:
            body = frigate.face_image_bytes(name, filename)
            path = face_dir / _safe_backup_component(filename)
            path.write_bytes(body)
            rows.append(
                {
                    "filename": filename,
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
        manifest["faces"][name] = rows
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def reset_registered_library(frigate) -> int:
    """Delete all registered face images from a freshly read inventory."""

    inventory = frigate.inventory()
    deleted = 0
    for name, filenames in inventory.items():
        if filenames:
            frigate.delete_faces(name, list(filenames))
            deleted += len(filenames)
    remaining = frigate.inventory()
    nonempty = {name: files for name, files in remaining.items() if files}
    if nonempty:
        raise RuntimeError("Frigate face reset did not produce an empty registered library")
    return deleted


def apply_rebuild_plan(
    plan: RebuildPlan,
    immich: ImmichReadOnlyClient,
    frigate,
    *,
    registry: PersonIdentityRegistry | None = None,
) -> RebuildResult:
    """Register exactly target_per_person successful images per planned person."""

    registered_total = 0
    for person_plan in plan.people:
        name = frigate_face_name(person_plan.person.name)
        frigate.create_face(name)
        successful = 0

        for candidate in person_plan.candidates:
            if successful >= plan.target_per_person:
                break
            preview = immich.preview(candidate.source.asset_id)
            try:
                upload = prepare_candidate_upload(candidate.source, preview)
            except ValueError:
                # A local crop/encoding rejection is known before any remote
                # mutation, so it is safe to use the next representative.
                continue

            # From this point onward, fail closed. A transport error can be
            # ambiguous: Frigate may have accepted the image even if the
            # response was lost, so never auto-retry with another candidate.
            response = frigate.register_face(name, upload.encoded)
            if response.get("success") is not True:
                raise RuntimeError("Frigate did not confirm face registration")
            successful += 1

        current = frigate.inventory().get(name, ())
        if successful != plan.target_per_person or len(current) != plan.target_per_person:
            raise RuntimeError(
                f"Frigate registered {len(current)} images for {name}, expected {plan.target_per_person}"
            )
        if registry is not None:
            registry.bind(person_plan.person.person_id, name)
        registered_total += successful

    return RebuildResult(
        people=len(plan.people),
        registered_images=registered_total,
        target_per_person=plan.target_per_person,
    )


def _safe_backup_component(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("unsafe backup component")
    return value
