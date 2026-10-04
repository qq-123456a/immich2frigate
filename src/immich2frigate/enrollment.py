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
from .selection import (
    FOUNDATION_COUNT,
    MAX_TRAINING_COUNT,
    VectorFace,
    select_adaptive_faces,
)
from .training_quality import assess_training_crop
from .upload_image import prepare_candidate_upload


@dataclass(frozen=True, slots=True)
class PersonTrainingPlan:
    person: PersonRecord = field(repr=False)
    candidates: tuple[VectorFace, ...] = field(repr=False)

    @property
    def target_count(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True, slots=True)
class RebuildPlan:
    people: tuple[PersonTrainingPlan, ...]

    @property
    def total_images(self) -> int:
        return sum(person.target_count for person in self.people)


@dataclass(frozen=True, slots=True)
class RebuildResult:
    people: int
    registered_images: int
    minimum_per_person: int
    maximum_per_person: int


def build_rebuild_plan(
    immich: ImmichReadOnlyClient,
    vectors: ImmichVectorStore,
    *,
    foundation_count: int = FOUNDATION_COUNT,
    max_count: int = MAX_TRAINING_COUNT,
) -> RebuildPlan:
    """Build every person's adaptive plan before any Frigate deletion occurs.

    The destructive reset is allowed only when every named person has at least
    foundation_count high-quality foundation images. Extra images are optional
    and selected only when their Immich face and scene vectors add useful coverage.
    """

    plans: list[PersonTrainingPlan] = []
    for person in immich.people():
        raw = immich.candidates(person.person_id)
        embedded = vectors.vectors_for_person(person, raw)

        foundation: list[VectorFace] = []
        expansion: list[VectorFace] = []
        for candidate in embedded:
            try:
                preview = immich.preview(candidate.source.asset_id)
                upload = prepare_candidate_upload(candidate.source, preview)
                quality = assess_training_crop(candidate.source, upload.crop_bgr)
            except ValueError:
                continue
            if quality.training_eligible:
                expansion.append(candidate)
            if quality.foundation_eligible:
                foundation.append(candidate)

        if len(foundation) < foundation_count:
            raise ValueError(
                f"Immich person {person.name!r} has only {len(foundation)} "
                f"foundation-quality faces; {foundation_count} are required before "
                "a destructive rebuild"
            )

        selected = select_adaptive_faces(
            person,
            foundation,
            expansion,
            foundation_count=foundation_count,
            max_count=max_count,
        )
        if len(selected) < foundation_count:
            raise RuntimeError("adaptive selection returned too few foundation faces")
        plans.append(PersonTrainingPlan(person=person, candidates=tuple(selected)))

    if not plans:
        raise ValueError("Immich returned no named people")
    return RebuildPlan(tuple(plans))


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
    """Register each person's adaptive candidate count exactly."""

    registered_total = 0
    per_person_counts: list[int] = []

    for person_plan in plan.people:
        name = frigate_face_name(person_plan.person.name)
        frigate.create_face(name)
        successful = 0

        for candidate in person_plan.candidates:
            preview = immich.preview(candidate.source.asset_id)
            upload = prepare_candidate_upload(candidate.source, preview)

            response = frigate.register_face(name, upload.encoded)
            if response.get("success") is not True:
                raise RuntimeError("Frigate did not confirm face registration")
            successful += 1

        current = frigate.inventory().get(name, ())
        if successful != person_plan.target_count or len(current) != person_plan.target_count:
            raise RuntimeError(
                f"Frigate registered {len(current)} images for {name}, "
                f"expected {person_plan.target_count}"
            )
        if registry is not None:
            registry.bind(person_plan.person.person_id, name)

        registered_total += successful
        per_person_counts.append(successful)

    return RebuildResult(
        people=len(plan.people),
        registered_images=registered_total,
        minimum_per_person=min(per_person_counts),
        maximum_per_person=max(per_person_counts),
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
