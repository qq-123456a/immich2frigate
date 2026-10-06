"""One-shot rebuild of the Frigate face library from Immich representatives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .frigate_names import frigate_face_name
from .identity_registry import PersonIdentityRegistry, validate_frigate_name
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
    uploads: tuple[bytes, ...] = field(repr=False)
    examined_face_ids: tuple[str, ...] = field(default=(), repr=False)

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
    people: Sequence[PersonRecord] | None = None,
    years: int = 100,
    foundation_count: int = FOUNDATION_COUNT,
    max_count: int = MAX_TRAINING_COUNT,
) -> RebuildPlan:
    """Build every person's adaptive plan before any Frigate deletion occurs.

    The destructive reset is allowed only when every named person has at least
    foundation_count high-quality foundation images. Extra images are optional
    and selected when face vectors, plus available scene vectors, add useful coverage.
    """

    plans: list[PersonTrainingPlan] = []
    for person in (immich.people() if people is None else people):
        validate_frigate_name(frigate_face_name(person.name))
        if hasattr(vectors, "candidates_for_person"):
            embedded = vectors.candidates_for_person(person, years=years)
        else:
            raw = immich.candidates(person.person_id, years=years)
            embedded = vectors.vectors_for_person(person, raw)
        if len(embedded) < foundation_count:
            raise ValueError(
                f"Immich person {person.name!r} has only {len(embedded)} vector-backed faces; "
                f"{foundation_count} are required before a destructive rebuild"
            )

        # Vectors are cheap compared with fetching and decoding thousands of
        # previews. Pick a bounded diverse set first, then run image-quality
        # checks only on samples that could actually be registered.
        preselected = select_adaptive_faces(
            person,
            embedded,
            embedded,
            foundation_count=foundation_count,
            max_count=MAX_TRAINING_COUNT,
        )
        foundation: list[VectorFace] = []
        expansion: list[VectorFace] = []
        prepared = {}
        for candidate in preselected:
            try:
                preview = immich.preview(candidate.source.asset_id)
                upload = prepare_candidate_upload(candidate.source, preview)
                quality = assess_training_crop(
                    candidate.source, upload.crop_bgr, face_box=upload.face_box
                )
            except ValueError:
                continue
            prepared[candidate.source.face_id] = upload
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
        uploads = [prepared[candidate.source.face_id].encoded for candidate in selected]
        plans.append(
            PersonTrainingPlan(
                person=person,
                candidates=tuple(selected),
                uploads=tuple(uploads),
                examined_face_ids=tuple(
                    item.source.face_id for item in embedded if item.source.face_id is not None
                ),
            )
        )

    if not plans:
        raise ValueError("Immich returned no named people")
    return RebuildPlan(tuple(plans))


def build_incremental_plan(
    immich: ImmichReadOnlyClient,
    vectors: ImmichVectorStore,
    person: PersonRecord,
    known_face_ids: Sequence[str],
    examined_face_ids: Sequence[str],
    *,
    years: int = 100,
    max_count: int = MAX_TRAINING_COUNT,
) -> tuple[PersonTrainingPlan, tuple[str, ...]]:
    """Prepare only unseen, unexamined candidates against the locked face core."""
    known_ids = set(known_face_ids)
    examined_ids = set(examined_face_ids)
    if len(known_ids) < FOUNDATION_COUNT or len(known_ids) > MAX_TRAINING_COUNT:
        raise ValueError("incremental sync state has an invalid enrolled face count")
    if max_count < len(known_ids) or max_count > MAX_TRAINING_COUNT:
        raise ValueError("incremental sync maximum is outside the valid range")

    if hasattr(vectors, "candidates_for_person"):
        embedded = vectors.candidates_for_person(
            person, years=years, include_face_ids=tuple(known_ids)
        )
    else:
        raw = immich.candidates(person.person_id, years=years)
        embedded = vectors.vectors_for_person(person, raw)
    by_id = {item.source.face_id: item for item in embedded if item.source.face_id is not None}
    known = [by_id[face_id] for face_id in known_ids if face_id in by_id]
    if len(known) != len(known_ids):
        raise ValueError("a registered Immich face is no longer available in the vector store")

    new_candidates = [
        item for item in embedded
        if item.source.face_id not in known_ids and item.source.face_id not in examined_ids
    ]
    known_assets = {item.source.asset_id for item in known}
    new_candidates, duplicate_face_ids = _one_face_per_asset(
        new_candidates, excluded_assets=known_assets
    )
    proposed = select_adaptive_faces(
        person,
        known,
        [*known, *new_candidates],
        foundation_count=FOUNDATION_COUNT,
        max_count=max_count,
    )
    proposed_ids = {item.source.face_id for item in proposed}
    eligible: list[VectorFace] = []
    reviewed: set[str] = set(duplicate_face_ids)
    prepared = {}
    for candidate in new_candidates:
        face_id = candidate.source.face_id
        if face_id not in proposed_ids:
            continue
        try:
            preview = immich.preview(candidate.source.asset_id)
            upload = prepare_candidate_upload(candidate.source, preview)
            quality = assess_training_crop(
                candidate.source, upload.crop_bgr, face_box=upload.face_box
            )
        except ValueError:
            reviewed.add(face_id)
            continue
        reviewed.add(face_id)
        if quality.training_eligible:
            eligible.append(candidate)
            prepared[face_id] = upload

    selected = select_adaptive_faces(
        person,
        known,
        [*known, *eligible],
        foundation_count=FOUNDATION_COUNT,
        max_count=max_count,
    )
    additions = [item for item in selected if item.source.face_id not in known_ids]
    additions = additions[: max_count - len(known_ids)]
    uploads = [prepared[item.source.face_id].encoded for item in additions]
    addition_ids = {item.source.face_id for item in additions}
    newly_examined = tuple(sorted(examined_ids | known_ids | reviewed))
    newly_examined = tuple(face_id for face_id in newly_examined if face_id not in addition_ids)
    plan = PersonTrainingPlan(
        person=person,
        candidates=tuple(additions),
        uploads=tuple(uploads),
        examined_face_ids=newly_examined,
    )
    return plan, newly_examined


def _one_face_per_asset(
    candidates: Sequence[VectorFace], *, excluded_assets: set[str]
) -> tuple[list[VectorFace], set[str]]:
    """Keep the largest detected face per source image for Frigate uploads."""

    best_by_asset: dict[str, VectorFace] = {}
    discarded: set[str] = set()
    for candidate in candidates:
        face_id = candidate.source.face_id
        asset_id = candidate.source.asset_id
        if face_id is None:
            continue
        if asset_id in excluded_assets:
            discarded.add(face_id)
            continue
        previous = best_by_asset.get(asset_id)
        if previous is None:
            best_by_asset[asset_id] = candidate
            continue
        if _normalized_face_area(candidate) > _normalized_face_area(previous):
            discarded.add(previous.source.face_id)
            best_by_asset[asset_id] = candidate
        else:
            discarded.add(face_id)
    unique = sorted(
        best_by_asset.values(),
        key=lambda item: (
            item.source.taken_at,
            item.source.asset_id,
            item.source.face_id or "",
        ),
    )
    return unique, discarded


def _normalized_face_area(candidate: VectorFace) -> float:
    x1, y1, x2, y2 = candidate.source.box
    width, height = candidate.source.frame
    return ((x2 - x1) * (y2 - y1)) / (width * height)


def backup_registered_library(frigate, backup_dir: str | Path) -> dict[str, object]:
    """Back up registered Frigate face images before the one-time reset."""

    root = Path(backup_dir)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or any(root.iterdir()):
        raise ValueError("backup directory must be an empty, private directory")
    root.chmod(0o700)
    inventory = frigate.inventory()
    manifest: dict[str, object] = {
        "faces": {},
        "inventory": {name: list(filenames) for name, filenames in inventory.items()},
    }
    for name, filenames in inventory.items():
        face_dir = root / _safe_backup_component(name)
        face_dir.mkdir(parents=True, exist_ok=True)
        face_dir.chmod(0o700)
        rows = []
        for filename in filenames:
            body = frigate.face_image_bytes(name, filename)
            path = face_dir / _safe_backup_component(filename)
            path.write_bytes(body)
            path.chmod(0o600)
            rows.append(
                {
                    "filename": filename,
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
        manifest["faces"][name] = rows
    if frigate.inventory() != inventory:
        raise RuntimeError("Frigate face inventory changed during backup; refusing reset")
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").chmod(0o600)
    return manifest


def reset_registered_library(
    frigate,
    *,
    expected_inventory: Mapping[str, Sequence[str]] | None = None,
    on_delete=None,
) -> int:
    """Delete all registered face images from a freshly read inventory."""

    inventory = frigate.inventory()
    if expected_inventory is not None:
        expected = {
            name: tuple(sorted(filenames))
            for name, filenames in sorted(expected_inventory.items())
        }
        if inventory != expected:
            raise RuntimeError("Frigate face inventory changed after backup; refusing reset")
    deleted = 0
    for name, filenames in inventory.items():
        for offset in range(0, len(filenames), 512):
            batch = list(filenames[offset : offset + 512])
            if on_delete is not None:
                on_delete(name, batch, "deleting")
            frigate.delete_faces(name, batch)
            if set(batch) & set(frigate.inventory().get(name, ())):
                raise RuntimeError("Frigate face reset did not produce an empty registered library")
            deleted += len(batch)
            if on_delete is not None:
                on_delete(name, batch, "deleted")
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

    if not plan.people or any(
        len(person_plan.candidates) != len(person_plan.uploads)
        or not FOUNDATION_COUNT <= person_plan.target_count <= MAX_TRAINING_COUNT
        or any(not isinstance(upload, bytes) or not upload for upload in person_plan.uploads)
        for person_plan in plan.people
    ):
        raise ValueError("rebuild plan contains invalid preflighted uploads")

    registered_total = 0
    per_person_counts: list[int] = []

    for person_plan in plan.people:
        name = frigate_face_name(person_plan.person.name)
        frigate.create_face(name)
        successful = 0

        for image_bytes in person_plan.uploads:
            response = frigate.register_face(name, image_bytes)
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
