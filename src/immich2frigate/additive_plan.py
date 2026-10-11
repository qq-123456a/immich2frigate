"""Frozen, append-only enrollment for named Immich people."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import numpy as np

from .face_pipeline import (
    MIN_CROSS_PERSON_MARGIN,
    _prototype_margin,
    cross_person_margin,
    scan_person,
    select_foundation,
)
from .enrollment import backup_registered_library
from .frigate_names import frigate_face_name
from .frozen_plan import (
    _image_path,
    _inventory,
    digest,
    file_hash,
    register_verified,
    save_plan,
    verify_prepared_images,
    verify_sources,
)
from .identity_registry import validate_frigate_name
from .sync_state import write_json

_MAX_PREVIEW_CANDIDATES_PER_PERSON = 200


def additions_needed(current_count: int, max_images: int) -> int:
    """Top up a named Frigate label to the requested total without trimming it."""
    if (
        isinstance(current_count, bool)
        or not isinstance(current_count, int)
        or current_count < 0
        or isinstance(max_images, bool)
        or not isinstance(max_images, int)
        or not 1 <= max_images <= 30
    ):
        raise ValueError("current_count must be non-negative and max_images must be 1-30")
    return max(0, max_images - current_count)


def prepare_additive(
    root, *, target_names: Sequence[str], max_images: int, state, registry, frigate,
    immich, vectors, pipeline, teacher,
) -> dict:
    """Freeze strict Immich images for only the requested named people."""
    journal = _journal_path(state)
    if state.pending is not None or journal.exists():
        raise RuntimeError("an unresolved enrollment must be reviewed before preparation")
    if any((row.get("last_batch") or {}).get("status") == "pending" for row in state.roster):
        raise RuntimeError("an expansion batch is still awaiting validation")

    target = frigate.verify_target()
    recognition_profile = frigate.face_recognition_profile()
    state_sha, registry_sha = file_hash(state.path), file_hash(registry.path)
    inventory = _inventory(frigate)
    people = immich.people()
    by_name = {person.name: person for person in people}
    if any(not isinstance(name, str) for name in target_names):
        raise ValueError("target names must be strings")
    if len(target_names) != len(set(target_names)):
        raise ValueError("target names must be unique")
    targets = []
    for name in target_names:
        if not isinstance(name, str) or name not in by_name:
            raise ValueError(f"requested Immich person was not found: {name!r}")
        targets.append(by_name[name])
    if not targets:
        raise ValueError("at least one Immich person must be specified")
    additions_needed(0, max_images)
    actions = registry.reconcile(people, inventory)
    disallowed = {
        "RENAME_REQUIRED",
        "REMOTE_ALREADY_RENAMED",
        "RENAME_CONFLICT",
        "BOUND_LABELS_MISSING",
        "BOUND_LABEL_MISSING",
    }
    target_ids = {person.person_id for person in targets}
    if any(row["person_id"] in target_ids and row["status"] in disallowed for row in actions):
        raise RuntimeError("Immich and Frigate identity bindings need manual review")

    action_by_id = {row["person_id"]: row for row in actions}
    remote_names = {frigate_face_name(name).casefold(): name for name in inventory}
    reference_ids = set(target_ids)
    matched_labels: set[str] = set()
    for person in people:
        binding = registry.binding(person.person_id)
        action = action_by_id[person.person_id]
        label = (
            binding.frigate_name
            if binding is not None and binding.frigate_name in inventory
            else action.get("frigate_name")
            or remote_names.get(frigate_face_name(person.name).casefold())
        )
        if label in inventory and inventory[label]:
            reference_ids.add(person.person_id)
            matched_labels.add(label)
    if any(files and name not in matched_labels for name, files in inventory.items()):
        raise RuntimeError("a registered Frigate identity has no Immich face reference")
    labels: dict[str, str] = {}
    current_counts: dict[str, int] = {}
    for person in targets:
        action = action_by_id[person.person_id]
        binding = registry.binding(person.person_id)
        name = (
            action["frigate_name"]
            if action["status"] == "AUTO_BIND_REQUIRED"
            else binding.frigate_name
            if binding is not None
            else remote_names.get(
                frigate_face_name(person.name).casefold(), frigate_face_name(person.name)
            )
        )
        name = validate_frigate_name(name)
        if frigate_face_name(name).casefold() != frigate_face_name(person.name).casefold():
            raise RuntimeError("the locked Immich-to-Frigate label changed")
        if name.casefold() in {value.casefold() for value in labels.values()}:
            raise ValueError("Immich people have colliding Frigate labels")
        labels[person.person_id] = name
        current_counts[person.person_id] = len(inventory.get(name, ()))

    registry.preflight_bindings(
        (person.person_id, labels[person.person_id]) for person in targets
    )
    selected_people = [
        person for person in targets
        if additions_needed(current_counts[person.person_id], max_images) > 0
    ]
    locked_rows = {row["person_id"]: row for row in state.roster}
    for person in selected_people:
        row = locked_rows.get(person.person_id)
        if row is not None:
            face_ids = row.get("face_ids")
            images = row.get("images")
            if (
                not isinstance(face_ids, list)
                or any(not isinstance(face_id, str) for face_id in face_ids)
                or len(set(face_ids)) != len(face_ids)
                or len(face_ids) != current_counts[person.person_id]
            ):
                raise RuntimeError("locked face count differs from its source ledger")
            image_ids = [
                image.get("face_id") for image in images if isinstance(image, dict)
            ] if isinstance(images, list) else []
            filenames = [
                image.get("filename") for image in images if isinstance(image, dict)
            ] if isinstance(images, list) else []
            if (
                not isinstance(images, list)
                or len(images) != len(face_ids)
                or len(image_ids) != len(face_ids)
                or any(not isinstance(face_id, str) for face_id in image_ids)
                or len(set(image_ids)) != len(image_ids)
                or set(image_ids) != set(face_ids)
                or any(not isinstance(filename, str) for filename in filenames)
                or len(set(filenames)) != len(filenames)
            ):
                raise RuntimeError("locked source ledger is incomplete; refusing a targeted addition")
            if sorted(filenames) != sorted(inventory.get(labels[person.person_id], ())):
                raise RuntimeError("locked filenames differ from the Frigate inventory")
    skipped = [
        {
            "name": person.name,
            "frigate_name": labels[person.person_id],
            "current_count": current_counts[person.person_id],
            "reason": "over_limit" if current_counts[person.person_id] > max_images else "at_limit",
        }
        for person in targets
        if not additions_needed(current_counts[person.person_id], max_images)
    ]

    manifest = {
        "schema": 1,
        "kind": "additive",
        "created_at": time.time(),
        "target": asdict(target),
        "frigate_face_recognition": recognition_profile,
        "immich_origin": state.immich_origin,
        "years": state.years,
        "profile": pipeline.profile.metadata(),
        "teacher": teacher.profile(),
        "max_images": max_images,
        "candidate_preview_limit_per_person": _MAX_PREVIEW_CANDIDATES_PER_PERSON,
        "prototype_margin_version": 1,
        "target_people": [
            {"person_id": person.person_id, "name": person.name,
             "frigate_name": labels[person.person_id],
             "current_count": current_counts[person.person_id]}
            for person in targets
        ],
        "immich_roster": sorted(
            ({"person_id": person.person_id, "name": person.name} for person in people),
            key=lambda row: row["person_id"],
        ),
        "inventory": inventory,
        "state_sha256": state_sha,
        "registry_sha256": registry_sha,
        "existing_face_hashes": {},
        "skipped": skipped,
        "people": [],
    }
    files: dict[str, bytes] = {}
    if selected_people:
        pools = {
            person.person_id: scan_person(
                immich, vectors, person, pipeline, years=state.years,
                candidate_limit=_MAX_PREVIEW_CANDIDATES_PER_PERSON,
            )
            for person in people if person.person_id in reference_ids
        }
        existing_embeddings: dict[str, list[np.ndarray]] = {}
        for person in selected_people:
            count = current_counts[person.person_id]
            if not count:
                continue
            name = labels[person.person_id]
            hashes: dict[str, str] = {}
            vectors_for_existing: list[np.ndarray] = []
            for filename in inventory.get(name, ()):
                body = frigate.face_image_bytes(name, filename)
                hashes[filename] = hashlib.sha256(body).hexdigest()
                face = pipeline.evaluate_crop(body)
                vectors_for_existing.append(_unit(face.embedding))
            manifest["existing_face_hashes"][name] = hashes
            existing_embeddings[person.person_id] = vectors_for_existing

        confirmed: dict[str, bool] = {}

        def confirm(candidate) -> bool:
            source = candidate.source.source
            face_id = source.face_id
            if face_id not in confirmed:
                query = _unit(candidate.face.embedding)
                duplicate = any(
                    float(query @ row) >= 0.98
                    for row in existing_embeddings.get(source.person_id, ())
                )
                if duplicate:
                    confirmed[face_id] = False
                else:
                    result = teacher.confirm(
                        candidate.face.stored_webp,
                        exclude_asset_id=source.asset_id,
                        exclude_checksum=source.checksum,
                    )
                    confirmed[face_id] = (
                        result.confirmed and result.person_id == source.person_id
                    )
            return confirmed[face_id]

        for person in selected_people:
            additions = additions_needed(current_counts[person.person_id], max_images)
            rows = []
            selected = _select_additions(
                person, pools[person.person_id], additions, confirm, pools
            )
            if len(selected) < additions:
                manifest["skipped"].append({
                    "name": person.name,
                    "frigate_name": labels[person.person_id],
                    "current_count": current_counts[person.person_id],
                    "reason": "candidate_shortfall",
                    "planned_additions": len(selected),
                })
            if not selected:
                continue
            prototype_margins = [
                _prototype_margin(selected[index:index + 5], pools)
                for index in range(0, len(selected), 5)
            ]
            if any(
                margin is None or margin < MIN_CROSS_PERSON_MARGIN
                for margin in prototype_margins
            ):
                raise ValueError("a selected cohort failed the cross-person prototype margin")
            for candidate in selected:
                source = candidate.source.source
                margin = cross_person_margin(candidate, pools)
                if margin is None or margin < MIN_CROSS_PERSON_MARGIN:
                    raise ValueError("the selected face failed the cross-person margin")
                upload_file = f"{source.face_id}.upload.webp"
                registered_file = f"{source.face_id}.registered.webp"
                files[upload_file] = candidate.upload_bytes
                files[registered_file] = candidate.face.stored_webp
                rows.append({
                    "person_id": source.person_id,
                    "face_id": source.face_id,
                    "asset_id": source.asset_id,
                    "checksum": source.checksum,
                    "upload_file": upload_file,
                    "upload_sha256": hashlib.sha256(candidate.upload_bytes).hexdigest(),
                    "registered_file": registered_file,
                    "registered_sha256": hashlib.sha256(candidate.face.stored_webp).hexdigest(),
                    "embedding": candidate.face.embedding.tolist(),
                    "metrics": asdict(candidate.face.metrics),
                    "cross_person_margin": margin,
                })
            manifest["people"].append({
                "person_id": person.person_id,
                "name": person.name,
                "frigate_name": labels[person.person_id],
                "existing_count": current_counts[person.person_id],
                "prototype_margins": prototype_margins,
                "images": rows,
            })

    verify_sources(manifest, immich, vectors)
    if (
        {person.person_id: person.name for person in immich.people()}
        != {row["person_id"]: row["name"] for row in manifest["immich_roster"]}
        or
        frigate.verify_target() != target
        or frigate.face_recognition_profile() != recognition_profile
        or _inventory(frigate) != inventory
        or file_hash(state.path) != state_sha
        or file_hash(registry.path) != registry_sha
    ):
        raise RuntimeError("runtime changed during preparation; the Frigate library is unchanged")
    return save_plan(Path(root), manifest, files)


def load_additive_plan(root: str | Path) -> tuple[Path, dict]:
    root = Path(root)
    plan = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    plan_id = plan.pop("plan_id", None)
    if plan_id != digest(plan) or plan.get("schema") != 1 or plan.get("kind") != "additive":
        raise ValueError("frozen additive plan is invalid or has changed")
    plan["plan_id"] = plan_id
    if not isinstance(plan.get("people"), list) or not isinstance(plan.get("skipped"), list):
        raise ValueError("frozen additive plan has an invalid roster")
    max_images = plan.get("max_images")
    additions_needed(0, max_images)
    preview_limit = plan.get("candidate_preview_limit_per_person")
    if preview_limit is not None and (
        isinstance(preview_limit, bool) or not isinstance(preview_limit, int) or preview_limit < 5
    ):
        raise ValueError("frozen additive preview limit is invalid")
    prototype_margin_version = plan.get("prototype_margin_version")
    if prototype_margin_version is not None and (
        isinstance(prototype_margin_version, bool)
        or not isinstance(prototype_margin_version, int)
        or prototype_margin_version != 1
    ):
        raise ValueError("frozen additive prototype margin version is invalid")
    roster = plan.get("immich_roster")
    if not isinstance(roster, list):
        raise ValueError("frozen additive source roster is invalid")
    roster_by_id: dict[str, str] = {}
    roster_names: set[str] = set()
    for row in roster:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("frozen additive source roster is invalid")
        person_id = _uuid(row.get("person_id"))
        label = validate_frigate_name(frigate_face_name(row["name"]))
        if person_id in roster_by_id or label.casefold() in roster_names:
            raise ValueError("frozen additive source roster contains duplicate identities")
        roster_by_id[person_id] = row["name"]
        roster_names.add(label.casefold())
    targets = plan.get("target_people")
    if not isinstance(targets, list) or not targets:
        raise ValueError("frozen additive target list is invalid")
    target_ids: set[str] = set()
    target_labels: set[str] = set()
    for target in targets:
        person_id = _uuid(target["person_id"])
        label = validate_frigate_name(target["frigate_name"])
        count = target["current_count"]
        if (
            person_id in target_ids
            or roster_by_id.get(person_id) != target["name"]
            or frigate_face_name(target["name"]).casefold() != label.casefold()
            or label.casefold() in target_labels
            or isinstance(count, bool)
            or not isinstance(count, int)
            or len(plan["inventory"].get(label, ())) != count
        ):
            raise ValueError("frozen additive target identities conflict")
        target_ids.add(person_id)
        target_labels.add(label.casefold())
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    seen_faces: set[str] = set()
    seen_assets: set[tuple[str, str]] = set()
    for person in plan["people"]:
        person_id = _uuid(person["person_id"])
        label = validate_frigate_name(person["frigate_name"])
        if (
            person_id in seen_ids
            or person_id not in target_ids
            or label.casefold() in seen_names
            or roster_by_id.get(person_id) != person["name"]
            or frigate_face_name(person["name"]).casefold() != label.casefold()
        ):
            raise ValueError("frozen additive identities conflict")
        seen_ids.add(person_id)
        seen_names.add(label.casefold())
        existing_count = person["existing_count"]
        if (
            isinstance(existing_count, bool)
            or not isinstance(existing_count, int)
            or existing_count >= max_images
            or not 1 <= len(person["images"]) <= additions_needed(existing_count, max_images)
        ):
            raise ValueError("frozen additive plan exceeds the requested image limit")
        prototype_margins = person.get("prototype_margins")
        if prototype_margin_version == 1 and (
            not isinstance(prototype_margins, list)
            or len(prototype_margins) != (len(person["images"]) + 4) // 5
            or any(
                isinstance(margin, bool)
                or not isinstance(margin, (int, float))
                or not np.isfinite(margin)
                or margin < MIN_CROSS_PERSON_MARGIN
                for margin in prototype_margins
            )
        ):
            raise ValueError("frozen additive plan failed the cross-person prototype margin")
        inventory = plan["inventory"]
        if len(inventory.get(label, ())) != existing_count:
            raise ValueError("frozen additive inventory count changed")
        for image in person["images"]:
            face_id, asset_id = _uuid(image["face_id"]), _uuid(image["asset_id"])
            if face_id in seen_faces or (person_id, asset_id) in seen_assets:
                raise ValueError("frozen additive plan contains duplicate source images")
            seen_faces.add(face_id)
            seen_assets.add((person_id, asset_id))
            margin = image.get("cross_person_margin")
            if (
                isinstance(margin, bool)
                or not isinstance(margin, (int, float))
                or not np.isfinite(margin)
                or margin < MIN_CROSS_PERSON_MARGIN
            ):
                raise ValueError("frozen additive plan failed the cross-person margin")
            for kind in ("upload", "registered"):
                path = _image_path(root, image[kind + "_file"])
                if path.stat().st_size > 8 * 1024 * 1024 or hashlib.sha256(path.read_bytes()).hexdigest() != image[kind + "_sha256"]:
                    raise ValueError("frozen additive image bytes have changed")
            embedding = np.asarray(image["embedding"], dtype=np.float32)
            if embedding.shape != (512,) or not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
                raise ValueError("frozen additive embedding is invalid")
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("name"), str)
        or not isinstance(row.get("frigate_name"), str)
        or isinstance(row.get("current_count"), bool)
        or not isinstance(row.get("current_count"), int)
        for row in plan["skipped"]
    ):
        raise ValueError("frozen additive skipped target is invalid")
    skipped_by_label = {row["frigate_name"].casefold(): row for row in plan["skipped"]}
    if len(skipped_by_label) != len(plan["skipped"]):
        raise ValueError("frozen additive plan contains duplicate skipped targets")
    people_by_id = {_uuid(row["person_id"]): row for row in plan["people"]}
    for target in targets:
        skipped = skipped_by_label.get(target["frigate_name"].casefold())
        person = people_by_id.get(_uuid(target["person_id"]))
        needed = additions_needed(target["current_count"], max_images)
        planned = len(person["images"]) if person is not None else 0
        if needed == 0:
            expected_reason = "over_limit" if target["current_count"] > max_images else "at_limit"
            if (
                person is not None
                or skipped is None
                or skipped.get("name") != target["name"]
                or skipped.get("current_count") != target["current_count"]
                or skipped.get("reason") != expected_reason
            ):
                raise ValueError("frozen additive skipped target is inconsistent")
        elif planned < needed:
            planned_additions = skipped.get("planned_additions") if skipped is not None else None
            if (
                skipped is None
                or skipped.get("name") != target["name"]
                or skipped.get("current_count") != target["current_count"]
                or skipped.get("reason") != "candidate_shortfall"
                or isinstance(planned_additions, bool)
                or not isinstance(planned_additions, int)
                or planned_additions != planned
            ):
                raise ValueError("frozen additive candidate shortfall is inconsistent")
        elif skipped is not None:
            raise ValueError("frozen additive plan marks a complete target as skipped")
    if any(name not in {row["frigate_name"].casefold() for row in targets} for name in skipped_by_label):
        raise ValueError("frozen additive plan has an unknown skipped target")
    for name, hashes in plan.get("existing_face_hashes", {}).items():
        if name not in plan["inventory"] or not isinstance(hashes, dict):
            raise ValueError("frozen existing-face references are invalid")
        if any(filename not in plan["inventory"][name] for filename in hashes):
            raise ValueError("frozen existing-face inventory changed")
    return root, plan


def apply_additive_plan(
    root,
    *,
    backup_dir,
    state,
    registry,
    frigate,
    pipeline,
    teacher,
    immich,
    vectors,
) -> dict:
    journal = _journal_path(state)
    if journal.exists():
        raise RuntimeError("unfinished additive enrollment requires manual review")
    root, plan = load_additive_plan(root)
    _verify_context(plan, state, registry, frigate, pipeline, teacher, immich, vectors)
    if not plan["people"]:
        return {"status": "no_additions", "verified_uploads": 0, "skipped": plan["skipped"]}

    verify_prepared_images(root, plan, pipeline, teacher)
    registry.preflight_bindings(
        (person["person_id"], person["frigate_name"]) for person in plan["people"]
    )
    backup = Path(backup_dir)
    backup_registered_library(frigate, backup)
    shutil.copy2(state.path, backup / "sync-state.json")
    shutil.copy2(registry.path, backup / "identity-registry.json")
    (backup / "config.yml").write_bytes(frigate.raw_config_bytes())
    write_json(backup / "prepared-plan.json", plan)
    for path in backup.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        elif path.is_file():
            path.chmod(0o600)
    _verify_context(plan, state, registry, frigate, pipeline, teacher, immich, vectors)
    if (
        _inventory(frigate) != plan["inventory"]
        or file_hash(state.path) != plan["state_sha256"]
        or file_hash(registry.path) != plan["registry_sha256"]
    ):
        raise RuntimeError("runtime changed during backup; no Frigate images were added")

    completed: list[dict] = []
    write_json(journal, {
        "plan_id": plan["plan_id"],
        "backup_dir": str(backup),
        "phase": "prepared",
        "completed": completed,
    })
    for person in plan["people"]:
        name = person["frigate_name"]
        if name not in frigate.inventory():
            write_json(journal, {
                "plan_id": plan["plan_id"], "backup_dir": str(backup),
                "phase": "creating", "name": name, "completed": completed,
            })
            frigate.create_face(name)
            if name not in frigate.inventory():
                raise RuntimeError("Frigate face folder creation could not be verified")
        for image in person["images"]:
            write_json(journal, {
                "plan_id": plan["plan_id"], "backup_dir": str(backup),
                "phase": "registering",
                "pending": {"person_id": person["person_id"], "face_id": image["face_id"]},
                "completed": completed,
            })
            record = register_verified(
                frigate, name, {**image, "person_id": person["person_id"]},
                root, pipeline, teacher,
            )
            completed.append({**record, "person_id": person["person_id"], "name": name})
            write_json(journal, {
                "plan_id": plan["plan_id"], "backup_dir": str(backup),
                "phase": "registered", "completed": completed,
            })
        registry.bind(person["person_id"], name)

    expected = {name: list(files) for name, files in plan["inventory"].items()}
    for row in completed:
        expected.setdefault(row["name"], []).append(row["filename"])
    expected = {name: sorted(files) for name, files in sorted(expected.items())}
    actual = _inventory(frigate)
    if actual != expected or any(
        len(actual.get(person["frigate_name"], ()))
        != person["existing_count"] + len(person["images"])
        for person in plan["people"]
    ):
        raise RuntimeError("final face inventory differs from the append-only plan")
    _append_locked_ledger(state, completed)
    journal.unlink()
    return {
        "status": "applied",
        "registered_people": len(plan["people"]),
        "verified_uploads": len(completed),
        "registered_counts": {
            target["frigate_name"]: len(actual.get(target["frigate_name"], ()))
            for target in plan["target_people"]
        },
        "skipped": plan["skipped"],
    }


def _verify_context(plan, state, registry, frigate, pipeline, teacher, immich, vectors) -> None:
    if any((row.get("last_batch") or {}).get("status") == "pending" for row in state.roster):
        raise RuntimeError("an expansion batch is still awaiting validation")
    target = frigate.verify_target()
    if (
        plan["target"] != asdict(target)
        or plan["profile"] != pipeline.profile.metadata()
        or plan["teacher"] != teacher.profile()
        or plan["frigate_face_recognition"] != frigate.face_recognition_profile()
    ):
        raise RuntimeError("service, model, or profile changed after preparation")
    if (
        plan["state_sha256"] != file_hash(state.path)
        or plan["registry_sha256"] != file_hash(registry.path)
        or _inventory(frigate) != plan["inventory"]
        or state.pending is not None
    ):
        raise RuntimeError("state, identity registry, or Frigate inventory changed after preparation")
    people = {person.person_id: person.name for person in immich.people()}
    if people != {row["person_id"]: row["name"] for row in plan["immich_roster"]}:
        raise RuntimeError("the frozen Immich named-person roster changed after preparation")
    for row in plan["people"]:
        if people.get(row["person_id"]) != row["name"]:
            raise RuntimeError("a frozen Immich identity changed after preparation")
        if len(plan["inventory"].get(row["frigate_name"], ())) != row["existing_count"]:
            raise RuntimeError("an additive target's existing face count changed")
    for row in plan["target_people"]:
        if len(plan["inventory"].get(row["frigate_name"], ())) != row["current_count"]:
            raise RuntimeError("a requested target's existing face count changed")
    for name, hashes in plan.get("existing_face_hashes", {}).items():
        for filename, expected_hash in hashes.items():
            body = frigate.face_image_bytes(name, filename)
            if hashlib.sha256(body).hexdigest() != expected_hash:
                raise RuntimeError("an existing Frigate face changed after preparation")
    verify_sources(plan, immich, vectors)


def _unit(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("face embedding must be finite and non-zero")
    return value / norm


def _append_locked_ledger(state, completed) -> None:
    additions = {}
    for record in completed:
        additions.setdefault(record["person_id"], []).append({
            key: record[key]
            for key in ("face_id", "asset_id", "checksum", "filename", "sha256")
        })
    locked_ids = {row["person_id"] for row in state.roster}
    if not (set(additions) & locked_ids):
        return
    updated = []
    for row in state.roster:
        images = additions.get(row["person_id"], [])
        if not images:
            updated.append(row)
            continue
        face_ids = [image["face_id"] for image in images]
        updated.append({
            **row,
            "face_ids": [*row["face_ids"], *face_ids],
            "examined_face_ids": sorted(set(row.get("examined_face_ids", [])) | set(face_ids)),
            "images": [*row["images"], *images],
        })
    state.roster = updated
    state.save()


def _select_additions(person, candidates, count, teacher, references_by_person):
    remaining = list(candidates)
    selected = []
    while len(selected) < count:
        desired = min(5, count - len(selected))
        picked = []
        for batch_count in range(desired, 0, -1):
            try:
                batch = select_foundation(
                    person, remaining, teacher=teacher,
                    references_by_person=references_by_person, count=batch_count,
                )
            except ValueError as error:
                if not str(error).startswith(("fewer than ", "at least ")):
                    raise
                continue
            picked = batch[:desired]
            break
        if not picked:
            return selected
        selected.extend(picked)
        chosen_ids = {item.source.source.face_id for item in picked}
        chosen_vectors = [_unit(item.face.embedding) for item in picked]
        remaining = [
            item for item in remaining
            if item.source.source.face_id not in chosen_ids
            and not any(float(_unit(item.face.embedding) @ vector) >= 0.98 for vector in chosen_vectors)
        ]
        if len(picked) < desired:
            return selected
    return selected


def _uuid(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError("frozen additive IDs must be canonical UUIDs")
    return value


def _journal_path(state) -> Path:
    return state.path.with_name(state.path.name + ".additive-in-progress.json")
