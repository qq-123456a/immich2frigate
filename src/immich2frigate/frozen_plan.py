"""Private, frozen enrollment plans and verified sequential registration."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import numpy as np

from .enrollment import backup_registered_library, reset_registered_library
from .frigate_names import frigate_face_name
from .identity_registry import validate_frigate_name
from .sync_state import write_json


def prepare_foundation(root, *, state, registry, frigate, immich, vectors, pipeline, teacher) -> dict:
    """Freeze five verified images for each existing binding, without remote writes."""
    from .face_pipeline import scan_person, select_foundation, cross_person_margin, MIN_CROSS_PERSON_MARGIN
    if state.pending is not None or state.path.with_name(state.path.name + ".rebuild-in-progress.json").exists():
        raise RuntimeError("an unresolved enrollment must be recovered before preparation")
    target = frigate.verify_target()
    recognition_profile = frigate.face_recognition_profile()
    state_sha, registry_sha = file_hash(state.path), file_hash(registry.path)
    inventory = _inventory(frigate)
    people = {person.person_id: person for person in immich.people()}
    pools, bindings = {}, {}
    for row in state.roster:
        person = people.get(row["person_id"])
        binding = registry.binding(row["person_id"])
        if person is None or person.name != row["name"] or binding is None or binding.frigate_name != frigate_face_name(person.name):
            raise RuntimeError("the locked three-person identity roster changed")
        if len(inventory.get(binding.frigate_name, [])) != len(row["face_ids"]):
            raise RuntimeError("registered count differs from the source ledger")
        pools[person.person_id] = scan_person(immich, vectors, person, pipeline, years=state.years)
        bindings[person.person_id] = binding
        print(json.dumps({"preparation": "quality_scan", "eligible_candidates": len(pools[person.person_id])}), flush=True)
    if len(pools) != 3:
        raise ValueError("foundation must retain exactly three existing identities")
    if any(name not in {binding.frigate_name for binding in bindings.values()} for name in inventory):
        raise RuntimeError("unexpected registered identities; only the locked roster can be rebuilt")
    confirmed = {}
    def confirm(candidate):
        source = candidate.source.source
        key = source.face_id
        if key not in confirmed:
            result = teacher.confirm(candidate.face.stored_webp, exclude_asset_id=source.asset_id, exclude_checksum=source.checksum)
            confirmed[key] = result.confirmed and result.person_id == source.person_id
        return confirmed[key]
    selected = {person_id: select_foundation(people[person_id], pool, teacher=confirm, references_by_person=pools) for person_id, pool in pools.items()}
    for chosen in selected.values():
        for candidate in chosen:
            margin = cross_person_margin(candidate, selected)
            if margin is None or margin < MIN_CROSS_PERSON_MARGIN:
                raise ValueError("the final five-image prototypes failed the cross-person margin")
    manifest = {"schema": 1, "kind": "rebuild", "created_at": time.time(), "target": asdict(target), "frigate_face_recognition": recognition_profile, "immich_origin": state.immich_origin, "years": state.years, "profile": pipeline.profile.metadata(), "teacher": teacher.profile(), "inventory": inventory, "state_sha256": state_sha, "registry_sha256": registry_sha, "people": []}
    files = {}
    for person_id, chosen in selected.items():
        rows = []
        for candidate in chosen:
            source = candidate.source.source
            upload_name, registered_name = f"{source.face_id}.upload.webp", f"{source.face_id}.registered.webp"
            files[upload_name], files[registered_name] = candidate.upload_bytes, candidate.face.stored_webp
            rows.append({"person_id": person_id, "face_id": source.face_id, "asset_id": source.asset_id, "checksum": source.checksum, "upload_file": upload_name, "upload_sha256": hashlib.sha256(candidate.upload_bytes).hexdigest(), "registered_file": registered_name, "registered_sha256": hashlib.sha256(candidate.face.stored_webp).hexdigest(), "embedding": candidate.face.embedding.tolist(), "metrics": asdict(candidate.face.metrics)})
        manifest["people"].append({"person_id": person_id, "name": people[person_id].name, "frigate_name": bindings[person_id].frigate_name, "images": rows, "examined_face_ids": [candidate.source.source.face_id for candidate in pools[person_id]]})
    verify_sources(manifest, immich, vectors)
    if frigate.verify_target() != target or frigate.face_recognition_profile() != recognition_profile or _inventory(frigate) != inventory or file_hash(state.path) != state_sha or file_hash(registry.path) != registry_sha:
        raise RuntimeError("runtime changed during preparation; old library retained")
    return save_plan(Path(root), manifest, files)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_plan(root: Path, manifest: dict, files: dict[str, bytes]) -> dict:
    """Write a new directory only after all preparation gates have passed."""
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    for name, content in files.items():
        path = _image_path(root, name)
        path.write_bytes(content)
        path.chmod(0o600)
    manifest["plan_id"] = digest(manifest)
    write_json(root / "manifest.json", manifest)
    return manifest


def load_plan(root: str | Path) -> tuple[Path, dict]:
    root = Path(root)
    value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    plan_id = value.pop("plan_id", None)
    if plan_id != digest(value) or value.get("schema") != 1 or value.get("kind") not in {"rebuild", "expansion"}:
        raise ValueError("frozen plan is invalid or has changed")
    value["plan_id"] = plan_id
    if not isinstance(value.get("people"), list) or len(value["people"]) != 3:
        raise ValueError("frozen plan must keep the three-person roster")
    person_ids, names, face_ids, assets = set(), set(), set(), set()
    for person in value["people"]:
        person_id = _uuid(person["person_id"])
        name = validate_frigate_name(person["frigate_name"])
        if person_id in person_ids or name.casefold() in names or frigate_face_name(person["name"]) != name:
            raise ValueError("frozen identity bindings conflict")
        person_ids.add(person_id)
        names.add(name.casefold())
        images = person["images"]
        count = len(images)
        if (value["kind"] == "rebuild" and count != 5) or (value["kind"] == "expansion" and count > 2):
            raise ValueError("frozen plan exceeds the foundation or expansion limit")
        for item in images:
            face_id, asset_id = _uuid(item["face_id"]), _uuid(item["asset_id"])
            if face_id in face_ids or (person_id, asset_id) in assets:
                raise ValueError("frozen plan contains duplicate source images")
            face_ids.add(face_id)
            assets.add((person_id, asset_id))
            for kind in ("upload", "registered"):
                path = _image_path(root, item[kind + "_file"])
                if path.stat().st_size > 8 * 1024 * 1024 or file_hash(path) != item[kind + "_sha256"]:
                    raise ValueError("frozen image bytes have changed")
            vector = np.asarray(item["embedding"], dtype=np.float32)
            if vector.shape != (512,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                raise ValueError("frozen embedding is invalid")
    return root, value


def _image_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f-]{36}\.(upload|registered)\.webp", name):
        raise ValueError("unsafe frozen image path")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError("frozen image must remain inside its plan directory")
    return path


def _uuid(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError("plan IDs must be canonical UUIDs")
    return value


def verify_context(plan, state, registry, frigate, pipeline, teacher) -> None:
    target = frigate.verify_target()
    if plan["target"] != asdict(target) or plan["profile"] != pipeline.profile.metadata() or plan["teacher"] != teacher.profile():
        raise RuntimeError("service, model, or profile changed after preparation")
    if plan.get("frigate_face_recognition") != frigate.face_recognition_profile():
        raise RuntimeError("Frigate recognition settings changed after preparation")
    if plan["state_sha256"] != file_hash(state.path) or plan["registry_sha256"] != file_hash(registry.path):
        raise RuntimeError("identity or source ledger changed after preparation")
    roster = {item["person_id"]: item for item in state.roster}
    if set(roster) != {item["person_id"] for item in plan["people"]} or state.pending is not None:
        raise RuntimeError("roster changed or an upload is unresolved")
    for person in plan["people"]:
        binding = registry.binding(person["person_id"])
        if roster[person["person_id"]]["name"] != person["name"] or binding is None or binding.frigate_name != person["frigate_name"]:
            raise RuntimeError("locked identity binding changed")
        if plan["kind"] == "expansion" and len(roster[person["person_id"]]["face_ids"]) + len(person["images"]) > 30:
            raise ValueError("expansion would exceed thirty registered images")
    if _inventory(frigate) != plan["inventory"]:
        raise RuntimeError("Frigate library changed after preparation")
    if any(name not in {person["frigate_name"] for person in plan["people"]} for name in plan["inventory"]):
        raise RuntimeError("unexpected registered identity outside the locked roster")


def verify_sources(plan, immich, vectors) -> None:
    people = {person.person_id: person for person in immich.people()}
    for person in plan["people"]:
        current = people.get(person["person_id"])
        if current is None or current.name != person["name"]:
            raise RuntimeError("source identity was renamed or removed")
        source_rows = vectors.candidates_for_person(current, years=plan["years"], include_face_ids=tuple(item["face_id"] for item in person["images"]))
        by_id = {item.source.face_id: item.source for item in source_rows}
        for item in person["images"]:
            source = by_id.get(item["face_id"])
            if source is None or source.asset_id != item["asset_id"] or source.checksum != item["checksum"]:
                raise RuntimeError("a frozen source face was removed or reassigned")


def _inventory(frigate) -> dict:
    return {name: list(files) for name, files in frigate.inventory().items()}


def register_verified(frigate, name, item, root, pipeline, teacher) -> dict:
    """A count alone is insufficient: check the only new file and its bytes."""
    before = _inventory(frigate)
    response = frigate.register_face(name, _image_path(root, item["upload_file"]).read_bytes())
    if response.get("success") is not True:
        raise RuntimeError("Frigate did not confirm registration; journal retained")
    after = _inventory(frigate)
    added = set(after.get(name, [])) - set(before.get(name, []))
    if len(added) != 1 or set(before.get(name, [])) - set(after.get(name, [])):
        raise RuntimeError("registration did not create exactly one new file")
    filename = added.pop()
    expected = {key: list(files) for key, files in before.items()}
    expected[name] = sorted([*expected.get(name, []), filename])
    if after != dict(sorted(expected.items())):
        raise RuntimeError("another library write occurred during registration")
    content = frigate.face_image_bytes(name, filename)
    if hashlib.sha256(content).hexdigest() != item["registered_sha256"]:
        raise RuntimeError("Frigate stored a different crop than the frozen plan")
    face = pipeline.evaluate_crop(content)
    if not face.eligible:
        raise RuntimeError("registered crop failed the strict quality profile")
    actual, frozen = np.asarray(face.embedding), np.asarray(item["embedding"])
    cosine = float(actual @ frozen / (np.linalg.norm(actual) * np.linalg.norm(frozen)))
    if cosine < 0.99999:
        raise RuntimeError("Frigate readback embedding differs from the prepared crop")
    confirmation = teacher.confirm(content, exclude_asset_id=item["asset_id"], exclude_checksum=item["checksum"])
    if not confirmation.confirmed or confirmation.person_id != item["person_id"]:
        raise RuntimeError("Immich cannot confirm the registered crop")
    return {"face_id": item["face_id"], "asset_id": item["asset_id"], "checksum": item["checksum"], "filename": filename, "sha256": hashlib.sha256(content).hexdigest()}


def verify_prepared_images(root, plan, pipeline, teacher) -> None:
    """Recheck every frozen upload before touching the current library."""
    for person in plan["people"]:
        for item in person["images"]:
            crop = pipeline.simulate_upload(_image_path(root, item["upload_file"]).read_bytes())
            if hashlib.sha256(crop.stored_webp).hexdigest() != item["registered_sha256"]:
                raise RuntimeError("frozen upload no longer produces the expected Frigate crop")
            face = pipeline.evaluate_crop(crop.stored_webp)
            actual, frozen = np.asarray(face.embedding), np.asarray(item["embedding"])
            cosine = float(actual @ frozen / (np.linalg.norm(actual) * np.linalg.norm(frozen)))
            if not face.eligible or cosine < 0.99999:
                raise RuntimeError("frozen crop failed the current quality or embedding check")
            confirmation = teacher.confirm(crop.stored_webp, exclude_asset_id=item["asset_id"], exclude_checksum=item["checksum"])
            if not confirmation.confirmed or confirmation.person_id != person["person_id"]:
                raise RuntimeError("Immich cannot confirm a frozen image; current library retained")


def apply_plan(root, plan, *, state, registry, frigate, pipeline, teacher, immich, vectors, backup_dir=None, validation_report=None) -> dict:
    journal = state.path.with_name(state.path.name + ".rebuild-in-progress.json")
    if journal.exists():
        raise RuntimeError("unfinished enrollment journal requires recovery before further writes")
    root, current_plan = load_plan(root)
    if current_plan != plan:
        raise RuntimeError("frozen plan changed before application")
    verify_context(plan, state, registry, frigate, pipeline, teacher)
    verify_sources(plan, immich, vectors)
    state.preflight()
    if plan["kind"] == "expansion":
        from .validation import calibration_gate
        from .expansion import check_validation_context
        check_validation_context(validation_report, state, frigate, pipeline, teacher)
        if validation_report is None or any(not calibration_gate(validation_report, person["person_id"])["passed"] for person in plan["people"] if person["images"]):
            raise RuntimeError("expansion requires current, passing calibration evidence")
        evidence = plan.get("validation_events")
        if (not isinstance(evidence, dict) or not evidence
                or validation_report.get("epoch") != plan.get("validation_epoch")
                or digest(evidence) != plan.get("validation_digest")
                or any(validation_report.get("events", {}).get(event_id) != row for event_id, row in evidence.items())):
            raise RuntimeError("validation evidence changed after preparation")
        if not any(person["images"] for person in plan["people"]):
            return {"status": "no_improving_candidates", "verified_uploads": 0, "registered_counts": [len(person["face_ids"]) for person in state.roster]}
        backup_dir = root / "before-apply"
    verify_prepared_images(root, plan, pipeline, teacher)
    if plan["kind"] in {"rebuild", "expansion"}:
        if backup_dir is None:
            raise ValueError("rebuild requires a fresh backup directory")
        backup = Path(backup_dir)
        backup.mkdir(parents=True, exist_ok=False)
        backup.chmod(0o700)
        backup_registered_library(frigate, backup)
        shutil.copy2(state.path, backup / "sync-state.json")
        shutil.copy2(registry.path, backup / "identity-registry.json")
        (backup / "config.yml").write_bytes(frigate.raw_config_bytes())
        write_json(backup / "prepared-plan.json", plan)
        for path in backup.iterdir():
            if path.is_file():
                path.chmod(0o600)
    verify_context(plan, state, registry, frigate, pipeline, teacher)
    verify_sources(plan, immich, vectors)
    if load_plan(root)[1] != plan:
        raise RuntimeError("frozen plan changed during backup")
    write_json(journal, {"plan_id": plan["plan_id"], "plan_dir": str(root), "backup_dir": str(backup_dir), "phase": "prepared", "completed": []})
    if plan["kind"] == "rebuild":
        def deletion(name, batch, phase):
            write_json(journal, {"plan_id": plan["plan_id"], "plan_dir": str(root), "backup_dir": str(backup_dir), "phase": phase, "delete": {"name": name, "files": batch}, "completed": []})
        reset_registered_library(frigate, expected_inventory=plan["inventory"], on_delete=deletion)
    completed = []
    next_roster = []
    enrolled_at = time.time()
    for person in plan["people"]:
        old = next(item for item in state.roster if item["person_id"] == person["person_id"])
        name = person["frigate_name"]
        if person["images"] and name not in frigate.inventory():
            write_json(journal, {"plan_id": plan["plan_id"], "plan_dir": str(root), "backup_dir": str(backup_dir), "phase": "creating", "name": name, "completed": completed})
            frigate.create_face(name)
            if name not in frigate.inventory():
                raise RuntimeError("Frigate face folder creation could not be verified")
        records = [] if plan["kind"] == "rebuild" else list(old.get("images", []))
        for image in person["images"]:
            write_json(journal, {"plan_id": plan["plan_id"], "plan_dir": str(root), "backup_dir": str(backup_dir), "phase": "registering", "pending": {"person_id": person["person_id"], "face_id": image["face_id"]}, "completed": completed})
            record = register_verified(frigate, name, {**image, "person_id": person["person_id"]}, root, pipeline, teacher)
            records.append(record)
            completed.append(record)
        known = [] if plan["kind"] == "rebuild" else list(old["face_ids"])
        known.extend(image["face_id"] for image in person["images"])
        updated = {**old, "face_ids": known, "examined_face_ids": sorted(set(old.get("examined_face_ids", [])) | set(person.get("examined_face_ids", [])) | set(known)), "images": records, "profile": plan["profile"], "enrolled_at": enrolled_at, "plan_id": plan["plan_id"]}
        updated.pop("last_batch", None)
        if plan["kind"] == "expansion":
            updated["last_batch"] = {"plan_id": plan["plan_id"], "status": "pending", "added": [record for record in completed if record["face_id"] in {image["face_id"] for image in person["images"]}], "previous_enrolled_at": old["enrolled_at"], "previous_profile": old["profile"], "previous_plan_id": old["plan_id"]}
        next_roster.append(updated)
    inventory = _inventory(frigate)
    expected_names = {person["frigate_name"] for person in plan["people"]}
    if any(inventory.get(person["frigate_name"], []) != sorted(image["filename"] for image in row["images"]) for person, row in zip(plan["people"], next_roster)) or any(files for name, files in inventory.items() if name not in expected_names):
        raise RuntimeError("final registered inventory does not match the new ledger")
    state.roster = next_roster
    state.pending = None
    state.save()
    journal.unlink()
    return {"plan_id": plan["plan_id"], "registered_counts": [len(person["face_ids"]) for person in state.roster], "verified_uploads": len(completed)}


def reconcile_expansion(*, state, registry, frigate, pipeline, teacher, report) -> dict | None:
    """Accept a batch on new holdout events, or withdraw it on a new conflict."""
    from .expansion import verify_validation_context
    from .validation import final_gate

    pending = [row for row in state.roster if (row.get("last_batch") or {}).get("status") == "pending"]
    if not pending:
        return None
    verify_validation_context(report, state, frigate, pipeline, teacher)
    batches = {row["last_batch"]["plan_id"] for row in pending}
    if len(pending) != 3 or len(batches) != 1:
        raise RuntimeError("pending expansion metadata is inconsistent")
    batch_id = batches.pop()
    if not isinstance(batch_id, str) or not re.fullmatch(r"[0-9a-f]{64}", batch_id):
        raise RuntimeError("pending batch ID is invalid")
    conflict = any(row.get("category") == "conflict" for row in report.get("events", {}).values())
    if not conflict:
        if all(final_gate(report, row["person_id"])["passed"] for row in state.roster):
            for row in state.roster:
                row["last_batch"]["status"] = "accepted"
            state.save()
            return {"batch_status": "accepted", "plan_id": batch_id}
        return {"batch_status": "awaiting_new_holdout"}
    journal = state.path.with_name(state.path.name + ".rebuild-in-progress.json")
    if state.pending is not None or journal.exists():
        raise RuntimeError("unresolved writes prevent automatic batch withdrawal")
    before = _inventory(frigate)
    expected, next_roster, deletions = {}, [], []
    restored_at = time.time()
    for row in state.roster:
        binding = registry.binding(row["person_id"])
        if binding is None or binding.frigate_name != frigate_face_name(row["name"]):
            raise RuntimeError("identity binding changed before withdrawal")
        name = binding.frigate_name
        if before.get(name) != sorted(item["filename"] for item in row["images"]):
            raise RuntimeError("library differs from the source ledger before withdrawal")
        batch = row["last_batch"]
        additions = batch["added"]
        if any(item not in row["images"] for item in additions):
            raise RuntimeError("withdrawal records do not belong to this library")
        for item in additions:
            if hashlib.sha256(frigate.face_image_bytes(name, item["filename"])).hexdigest() != item["sha256"]:
                raise RuntimeError("batch image changed before withdrawal")
        ids = {item["face_id"] for item in additions}
        records = [item for item in row["images"] if item["face_id"] not in ids]
        expected[name] = sorted(item["filename"] for item in records)
        if len(records) < 5:
            raise RuntimeError("withdrawal cannot reduce a foundation below five images")
        updated = {**row, "face_ids": [face_id for face_id in row["face_ids"] if face_id not in ids], "images": records,
                   "enrolled_at": restored_at, "profile": batch["previous_profile"], "plan_id": batch["previous_plan_id"],
                   "rejected_face_ids": sorted(set(row.get("rejected_face_ids", [])) | ids)}
        updated.pop("last_batch")
        next_roster.append(updated)
        if additions:
            deletions.append((name, [item["filename"] for item in additions]))
    if set(before) != set(expected):
        raise RuntimeError("unexpected identities prevent automatic withdrawal")
    state.preflight()
    completed = []
    for name, files in deletions:
        write_json(journal, {"plan_id": batch_id, "phase": "withdrawing", "pending": {"name": name, "files": files}, "completed": completed, "previous_roster": state.roster})
        frigate.delete_faces(name, files)
        if set(files) & set(frigate.inventory().get(name, ())):
            raise RuntimeError("batch withdrawal could not be verified; journal retained")
        completed.append({"name": name, "files": files})
    if _inventory(frigate) != dict(sorted(expected.items())):
        raise RuntimeError("withdrawn library differs from the expected baseline")
    state.roster = next_roster
    state.save()
    write_json(state.path.parent / "withdrawals" / f"{batch_id}.json", {"plan_id": batch_id, "reason": "new_event_conflict", "removed": completed, "report": report})
    journal.unlink()
    return {"batch_status": "withdrawn", "plan_id": batch_id, "removed_images": sum(len(row["files"]) for row in completed)}
