"""Prepare calibration-backed, frozen face-library expansion proposals."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .face_pipeline import (
    MIN_CROSS_PERSON_MARGIN,
    PreparedCandidate,
    _unit,
    cross_person_margin,
)
from .frigate018 import build_class_mean, reported_confidence
from .frigate_names import frigate_face_name
from .selection import _identity_safe_mask, _row_normalize, _validated_matrix
from .upload_image import prepare_candidate_upload
from .enrollment import _one_face_per_asset
from .validation import calibration_gate

_SHA256_FILE = re.compile(r"^[0-9a-f]{64}\.(?:webp|png|jpe?g)$")
_MAX_BATCH = 2
_MAX_LIBRARY = 30
_MIN_GROUP_IMAGES = 3
_MIN_CONFIDENCE = 0.95
_MAX_REPORT_AGE_SECONDS = 120


def prepare_expansion(
    root,
    *,
    validation_report: dict,
    state,
    registry,
    frigate,
    immich,
    vectors,
    pipeline,
    teacher,
) -> dict:
    """Freeze at most two new, source-confirmed images per existing identity."""
    from .frozen_plan import file_hash, save_plan, verify_sources

    validation_context = check_validation_context(validation_report, state, frigate, pipeline, teacher)
    if state.pending is not None or state.path.with_name(state.path.name + ".rebuild-in-progress.json").exists():
        raise RuntimeError("an unresolved enrollment must be recovered before expansion")
    if len(state.roster) != 3:
        raise ValueError("expansion requires the locked three-person roster")
    target = frigate.verify_target()
    state_sha, registry_sha = file_hash(state.path), file_hash(registry.path)
    inventory = _inventory(frigate)
    people = {person.person_id: person for person in immich.people()}
    pools: dict[str, list[PreparedCandidate]] = {}
    known: dict[str, list[PreparedCandidate]] = {}
    bindings = {}
    rows_by_person = {row["person_id"]: row for row in state.roster}
    if len(rows_by_person) != 3:
        raise ValueError("expansion roster has duplicate identities")

    for row in state.roster:
        person_id = row["person_id"]
        person = people.get(person_id)
        binding = registry.binding(person_id)
        if person is None or person.name != row["name"] or binding is None:
            raise RuntimeError("the locked Immich and Frigate identity roster changed")
        if binding.frigate_name != frigate_face_name(person.name):
            raise RuntimeError("the registry label does not match the locked person")
        bindings[person_id] = binding
        known[person_id] = _current_references(row, person, binding.frigate_name, inventory, frigate, vectors, pipeline, state.years)
        pools[person_id] = _scan_new_candidates(row, person, immich, vectors, pipeline, state.years)

    references = {person_id: [*known[person_id], *pools[person_id]] for person_id in rows_by_person}
    proposed: dict[str, list[PreparedCandidate]] = {}
    globally_selected: list[PreparedCandidate] = []
    for person_id, row in rows_by_person.items():
        room = _expansion_limit(len(row["face_ids"]))
        candidates = [
            item for item in pools[person_id]
            if (margin := cross_person_margin(item, references)) is not None
            and margin >= MIN_CROSS_PERSON_MARGIN
        ]
        picked = []
        if room == 0:
            proposed[person_id] = picked
            continue
        all_known = [item for values in known.values() for item in values]
        deduplicated = _deduplicate_candidates(candidates, [*all_known, *globally_selected])
        for candidate in deduplicated:
            source = candidate.source.source
            result = teacher.confirm(
                candidate.face.stored_webp,
                exclude_asset_id=source.asset_id,
                exclude_checksum=source.checksum,
            )
            if result.confirmed and result.person_id == person_id:
                picked.append(candidate)
                globally_selected.append(candidate)
                if len(picked) == room:
                    break
        proposed[person_id] = picked

    archive = state.path.parent / "validation" / "train-archive"
    groups = _calibration_groups(validation_report, archive, pipeline)
    current_prototypes = {
        person_id: build_class_mean([item.face.embedding for item in known[person_id]])
        for person_id in rows_by_person
    }
    if any(proposed.values()):
        trial = _prototypes_with_proposals(current_prototypes, known, proposed)
        before = _validation_scores(current_prototypes, groups)
        after = _validation_scores(trial, groups)
        if not _proposal_improves(before, after) or not _final_prototype_margins(proposed, trial):
            proposed = {person_id: [] for person_id in rows_by_person}

    # The watcher refreshes report.json periodically. Keep the epoch fixed, but
    # accept newer calibration rows after rechecking the current report and gates.
    latest_report = _latest_validation_report(validation_report, state)
    latest_context = check_validation_context(latest_report, state, frigate, pipeline, teacher)
    if latest_context["epoch"] != validation_context["epoch"]:
        raise RuntimeError("calibration epoch changed during expansion preparation")
    old_events = _calibration_report_events(validation_report)
    new_events = _calibration_report_events(latest_report)
    if any(new_events.get(key) != value for key, value in old_events.items()):
        raise RuntimeError("existing calibration evidence changed during preparation")
    latest_groups = _calibration_groups(latest_report, archive, pipeline)
    if any(proposed.values()):
        trial = _prototypes_with_proposals(current_prototypes, known, proposed)
        if not _proposal_improves(
            _validation_scores(current_prototypes, latest_groups),
            _validation_scores(trial, latest_groups),
        ) or not _final_prototype_margins(proposed, trial):
            proposed = {person_id: [] for person_id in rows_by_person}
    validation_events = {event_id: latest_report["events"][event_id] for event_id in latest_groups}

    manifest = {
        "schema": 1,
        "kind": "expansion",
        "created_at": datetime.now(UTC).timestamp(),
        "target": asdict(target),
        "frigate_face_recognition": latest_report["profile"]["frigate_face_recognition"],
        "immich_origin": state.immich_origin,
        "years": state.years,
        "profile": pipeline.profile.metadata(),
        "teacher": teacher.profile(),
        "inventory": inventory,
        "state_sha256": state_sha,
        "registry_sha256": registry_sha,
        "validation_epoch": validation_context["epoch"],
        "validation_events": validation_events,
        "validation_digest": _json_digest(validation_events),
        "people": [],
    }
    files: dict[str, bytes] = {}
    for person_id, row in rows_by_person.items():
        binding = bindings[person_id]
        images = []
        for candidate in proposed[person_id]:
            source = candidate.source.source
            upload_file = f"{source.face_id}.upload.webp"
            registered_file = f"{source.face_id}.registered.webp"
            files[upload_file] = candidate.upload_bytes
            files[registered_file] = candidate.face.stored_webp
            images.append({
                "person_id": person_id,
                "face_id": source.face_id,
                "asset_id": source.asset_id,
                "checksum": source.checksum,
                "upload_file": upload_file,
                "upload_sha256": hashlib.sha256(candidate.upload_bytes).hexdigest(),
                "registered_file": registered_file,
                "registered_sha256": hashlib.sha256(candidate.face.stored_webp).hexdigest(),
                "embedding": candidate.face.embedding.tolist(),
                "metrics": asdict(candidate.face.metrics),
            })
        manifest["people"].append({
            "person_id": person_id,
            "name": row["name"],
            "frigate_name": binding.frigate_name,
            "images": images,
            "examined_face_ids": sorted({
                *row.get("examined_face_ids", []),
                *(item.source.source.face_id for item in pools[person_id]),
            }),
        })

    verify_sources(manifest, immich, vectors)
    if (
        frigate.verify_target() != target
        or _inventory(frigate) != inventory
        or file_hash(state.path) != state_sha
        or file_hash(registry.path) != registry_sha
    ):
        raise RuntimeError("runtime changed during preparation; no expansion was frozen")
    latest_report = _latest_validation_report(validation_report, state)
    latest_context = check_validation_context(latest_report, state, frigate, pipeline, teacher)
    latest_events = _calibration_report_events(latest_report)
    if latest_context["epoch"] != validation_context["epoch"] or any(
        latest_events.get(key) != value for key, value in old_events.items()
    ):
        raise RuntimeError("calibration evidence changed during preparation")
    validation_events = {event_id: latest_report["events"][event_id] for event_id in latest_groups}
    manifest["validation_events"] = validation_events
    manifest["validation_digest"] = _json_digest(validation_events)
    return save_plan(Path(root), manifest, files)


def check_validation_context(report, state, frigate, pipeline, teacher) -> dict[str, str]:
    """Require a current, matching report and passing calibration gates."""
    if state.pending is not None:
        raise ValueError("an unresolved enrollment must be recovered before expansion")
    if any((row.get("last_batch") or {}).get("status") == "pending" for row in state.roster):
        raise ValueError("an expansion batch is still awaiting validation")
    context = verify_validation_context(report, state, frigate, pipeline, teacher)
    face_profile = report["profile"].get("frigate_face_recognition", {})
    if face_profile.get("recognition_threshold") != 0.95 or face_profile.get("min_faces") != 3:
        raise ValueError("Frigate face-recognition settings must use threshold 0.95 and min_faces 3")
    for row in state.roster:
        gate = calibration_gate(report, row["person_id"])
        if not gate["passed"]:
            raise ValueError(f"calibration gate failed for {row['name']!r}")
    return context


def verify_validation_context(report, state, frigate, pipeline, teacher) -> dict[str, str]:
    """Verify fresh report epoch/profile/inventory without requiring a completed batch."""
    if not isinstance(report, dict) or report.get("fresh") is not True:
        raise ValueError("current validation evidence is required")
    try:
        updated = datetime.fromisoformat(report["updated_at"].replace("Z", "+00:00"))
        age = (datetime.now(UTC) - updated).total_seconds()
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("validation timestamp is invalid") from None
    if age < 0 or age > _MAX_REPORT_AGE_SECONDS:
        raise ValueError("validation evidence is older than 120 seconds")
    if len(state.roster) != 3:
        raise ValueError("validation state does not match the locked roster")
    profile = pipeline.profile.metadata()
    expected_profile = {
        "faces": {row["person_id"]: row.get("profile") for row in state.roster},
        "pipeline": profile,
        "teacher": teacher.profile(),
        "frigate_face_recognition": frigate.face_recognition_profile(),
        "frigate_target": asdict(frigate.verify_target()),
    }
    if report.get("profile") != expected_profile:
        raise ValueError("validation profile differs from current state or models")
    inventory_digest = _inventory_digest({name: files for name, files in _inventory(frigate).items() if name != "train"})
    if report.get("inventory_digest") != inventory_digest:
        raise ValueError("validation evidence was collected against another Frigate inventory")
    enrolled = sorted(str(row.get("enrolled_at")) for row in state.roster)
    if any(not row.get("enrolled_at") for row in state.roster):
        raise ValueError("roster enrollment timestamps are missing")
    epoch_seed = json.dumps(expected_profile, sort_keys=True, default=str, separators=(",", ":"))
    expected_epoch = hashlib.sha256(f"{epoch_seed}|{'|'.join(enrolled)}|{inventory_digest}".encode()).hexdigest()
    if report.get("epoch") != expected_epoch:
        raise ValueError("validation epoch does not match the current roster and inventory")
    return {"epoch": expected_epoch, "digest": _json_digest(_calibration_report_events(report))}


def _calibration_report_events(report):
    events = report.get("events", {}) if isinstance(report, dict) else {}
    return {
        event_id: row for event_id, row in events.items()
        if isinstance(event_id, str) and isinstance(row, dict)
        and row.get("partition") == "calibration"
        and row.get("category") in {"agree", "conflict", "frigate_unknown"}
        and isinstance(row.get("teacher_person_id"), str)
    }


def _json_digest(value):
    from .frozen_plan import digest

    return digest(value)


def _latest_validation_report(initial, state):
    path = Path(state.path).parent / "validation" / "report.json"
    if not path.is_file():
        return initial
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("the latest validation report is invalid")
    return report


def _current_references(row, person, frigate_name, inventory, frigate, vectors, pipeline, years):
    records = row.get("images")
    if not isinstance(records, list) or len(records) != len(row["face_ids"]):
        raise ValueError("state is missing the registered source ledger")
    by_face = {item.get("face_id"): item for item in records if isinstance(item, dict)}
    if set(by_face) != set(row["face_ids"]):
        raise ValueError("registered source ledger does not match the face IDs")
    filenames = tuple(sorted(item.get("filename", "") for item in records))
    if tuple(sorted(inventory.get(frigate_name, ()))) != filenames:
        raise RuntimeError("registered Frigate images differ from the source ledger")
    sources = vectors.candidates_for_person(person, years=years, include_face_ids=tuple(row["face_ids"]))
    by_id = {item.source.face_id: item for item in sources}
    if not set(row["face_ids"]) <= set(by_id):
        raise RuntimeError("an enrolled source face is missing from Immich vectors")
    result = []
    for filename in filenames:
        record = next(item for item in records if item["filename"] == filename)
        source = by_id[record["face_id"]]
        if source.source.asset_id != record["asset_id"] or source.source.checksum != record["checksum"]:
            raise RuntimeError("registered source metadata changed")
        body = frigate.face_image_bytes(frigate_name, filename)
        if hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise RuntimeError("a registered Frigate image differs from the source ledger")
        face = pipeline.evaluate_crop(body)
        if not face.eligible:
            raise RuntimeError("a current registered image failed the strict profile")
        result.append(PreparedCandidate(source, body, face, identity_safe=True))
    return result


def _scan_new_candidates(row, person, immich, vectors, pipeline, years):
    known_ids = set(row["face_ids"])
    rejected = set(row.get("rejected_face_ids", []))
    known_sources = vectors.candidates_for_person(person, years=years, include_face_ids=tuple(known_ids))
    known = [item for item in known_sources if item.source.face_id in known_ids]
    if len(known) != len(known_ids):
        raise RuntimeError("an enrolled Immich source is unavailable")
    known_assets = {item.source.asset_id for item in known}
    known_checksums = {item.source.checksum.casefold() for item in known}
    embedded = vectors.candidates_for_person(person, years=years, include_face_ids=tuple(known_ids))
    embedded, _ = _one_face_per_asset(embedded, excluded_assets=set())
    matrix = _row_normalize(_validated_matrix(person, embedded, attr="face_embedding"))
    safe = _identity_safe_mask(matrix, minimum_keep=5)
    safe_pool = [item for index, item in enumerate(embedded) if safe[index]]
    output = []
    for source in safe_pool:
        item = source.source
        if (
            item.face_id in known_ids or item.face_id in rejected
            or item.asset_id in known_assets or item.checksum.casefold() in known_checksums
        ):
            continue
        try:
            upload = prepare_candidate_upload(item, immich.preview(item.asset_id))
            face = pipeline.evaluate(upload.encoded, upload.face_box)
        except ValueError:
            continue
        if face.eligible:
            output.append(PreparedCandidate(source, upload.encoded, face, identity_safe=True))
    return output


def _calibration_groups(report, archive, pipeline):
    groups = {}
    for event_id, row in report.get("events", {}).items():
        if (
            not isinstance(row, dict)
            or row.get("partition") != "calibration"
            or row.get("category") not in {"agree", "conflict", "frigate_unknown"}
            or not isinstance(row.get("teacher_person_id"), str)
        ):
            continue
        selected_hash = row.get("archive_sha256")
        paths = row.get("archive_paths")
        if not isinstance(selected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", selected_hash):
            continue
        if not isinstance(paths, list) or not any(isinstance(name, str) and name.startswith(selected_hash + ".") for name in paths):
            continue
        faces = []
        for name in paths:
            if not isinstance(name, str) or not _SHA256_FILE.fullmatch(name):
                continue
            path = archive / name
            if path.is_symlink() or path.resolve().parent != archive.resolve() or not path.is_file():
                continue
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != path.stem:
                continue
            try:
                face = pipeline.evaluate_crop(body)
            except Exception:
                continue
            if face.eligible:
                faces.append(face)
        if len(faces) >= _MIN_GROUP_IMAGES:
            groups[event_id] = {
                "person_id": row["teacher_person_id"],
                "category": row["category"],
                "embeddings": [face.embedding for face in faces],
                "sharpness": float(np.mean([face.metrics.sharpness for face in faces])),
            }
    return groups


def _validation_scores(prototypes, groups):
    rows = [row for row in groups.values() if row["person_id"] in prototypes]
    per_person = {person_id: 0 for person_id in prototypes}
    correct = conflicts = 0
    for row in rows:
        person_id = row["person_id"]
        per_person[person_id] += 1
        query = _unit(build_class_mean(row["embeddings"]))
        confidence = {
            identity: reported_confidence(float(np.dot(query, _unit(center))), row["sharpness"])
            for identity, center in prototypes.items()
        }
        winner = max(confidence, key=lambda identity: (confidence[identity], identity))
        if confidence[winner] >= _MIN_CONFIDENCE:
            if winner == person_id:
                correct += 1
            else:
                conflicts += 1
    return {
        "coverage": correct / len(rows) if rows else 0.0,
        "conflicts": conflicts,
        "events": len(rows),
        "events_by_person": per_person,
    }


def _proposal_improves(before, after):
    counts = before.get("events_by_person", {})
    return (
        before["events"] >= 20
        and after["events"] >= 20
        and (not counts or all(count >= 20 for count in counts.values()))
        and after["coverage"] > before["coverage"]
        and after["conflicts"] <= before["conflicts"]
    )


def _prototypes_with_proposals(current, known, proposed):
    trial = dict(current)
    for person_id, picked in proposed.items():
        if picked:
            trial[person_id] = build_class_mean([
                *(item.face.embedding for item in known[person_id]),
                *(item.face.embedding for item in picked),
            ])
    return trial


def _final_prototype_margins(proposed, prototypes):
    identities = tuple(prototypes)
    for person_id, picked in proposed.items():
        for candidate in picked:
            own = float(np.dot(_unit(candidate.face.embedding), _unit(prototypes[person_id])))
            other = max(
                float(np.dot(_unit(candidate.face.embedding), _unit(prototypes[identity])))
                for identity in identities if identity != person_id
            )
            if own - other < MIN_CROSS_PERSON_MARGIN:
                return False
    return True


def _deduplicate_candidates(candidates, known):
    selected = []
    for candidate in _rank_candidates(candidates):
        source = candidate.source.source
        duplicate = False
        for previous in (*known, *selected):
            prior = previous.source.source
            if source.asset_id == prior.asset_id or source.checksum.casefold() == prior.checksum.casefold():
                duplicate = True
                break
            similarity = float(np.dot(_unit(candidate.face.embedding), _unit(previous.face.embedding)))
            if similarity >= 0.98:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    return selected


def _rank_candidates(candidates):
    if not candidates:
        return []
    vectors = np.stack([_unit(item.face.embedding) for item in candidates])
    similarity = np.clip(vectors @ vectors.T, -1.0, 1.0)
    center = _unit(build_class_mean([item.face.embedding for item in candidates]))
    np.fill_diagonal(similarity, -np.inf)
    neighbors = min(5, len(candidates) - 1)
    density = np.ones(len(candidates)) if neighbors == 0 else np.mean(
        np.partition(similarity, -neighbors, axis=1)[:, -neighbors:], axis=1
    )
    scores = [1 - float(np.dot(vector, center)) + 1 - float(density[i]) - 0.05 * candidates[i].face.metrics.ediffiqa
              for i, vector in enumerate(vectors)]
    return [item for _, item in sorted(zip(scores, candidates), key=lambda pair: (pair[0], pair[1].source.source.taken_at, pair[1].source.source.asset_id))]


def _expansion_limit(existing):
    if isinstance(existing, bool) or not isinstance(existing, int) or not 5 <= existing <= _MAX_LIBRARY:
        raise ValueError("registered face count is outside the 5-30 limit")
    return min(_MAX_BATCH, _MAX_LIBRARY - existing)


def _inventory(frigate):
    return {name: list(files) for name, files in frigate.inventory().items()}


def _inventory_digest(inventory):
    normalized = {name: sorted(files) for name, files in inventory.items()}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
