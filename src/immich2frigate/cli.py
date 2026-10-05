"""Command-line entrypoints for a deliberate one-shot rebuild."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from .enrollment import (
    apply_rebuild_plan,
    backup_registered_library,
    build_incremental_plan,
    build_rebuild_plan,
    reset_registered_library,
)
from .frigate_client import FrigateApiError
from .frigate_write import FrigateWriteClient
from .identity_registry import PersonIdentityRegistry
from .immich_client import ImmichReadOnlyClient
from .immich_vectors import ImmichVectorStore
from .selection import FOUNDATION_COUNT, MAX_TRAINING_COUNT
from .settings import FrigateSettings, ImmichDatabaseSettings, ImmichSettings
from .sync_state import SyncState


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="immich2frigate")
    sub = parser.add_subparsers(dest="command", required=True)

    rebuild = sub.add_parser("rebuild", help="Rebuild Frigate faces from Immich representatives")
    rebuild.add_argument("--backup-dir", required=True)
    rebuild.add_argument("--registry", required=True)
    rebuild.add_argument(
        "--confirm-reset",
        action="store_true",
        help="Required before deleting the existing registered Frigate face library",
    )
    rebuild.add_argument("--top-people", type=int, default=3)
    rebuild.add_argument("--years", type=int, default=100)
    rebuild.add_argument("--max-count", type=int, default=5)
    rebuild.add_argument("--state", required=True)

    sync = sub.add_parser("sync", help="Add newly selected faces for the locked three-person roster")
    sync.add_argument("--registry", required=True)
    sync.add_argument("--state", required=True)
    sync.add_argument("--years", type=int, default=100)
    sync.add_argument("--loop-seconds", type=int, default=0,
                      help="Repeat forever at this interval; zero runs once")

    args = parser.parse_args(argv)
    if args.command == "rebuild":
        with _registry_lock(Path(args.registry)):
            return _rebuild(args)
    if args.command == "sync":
        return _sync(args)
    raise AssertionError("unreachable")


def _rebuild(args) -> int:
    state_path = Path(args.state)
    journal_path = state_path.with_name(state_path.name + ".rebuild-in-progress.json")
    if journal_path.exists():
        raise RuntimeError(
            "an earlier rebuild did not finish; inspect its backup and Frigate inventory "
            "before clearing any faces again"
        )
    if not args.confirm_reset:
        raise SystemExit("--confirm-reset is required for the destructive rebuild")

    immich_settings = ImmichSettings.from_env()
    database_settings = ImmichDatabaseSettings.from_env()
    frigate_settings = FrigateSettings.from_env()

    with ImmichReadOnlyClient(immich_settings) as immich:
        vectors = ImmichVectorStore(database_settings.database_url)
        frigate = FrigateWriteClient(frigate_settings)
        target = frigate.verify_target()

        if args.top_people != 3:
            raise ValueError("first-phase rebuild requires exactly three people")
        if not 1 <= args.years <= 100 or not 5 <= args.max_count <= MAX_TRAINING_COUNT:
            raise ValueError("years must be 1-100 and max-count must be 5-30")
        people = immich.people()
        ranked = sorted(
            ((immich.photo_count(person.person_id, years=args.years), person) for person in people),
            key=lambda row: (-row[0], row[1].name.casefold(), row[1].person_id),
        )
        if len(ranked) < 3 or any(count == 0 for count, _ in ranked[:3]):
            raise ValueError("Immich does not have three people with photos")
        selected_people = [person for _, person in ranked[:3]]
        plan = build_rebuild_plan(
            immich, vectors, people=selected_people, years=args.years,
            max_count=args.max_count,
        )
        if frigate.verify_target() != target:
            raise RuntimeError("Frigate target changed while the rebuild plan was being prepared")
        registry = PersonIdentityRegistry(
            args.registry,
            immich_origin=immich_settings.immich_url,
            frigate_origin=target.origin,
        )
        registry.preflight_bindings(
            (person.person.person_id, person.person.name) for person in plan.people
        )
        state = SyncState(
            args.state,
            immich_origin=immich_settings.immich_url,
            frigate_origin=target.origin,
            years=args.years,
        )
        state.preflight()

        backup_path = Path(args.backup_dir)
        manifest = backup_registered_library(frigate, backup_path)
        _write_rebuild_journal(
            journal_path,
            {
                "version": 1,
                "backup_dir": str(backup_path),
                "people": [
                    {
                        "person_id": item.person.person_id,
                        "name": item.person.name,
                        "face_ids": [candidate.source.face_id for candidate in item.candidates],
                    }
                    for item in plan.people
                ],
            },
        )
        deleted = reset_registered_library(frigate, expected_inventory=manifest["inventory"])

        result = apply_rebuild_plan(plan, immich, frigate, registry=registry)

        state.roster = [
            {
                "person_id": item.person.person_id,
                "name": item.person.name,
                "face_ids": [candidate.source.face_id for candidate in item.candidates],
                "examined_face_ids": list(item.examined_face_ids) or [
                    candidate.source.face_id for candidate in item.candidates
                ],
            }
            for item in plan.people
        ]
        if any(any(face_id is None for face_id in item["face_ids"]) for item in state.roster):
            raise RuntimeError("selected source face has no stable Immich face ID")
        state.save()
        journal_path.unlink()

    summary = {
        "version": target.version,
        "model_size": target.model_size,
        "people": result.people,
        "foundation_count": FOUNDATION_COUNT,
        "maximum_allowed_per_person": MAX_TRAINING_COUNT,
        "minimum_registered_per_person": result.minimum_per_person,
        "maximum_registered_per_person": result.maximum_per_person,
        "registered_images": result.registered_images,
        "deleted_registered_images": deleted,
        "backup_face_names": len(manifest["faces"]),
        "people_names": [item.person.name for item in plan.people],
        "source_photo_counts": {person.name: count for count, person in ranked[:3]},
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _write_rebuild_journal(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _sync(args) -> int:
    if not 1 <= args.years <= 100 or args.loop_seconds < 0:
        raise ValueError("years must be 1-100 and loop-seconds cannot be negative")
    while True:
        try:
            with _registry_lock(Path(args.registry)):
                _sync_once(args)
        except Exception as error:
            print(json.dumps({"sync_error": type(error).__name__, "message": str(error)}, ensure_ascii=False))
            if args.loop_seconds == 0:
                raise
        if args.loop_seconds == 0:
            return 0
        time.sleep(args.loop_seconds)


def _sync_once(args) -> None:
    immich_settings = ImmichSettings.from_env()
    database_settings = ImmichDatabaseSettings.from_env()
    frigate_settings = FrigateSettings.from_env()
    with ImmichReadOnlyClient(immich_settings) as immich:
        vectors = ImmichVectorStore(database_settings.database_url)
        frigate = FrigateWriteClient(frigate_settings)
        target = frigate.verify_target()
        state = SyncState(
            args.state,
            immich_origin=immich_settings.immich_url,
            frigate_origin=target.origin,
            years=args.years,
        )
        if len(state.roster) != 3:
            raise ValueError("sync state must contain exactly three enrolled people; run rebuild first")
        registry = PersonIdentityRegistry(
            args.registry,
            immich_origin=immich_settings.immich_url,
            frigate_origin=target.origin,
        )
        people_by_id = {person.person_id: person for person in immich.people()}
        by_id = {person["person_id"]: person for person in state.roster}
        _recover_pending_sync(state)
        added = 0
        per_person = []
        for person_id, source in by_id.items():
            person = people_by_id.get(person_id)
            if person is None or person.name != source["name"]:
                raise ValueError("a selected Immich person was removed or renamed; manual review required")
            binding = registry.binding(person_id)
            if binding is None:
                raise ValueError("selected person has no verified identity binding")
            name = binding.frigate_name
            files = frigate.inventory().get(name, ())
            if len(files) != len(source["face_ids"]):
                raise ValueError("Frigate face count differs from sync state; manual review required")
            if len(files) >= MAX_TRAINING_COUNT:
                per_person.append({"name": person.name, "registered": len(files), "added": 0})
                continue
            plan, examined = build_incremental_plan(
                immich, vectors, person, source["face_ids"],
                source.get("examined_face_ids", source["face_ids"]),
                years=args.years, max_count=MAX_TRAINING_COUNT,
            )
            source["examined_face_ids"] = list(examined)
            state.save()
            additions = list(zip(plan.candidates, plan.uploads))
            for candidate, image in additions:
                face_id = candidate.source.face_id
                before = len(frigate.inventory().get(name, ()))
                state.pending = {"person_id": person_id, "face_id": face_id, "before_count": before}
                state.save()
                if not before:
                    frigate.create_face(name)
                try:
                    response = frigate.register_face(name, image)
                except FrigateApiError as error:
                    if error.status_code == 400 and error.api_message == "No face was detected.":
                        source["examined_face_ids"] = sorted(
                            set(source.get("examined_face_ids", [])) | {face_id}
                        )
                        state.pending = None
                        state.save()
                        continue
                    raise
                if response.get("success") is not True:
                    source["examined_face_ids"] = sorted(
                        set(source.get("examined_face_ids", [])) | {face_id}
                    )
                    state.pending = None
                    state.save()
                    raise RuntimeError("Frigate rejected a candidate; it was marked as examined")
                after = len(frigate.inventory().get(name, ()))
                if after != before + 1:
                    raise RuntimeError("Frigate face count did not advance exactly once; pending upload retained")
                source["face_ids"].append(face_id)
                source["examined_face_ids"] = sorted(
                    set(source.get("examined_face_ids", [])) | {face_id}
                )
                state.pending = None
                state.save()
                added += 1
            per_person.append({"name": person.name, "registered": len(frigate.inventory().get(name, ())), "added": len(additions)})
        print(json.dumps({"sync_added": added, "people": per_person}, ensure_ascii=False, sort_keys=True))


def _recover_pending_sync(state: SyncState) -> None:
    pending = state.pending
    if pending is None:
        return
    # A process can stop after Frigate accepts a request but before the state
    # file records its response. Inventory counts cannot identify which image
    # was accepted (and an in-flight request may finish after this check), so
    # neither before_count nor before_count + 1 is safe evidence for retrying
    # or recording the Immich face ID. Keep the journal intact for review.
    raise ValueError(
        "pending Frigate upload outcome cannot be verified from image counts; "
        "manual review required"
    )


@contextmanager
def _registry_lock(registry_path: Path):
    lock_path = registry_path.with_name(registry_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise SystemExit("another synchronization is active or its lock needs manual review") from None
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
