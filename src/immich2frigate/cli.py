"""Prepare first, register frozen plans, and collect independent validation."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from .frigate_write import FrigateWriteClient
from .identity_registry import PersonIdentityRegistry
from .immich_client import ImmichReadOnlyClient
from .immich_vectors import ImmichVectorStore
from .settings import FrigateSettings, ImmichDatabaseSettings, ImmichSettings
from .sync_state import SyncState, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="immich2frigate")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "rebuild", "sync", "validate"):
        child = sub.add_parser(command)
        child.add_argument("--registry", required=True)
        child.add_argument("--state", required=True)
        child.add_argument("--years", type=int, default=100)
        if command in {"prepare", "rebuild"}:
            child.add_argument("--plan-dir", required=True)
        if command == "rebuild":
            child.add_argument("--backup-dir", required=True)
            child.add_argument("--confirm-reset", action="store_true")
        if command == "sync":
            mode = child.add_mutually_exclusive_group()
            mode.add_argument("--propose-only", action="store_true")
            mode.add_argument("--apply-plan")
            child.add_argument("--plan-dir")
            child.add_argument("--validation-report")
            child.add_argument("--loop-seconds", type=int, default=0)
        if command == "validate":
            child.add_argument("--data-dir", required=True)
            child.add_argument("--watch", action="store_true")
            child.add_argument("--interval", type=int, default=60)
    args = parser.parse_args(argv)
    if not 1 <= args.years <= 100:
        parser.error("years must be 1-100")
    if args.command == "sync":
        if args.loop_seconds < 0 or (args.apply_plan and args.loop_seconds):
            parser.error("apply-plan runs once; loop-seconds cannot be negative")
        while True:
            try:
                with _registry_lock(Path(args.registry)):
                    _run(args)
            except Exception as error:
                if args.loop_seconds == 0:
                    raise
                print(json.dumps({"sync_error": type(error).__name__}), flush=True)
            if not args.loop_seconds:
                return 0
            time.sleep(args.loop_seconds)
    if args.command == "validate":
        if args.interval < 1:
            parser.error("validation interval must be positive")
        return _run(args)
    with _registry_lock(Path(args.registry)):
        return _run(args)


def _run(args) -> int:
    from .face_pipeline import FacePipeline
    from .frozen_plan import apply_plan, load_plan, prepare_foundation, reconcile_expansion
    from .teacher import ImmichTeacher
    from .validation import ValidationWatcher, calibration_gate

    immich_settings = ImmichSettings.from_env()
    frigate_settings = FrigateSettings.from_env()
    frigate = FrigateWriteClient(frigate_settings)
    target = frigate.verify_target()
    state = SyncState(args.state, immich_origin=immich_settings.immich_url, frigate_origin=target.origin, years=args.years)
    if len(state.roster) != 3:
        raise ValueError("a locked three-person source roster is required")
    _recover_pending_sync(state)
    if state.path.with_name(state.path.name + ".rebuild-in-progress.json").exists():
        raise RuntimeError("unfinished enrollment requires recovery before further work")
    registry = PersonIdentityRegistry(args.registry, immich_origin=immich_settings.immich_url, frigate_origin=target.origin)
    pipeline, teacher = FacePipeline.from_env(), ImmichTeacher.from_env()
    if args.command == "validate":
        while True:
            cycle_started = time.monotonic()
            # Refresh the ledger each cycle: another process may have applied a batch.
            state = SyncState(args.state, immich_origin=immich_settings.immich_url, frigate_origin=target.origin, years=args.years)
            watcher = ValidationWatcher(frigate_settings.frigate_url, teacher, pipeline, state.roster,
                                        args.data_dir, frigate=frigate)
            try:
                report = watcher.run_once()
                if any((row.get("last_batch") or {}).get("status") == "pending" for row in state.roster):
                    with _registry_lock(Path(args.registry)):
                        if file_state_unchanged(state):
                            status = reconcile_expansion(state=state, registry=registry, frigate=frigate, pipeline=pipeline, teacher=teacher, report=report)
                            if status:
                                print(json.dumps(status), flush=True)
                print(json.dumps({"epoch": report["epoch"], "reviewed_events": len(report["events"]), "archive_count": report["archive_count"]}), flush=True)
            except Exception as error:
                watcher._write_failure(error)
                if not args.watch:
                    raise
                print(json.dumps({"validation_error": type(error).__name__}), flush=True)
            if not args.watch:
                return 0
            time.sleep(max(1, args.interval - (time.monotonic() - cycle_started)))
    vectors = ImmichVectorStore(ImmichDatabaseSettings.from_env().database_url)
    with ImmichReadOnlyClient(immich_settings) as immich:
        context = dict(state=state, registry=registry, frigate=frigate, immich=immich, vectors=vectors, pipeline=pipeline, teacher=teacher)
        if args.command == "prepare":
            plan = prepare_foundation(args.plan_dir, **context)
            print(json.dumps({"prepared_plan": plan["plan_id"], "people": 3, "images": 15}), flush=True)
        elif args.command == "rebuild":
            if not args.confirm_reset:
                raise SystemExit("--confirm-reset is required for the destructive rebuild")
            root, plan = load_plan(args.plan_dir)
            if plan["kind"] != "rebuild":
                raise ValueError("rebuild requires a foundation plan")
            print(json.dumps(apply_plan(root, plan, backup_dir=args.backup_dir, **context)), flush=True)
        elif args.command == "sync":
            report_path = Path(args.validation_report) if args.validation_report else state.path.parent / "validation" / "report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            if args.apply_plan:
                root, plan = load_plan(args.apply_plan)
                if plan["kind"] != "expansion":
                    raise ValueError("sync apply requires a verified expansion plan")
                print(json.dumps(apply_plan(root, plan, validation_report=report, **context)), flush=True)
            elif not all(calibration_gate(report, person["person_id"])["passed"] for person in state.roster):
                print(json.dumps({"status": "waiting_for_calibration", "registered_counts": [len(p["face_ids"]) for p in state.roster]}), flush=True)
            else:
                from .expansion import prepare_expansion
                directory = Path(args.plan_dir) if args.plan_dir else state.path.parent / "proposals"
                plan = prepare_expansion(directory / str(time.time_ns()), validation_report=report, **context)
                print(json.dumps({"proposed_plan": plan["plan_id"], "proposed_images": sum(len(p["images"]) for p in plan["people"])}), flush=True)
    return 0


def _write_rebuild_journal(path: Path, value: dict[str, object]) -> None:
    write_json(path, value)


def _recover_pending_sync(state: SyncState) -> None:
    if state.pending is not None:
        raise ValueError("pending Frigate upload outcome cannot be verified from image counts; manual review required")


def file_state_unchanged(state: SyncState) -> bool:
    current = SyncState(state.path, immich_origin=state.immich_origin, frigate_origin=state.frigate_origin, years=state.years)
    return current.roster == state.roster and current.pending == state.pending


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
