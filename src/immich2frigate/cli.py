"""Command-line entrypoints for a deliberate one-shot rebuild."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .enrollment import (
    apply_rebuild_plan,
    backup_registered_library,
    build_rebuild_plan,
    reset_registered_library,
)
from .frigate_write import FrigateWriteClient
from .identity_registry import PersonIdentityRegistry
from .immich_client import ImmichReadOnlyClient
from .immich_vectors import ImmichVectorStore
from .selection import FOUNDATION_COUNT, MAX_TRAINING_COUNT
from .settings import FrigateSettings, ImmichDatabaseSettings, ImmichSettings


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

    args = parser.parse_args(argv)
    if args.command == "rebuild":
        return _rebuild(args)
    raise AssertionError("unreachable")


def _rebuild(args) -> int:
    if not args.confirm_reset:
        raise SystemExit("--confirm-reset is required for the destructive rebuild")

    immich_settings = ImmichSettings.from_env()
    database_settings = ImmichDatabaseSettings.from_env()
    frigate_settings = FrigateSettings.from_env()

    with ImmichReadOnlyClient(immich_settings) as immich:
        vectors = ImmichVectorStore(database_settings.database_url)
        frigate = FrigateWriteClient(frigate_settings)
        target = frigate.verify_target()

        # Build the full adaptive plan for every person before deleting anything.
        plan = build_rebuild_plan(immich, vectors)

        backup_path = Path(args.backup_dir)
        manifest = backup_registered_library(frigate, backup_path)
        deleted = reset_registered_library(frigate)

        registry = PersonIdentityRegistry(
            args.registry,
            immich_origin=immich_settings.immich_url,
            frigate_origin=target.origin,
        )
        result = apply_rebuild_plan(plan, immich, frigate, registry=registry)

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
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0
