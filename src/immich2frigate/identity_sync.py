"""Plan and safely apply label-only renames for already-bound people."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from .frigate_client import FrigateTarget
from .frigate018 import require_target
from .frigate_names import frigate_face_name
from .identity_registry import (
    PersonIdentityRegistry,
    normalize_origin,
    validate_frigate_name,
)


@dataclass(frozen=True, slots=True)
class NameSyncPlan:
    """Deterministic preview of person-name reconciliation."""

    plan_id: str
    target: str = field(repr=False)
    immich_origin: str = field(repr=False)
    version: str
    model_size: str
    inventory_sha256: str
    actions: tuple[Mapping[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "state": "NAME_SYNC_DRY_RUN",
            "target": self.target,
            "immich_origin": self.immich_origin,
            "version": self.version,
            "model_size": self.model_size,
            "inventory_sha256": self.inventory_sha256,
            "writes_enabled": False,
            "actions": [dict(action) for action in self.actions],
        }


@dataclass(frozen=True, slots=True)
class NameSyncResult:
    plan_id: str
    renamed_person_ids: tuple[str, ...]
    final_inventory_sha256: str
    recovered_person_ids: tuple[str, ...] = ()


def plan_person_name_sync(
    target: FrigateTarget,
    immich_origin: str,
    people: Sequence[object],
    frigate_faces: Mapping[str, tuple[str, ...]],
    registry: PersonIdentityRegistry,
) -> NameSyncPlan:
    """Build a read-only rename plan; no registry, journal, or API writes occur."""

    _require_registry_target(registry, immich_origin, target.origin)
    require_target(target.version, target.model_size)
    normalized_immich_origin = normalize_origin(immich_origin, allow_api=True)
    inventory = _validated_inventory(frigate_faces)
    actions = registry.reconcile(list(people), inventory)
    stable_actions = tuple(
        MappingProxyType(dict(action))
        for action in sorted(actions, key=lambda item: item["person_id"])
    )
    inventory_hash = _inventory_hash(inventory)
    identity = {
        "target": target.origin,
        "immich_origin": normalized_immich_origin,
        "version": target.version,
        "model_size": target.model_size,
        "inventory_sha256": inventory_hash,
        "actions": [dict(action) for action in stable_actions],
    }
    plan_id = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return NameSyncPlan(
        plan_id=plan_id,
        target=target.origin,
        immich_origin=normalized_immich_origin,
        version=target.version,
        model_size=target.model_size,
        inventory_sha256=inventory_hash,
        actions=stable_actions,
    )


def apply_person_name_sync(
    plan: NameSyncPlan,
    *,
    immich_origin: str,
    people: Sequence[object],
    frigate_client,
    registry: PersonIdentityRegistry,
    journal_path: str | Path,
    backup_verified: Callable[[], bool],
) -> NameSyncResult:
    """Serialize a name-sync run for this identity registry, then apply it."""

    with _registry_lock(registry.path):
        return _apply_person_name_sync_locked(
            plan,
            immich_origin=immich_origin,
            people=people,
            frigate_client=frigate_client,
            registry=registry,
            journal_path=journal_path,
            backup_verified=backup_verified,
        )


def _apply_person_name_sync_locked(
    plan: NameSyncPlan,
    *,
    immich_origin: str,
    people: Sequence[object],
    frigate_client,
    registry: PersonIdentityRegistry,
    journal_path: str | Path,
    backup_verified: Callable[[], bool],
) -> NameSyncResult:
    """Apply only planned, owned renames and verify Frigate before persisting.

    The supplied backup callback must create and verify the caller's Frigate
    database/face-library backup, returning ``True`` only after verification.
    Ambiguous write outcomes are journaled and are never retried here. If a
    multi-person run stops partway through, build a new plan from fresh state.
    """

    _require_registry_target(registry, immich_origin, plan.target)
    if normalize_origin(immich_origin, allow_api=True) != plan.immich_origin:
        raise ValueError("name-sync plan belongs to a different Immich instance")
    target = frigate_client.verify_target()
    if (target.origin, target.version, target.model_size) != (
        plan.target,
        plan.version,
        plan.model_size,
    ):
        raise ValueError("Frigate target changed after name-sync planning")

    inventory = _validated_inventory(frigate_client.inventory())
    if _inventory_hash(inventory) != plan.inventory_sha256:
        raise ValueError("Frigate face inventory changed after name-sync planning")
    current_plan = plan_person_name_sync(
        target, immich_origin, people, inventory, registry
    )
    if current_plan.plan_id != plan.plan_id:
        raise ValueError("identity registry or Immich people changed after planning")

    journal = Path(journal_path)
    recovered = _recover_journaled_operations(
        journal, plan.target, plan.immich_origin, inventory, registry
    )
    renames = [action for action in plan.actions if action["status"] == "RENAME_REQUIRED"]
    unsafe = [
        action
        for action in plan.actions
        if action["status"]
        in {
            "RENAME_CONFLICT",
            "BOUND_LABEL_MISSING",
            "BOUND_LABELS_MISSING",
        }
    ]
    unsafe.extend(
        action
        for action in plan.actions
        if action["status"] == "REMOTE_ALREADY_RENAMED"
        and action["person_id"] not in recovered
    )
    if unsafe:
        raise ValueError("name sync contains labels requiring manual review")
    _preflight_renames(renames, inventory)
    if not renames:
        return NameSyncResult(
            plan.plan_id, (), _inventory_hash(inventory), tuple(sorted(recovered))
        )

    if backup_verified() is not True:
        raise ValueError("verified Frigate backup is required before renaming")

    renamed_ids: list[str] = []
    for action in renames:
        operation_id = str(uuid4())
        old_name, new_name = action["old_name"], action["name"]
        before_files = inventory[old_name]
        expected_inventory = dict(inventory)
        expected_inventory.pop(old_name)
        expected_inventory[new_name] = before_files
        operation = {
            "operation_id": operation_id,
            "plan_id": plan.plan_id,
            "target": plan.target,
            "immich_origin": plan.immich_origin,
            "person_id": action["person_id"],
            "sync_person_id": action["sync_person_id"],
            "old_name": old_name,
            "new_name": new_name,
            "before_files": list(before_files),
            "pre_inventory_sha256": _inventory_hash(inventory),
            "expected_inventory_sha256": _inventory_hash(expected_inventory),
        }
        _append_journal(journal, {**operation, "state": "PENDING"})
        try:
            response = frigate_client.rename_face(old_name, new_name)
        except Exception as error:
            _append_journal(
                journal,
                {**operation, "state": "UNKNOWN_RESULT", "error": type(error).__name__},
            )
            raise
        if not isinstance(response, Mapping) or response.get("success") is not True:
            state = (
                "FAILED"
                if isinstance(response, Mapping) and response.get("success") is False
                else "UNKNOWN_RESULT"
            )
            _append_journal(journal, {**operation, "state": state})
            raise RuntimeError("Frigate did not confirm the face-label rename")

        try:
            after = _validated_inventory(frigate_client.inventory())
            expected = dict(inventory)
            expected.pop(old_name)
            expected[new_name] = before_files
            if after != dict(sorted(expected.items())):
                raise ValueError("Frigate rename changed unexpected labels or face files")
        except Exception as error:
            _append_journal(
                journal,
                {**operation, "state": "UNKNOWN_RESULT", "error": type(error).__name__},
            )
            raise

        try:
            registry.bind(action["person_id"], new_name)
        except Exception as error:
            _append_journal(
                journal,
                {**operation, "state": "REGISTRY_UPDATE_PENDING", "error": type(error).__name__},
            )
            raise
        _append_journal(
            journal,
            {**operation, "state": "VERIFIED", "post_inventory_sha256": _inventory_hash(after)},
        )
        inventory = after
        renamed_ids.append(action["person_id"])

    return NameSyncResult(
        plan.plan_id,
        tuple(renamed_ids),
        _inventory_hash(inventory),
        tuple(sorted(recovered)),
    )


def _preflight_renames(
    renames: Sequence[Mapping[str, str]], inventory: Mapping[str, tuple[str, ...]]
) -> None:
    old_names = [action["old_name"] for action in renames]
    new_names = [action["name"] for action in renames]
    if len({name.casefold() for name in old_names}) != len(old_names):
        raise ValueError("name sync contains duplicate source labels")
    if len({name.casefold() for name in new_names}) != len(new_names):
        raise ValueError("name sync contains duplicate destination labels")
    for action in renames:
        old_name, new_name = action["old_name"], action["name"]
        if old_name not in inventory:
            raise ValueError("bound Frigate source label is missing")
        if new_name in inventory:
            raise ValueError("Frigate destination label already exists")


def _validated_inventory(value: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError("Frigate inventory must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    normalized: set[str] = set()
    for name, filenames in value.items():
        if not isinstance(name, str) or not isinstance(filenames, (list, tuple)):
            raise ValueError("Frigate inventory has an invalid shape")
        validate_frigate_name(name)
        key = frigate_face_name(name).casefold()
        if key in normalized:
            raise ValueError("Frigate inventory has colliding labels")
        normalized.add(key)
        if any(
            not isinstance(filename, str)
            or filename in {"", ".", ".."}
            or any(char in filename for char in "/\\")
            or any(ord(char) < 32 or ord(char) == 127 for char in filename)
            or not filename.lower().endswith((".webp", ".png", ".jpg", ".jpeg"))
            for filename in filenames
        ):
            raise ValueError("Frigate inventory contains an invalid filename")
        result[name] = tuple(sorted(filenames))
    return dict(sorted(result.items()))


def _inventory_hash(value: Mapping[str, tuple[str, ...]]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_journal(path: Path, entry: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(encoded + "\n")
        output.flush()
        os.fsync(output.fileno())


@contextmanager
def _registry_lock(registry_path: Path):
    lock_path = registry_path.with_name(registry_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("another name-sync run is active or its lock needs manual review") from None
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _recover_journaled_operations(
    journal: Path,
    target: str,
    immich_origin: str,
    inventory: Mapping[str, tuple[str, ...]],
    registry: PersonIdentityRegistry,
) -> set[str]:
    if not journal.exists():
        return set()
    latest: dict[str, dict[str, object]] = {}
    try:
        with journal.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("operation_id"), str):
                    raise ValueError
                latest[record["operation_id"]] = record
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("operation journal is invalid; manual review is required") from None

    recovered: set[str] = set()
    unresolved = {"PENDING", "UNKNOWN_RESULT", "FAILED", "REGISTRY_UPDATE_PENDING"}
    for record in latest.values():
        if record.get("state") not in unresolved:
            continue
        if record.get("target") != target or record.get("immich_origin") != immich_origin:
            continue
        if record.get("state") == "FAILED":
            raise ValueError("failed journal operation needs manual review")

        person_id = record.get("person_id")
        sync_person_id = record.get("sync_person_id")
        old_name = record.get("old_name")
        new_name = record.get("new_name")
        expected_files = record.get("before_files")
        expected_hash = record.get("expected_inventory_sha256")
        if (
            not isinstance(person_id, str)
            or not isinstance(sync_person_id, str)
            or not isinstance(old_name, str)
            or not isinstance(new_name, str)
            or not isinstance(expected_files, list)
            or not isinstance(expected_hash, str)
            or _inventory_hash(inventory) != expected_hash
            or old_name in inventory
            or inventory.get(new_name) != tuple(sorted(expected_files))
        ):
            raise ValueError(
                "unresolved journal operation needs manual review; automatic retry is disabled"
            )
        binding = registry.binding(person_id)
        if binding is None or binding.sync_person_id != sync_person_id:
            raise ValueError("unresolved journal identity needs manual review")
        if binding.frigate_name == old_name:
            try:
                registry.bind(person_id, new_name)
            except Exception as error:
                _append_journal(
                    journal,
                    {**record, "state": "REGISTRY_UPDATE_PENDING", "error": type(error).__name__},
                )
                raise
        elif binding.frigate_name != new_name:
            raise ValueError(
                "unresolved journal binding does not match Frigate; manual review is required"
            )
        _append_journal(
            journal,
            {
                **record,
                "state": "RECOVERED",
                "post_inventory_sha256": _inventory_hash(inventory),
            },
        )
        recovered.add(person_id)
    return recovered


def _require_registry_target(
    registry: PersonIdentityRegistry, immich_origin: str, frigate_origin: str
) -> None:
    if registry.immich_origin != normalize_origin(immich_origin, allow_api=True):
        raise ValueError("identity registry belongs to a different Immich instance")
    if registry.frigate_origin != normalize_origin(frigate_origin):
        raise ValueError("identity registry belongs to a different Frigate instance")
