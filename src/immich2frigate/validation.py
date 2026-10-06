"""Private, event-isolated checks of Frigate labels against Immich."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.error import HTTPError

from .frigate_client import (
    _NoRedirect,
    _matches_image_signature,
    FrigateReadOnlyClient,
    parse_face_recognition_profile,
)
from .frigate_names import frigate_face_name
from .settings import FrigateSettings
from .sync_state import write_json
from urllib.request import Request, build_opener

_MAX_RESPONSE = 2 * 1024 * 1024
_IMAGE_EXTENSIONS = (".webp", ".png", ".jpg", ".jpeg")
_EVENT_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class ValidationWatcher:
    def __init__(self, frigate_url: str, teacher, pipeline, roster, data_dir: str | Path,
                 *, frigate, opener=None):
        parts = urlsplit(frigate_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
                or parts.password or parts.path not in {"", "/"} or parts.query or parts.fragment):
            raise ValueError("FRIGATE_URL must be an origin URL")
        if not isinstance(roster, list) or not roster:
            raise ValueError("validation roster is required")
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.teacher = teacher
        self.pipeline = pipeline
        self.frigate = frigate
        self.roster = {entry["person_id"]: entry for entry in roster
                       if isinstance(entry, dict) and isinstance(entry.get("person_id"), str)}
        if not self.roster or len(self.roster) != len(roster):
            raise ValueError("validation roster is invalid")
        self.data_dir = Path(data_dir)
        self.archive = self.data_dir / "train-archive"
        profiles = {str(entry.get("person_id")): entry.get("profile") for entry in roster}
        pipeline_profile = getattr(pipeline, "profile", None)
        if pipeline_profile is not None and hasattr(pipeline_profile, "metadata"):
            pipeline_profile = pipeline_profile.metadata()
        self._base_profile = {"faces": profiles, "pipeline": pipeline_profile}
        self.profile: dict[str, object] = {}
        self.epoch = ""
        self.report_path = self.data_dir / "reports" / "uninitialized.json"
        self.current_report_path = self.data_dir / "report.json"
        self._opener = opener or build_opener(_NoRedirect())

    @classmethod
    def from_env(cls, teacher, pipeline, roster, data_dir=None, *, environ=None):
        env = os.environ if environ is None else environ
        settings = FrigateSettings.from_env(env)
        root = data_dir or env.get("IMMICH2FRIGATE_DATA_DIR")
        if not root:
            raise ValueError("IMMICH2FRIGATE_DATA_DIR is required for private validation storage")
        return cls(settings.frigate_url, teacher, pipeline, roster, root,
                   frigate=FrigateReadOnlyClient(settings))

    def run_once(self) -> dict[str, object]:
        try:
            return self._run_once()
        except Exception as error:
            self._write_failure(error)
            raise

    def _run_once(self) -> dict[str, object]:
        target = self.frigate.verify_target()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._mkdir_private(self.data_dir)
        write_json(self.current_report_path, {"version": 1, "fresh": False,
                                              "error": "validation_in_progress", "events": {}})
        self._mkdir_private(self.archive)
        teacher_profile = self.teacher.profile()
        frigate_config = self._get_json("/api/config")
        frigate_profile = parse_face_recognition_profile(frigate_config)
        self.profile = {**self._base_profile, "teacher": teacher_profile,
                        "frigate_face_recognition": frigate_profile,
                        "frigate_target": asdict(target)}
        enrolled = sorted(str(entry.get("enrolled_at")) for entry in self.roster.values())
        epoch_seed = json.dumps(self.profile, sort_keys=True, default=str, separators=(",", ":"))
        epoch_seed += "|" + "|".join(enrolled)
        inventory = self._get_json("/api/faces")
        filenames = inventory.get("train") if isinstance(inventory, dict) else None
        if not isinstance(filenames, list):
            raise RuntimeError("Frigate train inventory was unavailable")
        library = {name: sorted(files) for name, files in inventory.items() if name != "train"}
        inventory_digest = hashlib.sha256(json.dumps(
            library, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        self.epoch = hashlib.sha256(f"{epoch_seed}|{inventory_digest}".encode()).hexdigest()
        self.report_path = self.data_dir / "reports" / f"{self.epoch}.json"
        self._mkdir_private(self.report_path.parent)
        report = self._load_report()
        by_event: dict[str, list[tuple[str, Path]]] = {}
        for filename in filenames:
            if not self._valid_filename(filename):
                raise RuntimeError("Frigate returned an unsafe train filename")
            image = self._get_bytes(f"/clips/faces/train/{quote(filename, safe='')}")
            if not _matches_image_signature(filename, image):
                raise RuntimeError("Frigate train image did not match its file format")
            digest = hashlib.sha256(image).hexdigest()
            archived = self.archive / f"{digest}{Path(filename).suffix.lower()}"
            if not archived.exists():
                self._write_private(archived, image)
            event_id = _event_id(filename)
            if event_id is None:
                continue
            by_event.setdefault(event_id, []).append((filename, archived))
        for event_id, prior in report["events"].items():
            if isinstance(prior, dict):
                by_event.setdefault(event_id, [])
        for event_id, images in sorted(by_event.items()):
            prior = report["events"].get(event_id)
            for archived_name in prior.get("archive_paths", []) if isinstance(prior, dict) else []:
                if isinstance(archived_name, str) and Path(archived_name).name == archived_name:
                    for path in self.archive.glob(f"{Path(archived_name).stem}.*"):
                        images.append((archived_name, path))
            images = list({str(path): (name, path) for name, path in images}.values())
            images = sorted(images, key=lambda item: (item[0], str(item[1])))
            image_hashes = sorted({path.stem for _, path in images})
            event = self._get_json(f"/api/events/{quote(event_id, safe='')}", allow_not_found=True)
            if event is None:
                if not isinstance(prior, dict):
                    report["events"][event_id] = self._pending(
                        event_id, images, {"start_time": None}, reason="event_not_found")
                continue
            if not isinstance(event, dict):
                raise RuntimeError("Frigate event response was invalid")
            if not event.get("end_time"):
                report["events"][event_id] = self._pending(event_id, images, event)
                continue
            if (isinstance(prior, dict) and prior.get("pending") is False
                    and prior.get("file_sha256s") == image_hashes):
                report["events"][event_id] = _finalize_label(prior, event, self.roster)
                continue
            if any(entry.get("enrolled_at") is None or not isinstance(entry.get("profile"), dict)
                   for entry in self.roster.values()):
                report["events"][event_id] = self._unconfirmed(event_id, images, event)
                continue
            event_time = event.get("start_time") or event.get("end_time")
            if not any(_after(event_time, entry.get("enrolled_at")) for entry in self.roster.values()):
                report["events"][event_id] = self._pre_enrollment(event_id, images, event)
                continue
            report["events"][event_id] = self._evaluate(event_id, images, event)
        if self.frigate.verify_target() != target:
            raise RuntimeError("Frigate target changed during validation")
        report.update({"epoch": self.epoch, "profile": self.profile,
                       "inventory_digest": inventory_digest,
                       "fresh": True,
                       "train_inventory_digest": hashlib.sha256(json.dumps(
                           sorted(inventory.get("train", [])), separators=(",", ":")
                       ).encode()).hexdigest(),
                       "updated_at": datetime.now(timezone.utc).isoformat()})
        report["archive_count"] = len(list(self.archive.iterdir()))
        write_json(self.report_path, report)
        write_json(self.current_report_path, report)
        return report

    def watch(self, interval: int = 60) -> None:
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("interval must be a positive integer")
        while True:
            try:
                self.run_once()
            except Exception:
                pass
            time.sleep(interval)

    def _evaluate(self, event_id, images, event):
        reviewed = []
        for filename, archived in images:
            try:
                image = archived.read_bytes()
                prepared = self.pipeline.evaluate_crop(image)
                if prepared.eligible:
                    metrics = getattr(prepared, "metrics", None)
                    quality = (getattr(metrics, "ediffiqa", 0), getattr(metrics, "sharpness", 0),
                               getattr(metrics, "width", 0) * getattr(metrics, "height", 0))
                    reviewed.append((quality, filename, archived, image))
            except Exception:
                continue
        reviewed.sort(key=lambda item: item[0], reverse=True)
        reviewed = reviewed[:3]
        confirmations = []
        for quality, filename, archived, image in reviewed:
            try:
                result = self.teacher.confirm(image)
            except Exception:
                continue
            if getattr(result, "confirmed", False):
                confirmations.append((quality, filename, archived, result))
        file_hashes = sorted(path.stem for _, path in images)
        archive_paths = sorted({path.name for _, path in images})
        reason = None
        if len({getattr(result, "person_id", None) for _, _, _, result in confirmations}) > 1:
            teacher_result = None
            reason = "teacher_disagreement"
        elif confirmations:
            _, filename, archived, teacher_result = max(confirmations, key=lambda row: row[0])
        else:
            teacher_result = None
            reason = "no_clear_teacher_confirmed_crop" if reviewed else "no_clear_crop"
        teacher_id = getattr(teacher_result, "person_id", None) if teacher_result else None
        teacher_name = getattr(teacher_result, "name", None) if teacher_result else None
        return _finalize_label({
            "filename": filename if confirmations else None,
            "archive_sha256": archived.stem if confirmations else None,
            "archive_paths": archive_paths,
            "file_sha256s": file_hashes,
            "pending": False,
            "partition": event_partition(event_id),
            "teacher_confirmed": bool(teacher_result and getattr(teacher_result, "confirmed", False)),
            "teacher_person_id": teacher_id,
            "teacher_name": teacher_name,
            "reason": reason or (getattr(teacher_result, "reason", None) if teacher_result else None),
        }, event, self.roster)

    @staticmethod
    def _pending(event_id, images, event, reason="event_not_finalized"):
        return {"filename": None, "archive_sha256": None,
                "archive_paths": sorted({path.name for _, path in images}),
                "file_sha256s": sorted({path.stem for _, path in images}),
                "partition": event_partition(event_id), "category": "teacher_unconfirmed",
                "teacher_person_id": None, "teacher_name": None, "frigate_label": None,
                "reason": reason, "event_time": event.get("start_time"),
                "pending": True}

    @staticmethod
    def _unconfirmed(event_id, images, event):
        row = ValidationWatcher._pending(event_id, images, event, reason="classifier_profile_missing")
        row.update({"pending": False, "frigate_label": event.get("sub_label")
                    if isinstance(event.get("sub_label"), str) else None})
        return row

    @staticmethod
    def _pre_enrollment(event_id, images, event):
        row = ValidationWatcher._pending(event_id, images, event, reason="event_before_enrollment")
        label = event.get("sub_label")
        if isinstance(label, list):
            label = label[0] if label and isinstance(label[0], str) else None
        row.update({"pending": False, "teacher_confirmed": False,
                    "frigate_label": label if isinstance(label, str) else None,
                    "event_time": event.get("start_time") or event.get("end_time")})
        return row

    def _load_report(self):
        if not self.report_path.exists():
            return {"version": 1, "events": {}}
        try:
            value = json.loads(self.report_path.read_text(encoding="utf-8"))
            if value.get("version") != 1 or not isinstance(value.get("events"), dict):
                raise ValueError
            return value
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, AttributeError):
            raise ValueError("private validation report is invalid") from None

    def _get_bytes(self, path, *, allow_not_found=False):
        request = Request(self.origin + path, headers={"Accept": "application/json, image/*"}, method="GET")
        try:
            with self._opener.open(request, timeout=10) as response:
                final = urlsplit(response.geturl())
                if (final.scheme, final.netloc) != (urlsplit(self.origin).scheme, urlsplit(self.origin).netloc):
                    raise RuntimeError("Frigate redirected outside its configured origin")
                body = response.read(_MAX_RESPONSE + 1)
        except HTTPError as error:
            if allow_not_found and error.code == 404:
                return None
            raise RuntimeError(f"Frigate request failed (HTTP {error.code})") from None
        except Exception as error:
            raise RuntimeError(f"Frigate request failed ({type(error).__name__})") from None
        if len(body) > _MAX_RESPONSE:
            raise RuntimeError("Frigate response exceeded the size limit")
        return body

    def _get_json(self, path, *, allow_not_found=False):
        try:
            body = self._get_bytes(path, allow_not_found=allow_not_found)
            if body is None:
                return None
            value = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RuntimeError("Frigate returned invalid JSON") from None
        return value

    def _write_failure(self, error):
        report = {"version": 1, "epoch": self.epoch or None, "profile": self.profile,
                  "inventory_digest": None, "fresh": False,
                  "error": type(error).__name__, "updated_at": datetime.now(timezone.utc).isoformat(),
                  "events": {}}
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._mkdir_private(self.data_dir)
            write_json(self.current_report_path, report)
        except Exception:
            pass

    @staticmethod
    def _valid_filename(value):
        return (isinstance(value, str) and value not in {"", ".", ".."}
                and not any(c in value for c in "/\\")
                and not any(ord(c) < 32 or ord(c) == 127 for c in value)
                and value.lower().endswith(_IMAGE_EXTENSIONS))

    @staticmethod
    def _mkdir_private(path):
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o700)
        except OSError:
            pass

    @staticmethod
    def _write_private(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("xb") as output:
                try:
                    temporary.chmod(0o600)
                except OSError:
                    pass
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def event_partition(event_id: str) -> str:
    """Deterministically isolate calibration and never-trained holdout events."""
    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    return "calibration" if digest[0] < 128 else "holdout"


def _event_id(filename: str) -> str | None:
    """Frigate train names start with event-id (timestamp-random), then event time."""
    parts = Path(filename).stem.split("-")
    if len(parts) < 5 or not _EVENT_ID.fullmatch(f"{parts[0]}-{parts[1]}"):
        return None
    try:
        float(parts[0])
        float(parts[2])
        float(parts[-1])
    except ValueError:
        return None
    return f"{parts[0]}-{parts[1]}"


def calibration_gate(report: dict, person_id: str) -> dict[str, object]:
    return _gate(report, person_id, "calibration")


def final_gate(report: dict, person_id: str) -> dict[str, object]:
    return _gate(report, person_id, "holdout")


def validation_event_ids(report: dict) -> frozenset[str]:
    """Return every reviewed ID so enrollment/sync can exclude it from training."""
    events = report.get("events", {}) if isinstance(report, dict) else {}
    return frozenset(events) if isinstance(events, dict) else frozenset()


def _gate(report, person_id, partition):
    if (not isinstance(report, dict) or not isinstance(report.get("epoch"), str)
            or not report["epoch"] or not isinstance(report.get("profile"), dict)
            or not isinstance(report.get("inventory_digest"), str)
            or not report["inventory_digest"] or not _fresh(report)):
        return {"person_id": person_id, "partition": partition, "confirmed_events": 0,
                "conflicts": 0, "coverage": 0.0, "passed": False}
    events = report.get("events", {}) if isinstance(report, dict) else {}
    rows = [event for event in events.values() if isinstance(event, dict)
            and event.get("teacher_person_id") == person_id and event.get("partition") == partition]
    confirmed = [row for row in rows if row.get("category") in {"agree", "conflict", "frigate_unknown"}]
    conflicts = sum(row.get("category") == "conflict" for row in confirmed)
    coverage = sum(row.get("category") == "agree" for row in confirmed) / len(confirmed) if confirmed else 0.0
    passed = len(confirmed) >= 20 and conflicts == 0 and coverage >= 0.90
    return {"person_id": person_id, "partition": partition, "confirmed_events": len(confirmed),
            "conflicts": conflicts, "coverage": coverage, "passed": passed}


def _after(value, cutoff):
    try:
        start = (datetime.fromtimestamp(value, tz=timezone.utc)
                 if isinstance(value, (int, float)) and not isinstance(value, bool)
                 else datetime.fromisoformat(value.replace("Z", "+00:00")))
        enrolled = (datetime.fromtimestamp(cutoff, tz=timezone.utc)
                    if isinstance(cutoff, (int, float)) and not isinstance(cutoff, bool)
                    else datetime.fromisoformat(cutoff.replace("Z", "+00:00")))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if enrolled.tzinfo is None:
            enrolled = enrolled.replace(tzinfo=timezone.utc)
        return start >= enrolled
    except (AttributeError, TypeError, ValueError):
        return False


def _fresh(report):
    if report.get("fresh") is not True:
        return False
    try:
        updated = datetime.fromisoformat(report["updated_at"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        return 0 <= age <= 120
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _finalize_label(row, event, roster):
    result = dict(row)
    event_time = event.get("start_time") or event.get("end_time")
    sub_label = event.get("sub_label")
    if isinstance(sub_label, list):
        sub_label = sub_label[0] if sub_label and isinstance(sub_label[0], str) else None
    if not isinstance(sub_label, str):
        sub_label = None
    person_id = result.get("teacher_person_id")
    expected = roster.get(person_id) if result.get("teacher_confirmed") is True else None
    if not expected:
        category = "teacher_unconfirmed"
    elif not expected.get("enrolled_at") or not _after(event_time, expected["enrolled_at"]):
        category = "teacher_unconfirmed"
        result["reason"] = "event_before_classifier_start"
    elif not sub_label or sub_label.strip().casefold() in {"unknown", "unrecognized"}:
        category = "frigate_unknown"
    elif frigate_face_name(sub_label) in {
            frigate_face_name(expected.get("name", "")),
            frigate_face_name(result.get("teacher_name") or ""),
    }:
        category = "agree"
    else:
        category = "conflict"
    result.update({"category": category, "frigate_label": sub_label, "event_time": event_time,
                   "pending": False})
    return result
