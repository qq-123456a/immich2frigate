import json
from types import SimpleNamespace
from datetime import datetime, timezone
from urllib.error import HTTPError

from immich2frigate.frigate_client import FrigateTarget
from immich2frigate.validation import (
    ValidationWatcher, _event_id, calibration_gate, event_partition, final_gate,
    validation_event_ids,
)


EVENT_ID = "1000.25-track7"
FILENAME = f"{EVENT_ID}-1001.5-unknown-0.91.jpg"
IMAGE = b"\xff\xd8\xfftest-image"
OTHER_FILENAME = f"{EVENT_ID}-1001.6-unknown-0.90.jpg"
DEFAULT_EVENT = object()


class Response:
    def __init__(self, url, body):
        self.url, self.body = url, body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def geturl(self):
        return self.url

    def read(self, size):
        return self.body[:size]


class Opener:
    def __init__(self, *, filenames=None, event=DEFAULT_EVENT):
        self.calls = []
        self.filenames = [FILENAME, OTHER_FILENAME] if filenames is None else filenames
        self.recognition_threshold = 0.9
        self.min_faces = 1
        self.event = {"id": EVENT_ID, "start_time": 1000, "end_time": 1002,
                      "sub_label": "Lili_Luo"} if event is DEFAULT_EVENT else event

    def open(self, request, timeout):
        self.calls.append(request.full_url)
        if request.full_url.endswith("/api/config"):
            body = json.dumps({"face_recognition": {"enabled": True, "model_size": "large",
                "recognition_threshold": self.recognition_threshold, "min_faces": self.min_faces},
                "environment_vars": {"SECRET": "never expose"}}).encode()
        elif request.full_url.endswith("/api/faces"):
            body = json.dumps({"train": self.filenames}).encode()
        elif request.full_url.endswith(f"/api/events/{EVENT_ID}"):
            if self.event is None:
                raise HTTPError(request.full_url, 404, "not found", {}, None)
            body = json.dumps(self.event).encode()
        else:
            body = IMAGE + request.full_url.rsplit("/", 1)[-1].encode()
        return Response(request.full_url, body)


class TargetClient:
    def __init__(self):
        self.target = FrigateTarget("http://frigate:5000", "0.18.0", "large")

    def verify_target(self):
        return self.target


def test_watcher_archives_train_and_uses_final_event_label(tmp_path):
    person = "person-1"
    teacher = SimpleNamespace(profile=lambda: {"modelName": "buffalo_l"}, confirm=lambda image: SimpleNamespace(
        confirmed=True, person_id=person, name="Lili Luo", reason=None))
    pipeline = SimpleNamespace(profile=SimpleNamespace(metadata=lambda: {"profile_id": "test"}),
                               evaluate_crop=lambda image: SimpleNamespace(
                                   eligible=True, metrics=SimpleNamespace(ediffiqa=.8, sharpness=300,
                                                                          width=100, height=100)))
    opener = Opener()
    frigate = TargetClient()
    watcher = ValidationWatcher("http://frigate:5000", teacher, pipeline,
                                [{"person_id": person, "name": "Lili Luo", "enrolled_at": 900,
                                  "profile": {"model": "new"}}], tmp_path,
                                frigate=frigate, opener=opener)

    report = watcher.run_once()

    assert report["events"][EVENT_ID]["category"] == "agree"
    assert report["events"][EVENT_ID]["partition"] == event_partition(EVENT_ID)
    assert report["epoch"] and report["profile"] and report["inventory_digest"]
    assert report["profile"]["teacher"] == {"modelName": "buffalo_l"}
    assert report["profile"]["frigate_face_recognition"] == {
        "enabled": True, "model_size": "large", "recognition_threshold": 0.9, "min_faces": 1}
    assert report["profile"]["frigate_target"] == {
        "origin": "http://frigate:5000", "version": "0.18.0", "model_size": "large"}
    assert report["fresh"] is True
    assert validation_event_ids(report) == {EVENT_ID}
    assert len(list(watcher.archive.iterdir())) == 2
    assert opener.calls[-1].endswith(f"/api/events/{EVENT_ID}")
    assert any("/clips/faces/train/" in url for url in opener.calls)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "reports" / f"{watcher.epoch}.json").exists()
    assert _event_id("Alice_1791278777.000.webp") is None


def test_frigate_recognition_config_is_bound_to_validation_epoch(tmp_path):
    person = "person-1"
    teacher = SimpleNamespace(profile=lambda: {"modelName": "buffalo_l"}, confirm=lambda image: SimpleNamespace(
        confirmed=True, person_id=person, name="Lili Luo", reason=None))
    pipeline = SimpleNamespace(profile=None, evaluate_crop=lambda image: SimpleNamespace(
        eligible=True, metrics=SimpleNamespace(ediffiqa=.8, sharpness=300, width=100, height=100)))
    roster = [{"person_id": person, "name": "Lili Luo", "enrolled_at": 900, "profile": {}}]
    opener = Opener()
    watcher = ValidationWatcher("http://frigate:5000", teacher, pipeline, roster, tmp_path,
                                frigate=TargetClient(), opener=opener)

    first = watcher.run_once()
    opener.recognition_threshold = 0.95
    opener.min_faces = 3
    second = watcher.run_once()

    assert first["epoch"] != second["epoch"]
    assert first["profile"]["frigate_face_recognition"]["recognition_threshold"] == 0.9
    assert second["profile"]["frigate_face_recognition"]["min_faces"] == 3


def test_watcher_rejects_target_change_during_validation_run(tmp_path):
    person = "person-1"
    teacher = SimpleNamespace(profile=lambda: {"modelName": "buffalo_l"}, confirm=lambda image: SimpleNamespace(
        confirmed=True, person_id=person, name="Lili Luo", reason=None))
    pipeline = SimpleNamespace(profile=None, evaluate_crop=lambda image: SimpleNamespace(
        eligible=True, metrics=SimpleNamespace(ediffiqa=.8, sharpness=300, width=100, height=100)))

    class ChangingTarget:
        def __init__(self):
            self.calls = 0

        def verify_target(self):
            self.calls += 1
            model_size = "large" if self.calls == 1 else "small"
            return FrigateTarget("http://frigate:5000", "0.18.0", model_size)

    watcher = ValidationWatcher("http://frigate:5000", teacher, pipeline,
        [{"person_id": person, "name": "Lili Luo", "enrolled_at": 900, "profile": {}}],
        tmp_path, frigate=ChangingTarget(), opener=Opener())

    try:
        watcher.run_once()
    except RuntimeError as error:
        assert "Frigate target changed" in str(error)
    else:
        raise AssertionError("a changing Frigate target must invalidate the validation run")
    current = json.loads((tmp_path / "report.json").read_text())
    assert current["fresh"] is False


def test_pre_enrollment_event_is_archived_without_face_or_teacher_inference(tmp_path):
    person = "person-1"
    teacher_calls = []
    pipeline_calls = []
    teacher = SimpleNamespace(
        profile=lambda: {"modelName": "buffalo_l"},
        confirm=lambda image: teacher_calls.append(image),
    )
    pipeline = SimpleNamespace(
        profile=None,
        evaluate_crop=lambda image: pipeline_calls.append(image),
    )
    watcher = ValidationWatcher("http://frigate:5000", teacher, pipeline,
        [{"person_id": person, "name": "Lili Luo", "enrolled_at": 2000, "profile": {}}],
        tmp_path, frigate=TargetClient(), opener=Opener())

    report = watcher.run_once()

    row = report["events"][EVENT_ID]
    assert row["category"] == "teacher_unconfirmed"
    assert row["reason"] == "event_before_enrollment"
    assert row["event_time"] == 1000
    assert row["frigate_label"] == "Lili_Luo"
    assert row["teacher_confirmed"] is False
    assert len(row["archive_paths"]) == 2
    assert len(list(watcher.archive.iterdir())) == 2
    assert teacher_calls == pipeline_calls == []


def test_calibration_and_holdout_gates_are_independent():
    person = "person-1"
    events = {}
    counts = {"calibration": 0, "holdout": 0}
    number = 0
    while min(counts.values()) < 20:
        event_id = f"event-{number}"
        partition = event_partition(event_id)
        if counts[partition] < 20:
            events[event_id] = {"teacher_person_id": person, "category": "agree", "partition": partition}
            counts[partition] += 1
        number += 1
    report = {"events": events, "epoch": "epoch", "profile": {}, "inventory_digest": "digest",
              "fresh": True, "updated_at": datetime.now(timezone.utc).isoformat()}
    calibration = calibration_gate(report, person)
    holdout = final_gate(report, person)
    assert calibration["confirmed_events"] == holdout["confirmed_events"] == 20
    assert calibration["passed"] and holdout["passed"]
    next(row for row in events.values() if row["partition"] == "calibration")["category"] = "conflict"
    assert not calibration_gate(report, person)["passed"]


def test_teacher_disagreement_is_unconfirmed_and_failed_run_clears_fresh_gate(tmp_path):
    person_a, person_b = "person-1", "person-2"
    teacher = SimpleNamespace(
        profile=lambda: {"modelName": "buffalo_l"},
        confirm=lambda image: SimpleNamespace(confirmed=True,
            person_id=person_a if image.endswith(FILENAME.encode()) else person_b,
            name="Alice" if image.endswith(FILENAME.encode()) else "Bob"),
    )
    pipeline = SimpleNamespace(profile=None, evaluate_crop=lambda image: SimpleNamespace(
        eligible=True, metrics=SimpleNamespace(ediffiqa=.8, sharpness=300, width=100, height=100)))
    roster = [{"person_id": person_a, "name": "Alice", "enrolled_at": 900, "profile": {}},
              {"person_id": person_b, "name": "Bob", "enrolled_at": 900, "profile": {}}]
    watcher = ValidationWatcher("http://frigate:5000", teacher, pipeline, roster, tmp_path,
                                frigate=TargetClient(), opener=Opener())
    report = watcher.run_once()
    assert report["events"][EVENT_ID]["category"] == "teacher_unconfirmed"
    assert report["events"][EVENT_ID]["reason"] == "teacher_disagreement"
    assert len(report["events"][EVENT_ID]["archive_paths"]) == 2

    class FailedOpener:
        def open(self, *args, **kwargs):
            raise OSError("offline")

    failed = ValidationWatcher("http://frigate:5000", teacher, pipeline, roster, tmp_path,
                              frigate=TargetClient(), opener=FailedOpener())
    try:
        failed.run_once()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the failed Frigate request")
    current = json.loads((tmp_path / "report.json").read_text())
    assert current["fresh"] is False
    assert not calibration_gate(current, person_a)["passed"]


def test_pending_event_is_rechecked_after_train_file_rolls_out_and_label_can_change(tmp_path):
    person = "person-1"
    teacher = SimpleNamespace(profile=lambda: {"modelName": "buffalo_l"}, confirm=lambda image: SimpleNamespace(
        confirmed=True, person_id=person, name="Lili Luo", reason=None))
    pipeline = SimpleNamespace(profile=None, evaluate_crop=lambda image: SimpleNamespace(
        eligible=True, metrics=SimpleNamespace(ediffiqa=.8, sharpness=300, width=100, height=100)))
    roster = [{"person_id": person, "name": "Lili Luo", "enrolled_at": 900, "profile": {}}]
    opener = Opener(filenames=[FILENAME], event=None)
    watcher = ValidationWatcher("http://frigate:5000", teacher, pipeline, roster, tmp_path,
                                frigate=TargetClient(), opener=opener)

    pending = watcher.run_once()
    assert pending["events"][EVENT_ID]["pending"] is True
    assert len(list(watcher.archive.iterdir())) == 1

    opener.filenames = []
    opener.event = {"id": EVENT_ID, "start_time": 1000, "end_time": 1002, "sub_label": "unknown"}
    unknown = watcher.run_once()
    assert unknown["events"][EVENT_ID]["category"] == "frigate_unknown"
    assert unknown["events"][EVENT_ID]["pending"] is False

    opener.event = {"id": EVENT_ID, "start_time": 1000, "end_time": 1002, "sub_label": "Lili_Luo"}
    corrected = watcher.run_once()
    assert corrected["events"][EVENT_ID]["category"] == "agree"
