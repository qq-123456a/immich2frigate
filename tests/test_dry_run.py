from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from immich2frigate.dry_run import DryRunCandidate, build_dry_run_plan
from immich2frigate.frigate_client import FrigateTarget
from immich2frigate.immich_client import FaceCandidate, PersonRecord

PERSON_ID = "00000000-0000-4000-8000-000000000001"
ASSET_ID = "00000000-0000-4000-8000-000000000002"


class FakeYuNet:
    def __init__(self, rows):
        self.rows = rows

    def setInputSize(self, size):
        pass

    def detect(self, image):
        return 1, self.rows


def make_candidate(upload_bytes=b"\xff\xd8\xffsynthetic"):
    person = PersonRecord(PERSON_ID, "Synthetic Person")
    source = FaceCandidate(
        person_id=PERSON_ID,
        face_id=None,
        asset_id=ASSET_ID,
        taken_at="never-export-date-marker",
        checksum="never-export-this-marker",
        box=(0, 0, 10, 10),
        frame=(10, 10),
    )
    return person, DryRunCandidate(person, source, upload_bytes)


def detector_with_face():
    row = np.array([2, 3, 30, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9])
    return FakeYuNet(np.array([row]))


def test_plan_is_deterministic_contains_only_hashes_and_never_enables_writes():
    image = np.full((48, 64, 3), 140, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    person, candidate = make_candidate(encoded.tobytes())
    target = FrigateTarget("http://frigate:5000", "0.18.0", "large")

    first = build_dry_run_plan(target, [person], [candidate], {}, detector=detector_with_face())
    second = build_dry_run_plan(target, [person], [candidate], {}, detector=detector_with_face())
    manifest = first.to_json()

    assert first.plan_id == second.plan_id
    assert first.entries[0]["status"] == "PROPOSED_ADD_MODELS_UNVERIFIED"
    assert first.entries[0]["upload_sha256"]
    assert first.entries[0]["simulated_registered_image_sha256"]
    assert first.as_dict()["state"] == "DRY_RUN_ONLY"
    assert first.as_dict()["writes_enabled"] is False
    assert first.as_dict()["successful_sync_recorded"] is False
    assert first.as_dict()["compatibility_status"] == "MODEL_ASSETS_NOT_VERIFIED"
    assert "never-export-this-marker" not in manifest
    assert "never-export-date-marker" not in manifest
    assert encoded.tobytes().hex() not in manifest
    with pytest.raises(TypeError):
        first.entries[0]["status"] = "SYNCED"


def test_existing_face_label_is_never_claimed_or_modified():
    person, candidate = make_candidate()
    target = FrigateTarget("http://frigate:5000", "0.18.0", "large")
    plan = build_dry_run_plan(
        target,
        [person],
        [candidate],
        {"synthetic person": ("manual.jpg",)},
        detector=object(),
    )

    assert plan.entries[0]["status"] == "SKIPPED_EXISTING_LABEL"
    assert plan.entries[0]["existing_label"] == "synthetic person"
    assert plan.entries[0]["simulated_registered_image_sha256"] is None


def test_plan_marks_undetected_faces_and_rejects_duplicates():
    image = np.full((48, 64, 3), 140, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    person, candidate = make_candidate(encoded.tobytes())
    target = FrigateTarget("http://frigate:5000", "0.18.0", "large")
    plan = build_dry_run_plan(target, [person], [candidate], {}, detector=FakeYuNet(None))
    assert plan.entries[0]["status"] == "BLOCKED_NO_FACE_DETECTED"

    with pytest.raises(ValueError, match="duplicate candidates"):
        build_dry_run_plan(
            target,
            [person],
            [candidate, candidate],
            {},
            detector=detector_with_face(),
        )


def test_plan_rejects_profiles_outside_pinned_compatibility_target():
    person, candidate = make_candidate()
    with pytest.raises(ValueError):
        build_dry_run_plan(
            FrigateTarget("http://frigate:5000", "0.18.1", "large"),
            [person],
            [candidate],
            {},
            detector=detector_with_face(),
        )


def test_plan_rejects_targets_with_credentials_or_paths():
    person, candidate = make_candidate()
    with pytest.raises(ValueError, match="credential-free"):
        build_dry_run_plan(
            FrigateTarget("http://user:password@frigate:5000", "0.18.0", "large"),
            [person],
            [candidate],
            {},
            detector=detector_with_face(),
        )


def test_plan_rejects_case_colliding_remote_labels():
    person, candidate = make_candidate()
    with pytest.raises(ValueError, match="case-colliding"):
        build_dry_run_plan(
            FrigateTarget("http://frigate:5000", "0.18.0", "large"),
            [person],
            [candidate],
            {"Amy": (), "amy": ()},
            detector=detector_with_face(),
        )
