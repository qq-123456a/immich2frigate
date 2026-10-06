from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import hashlib
import json

import pytest
import numpy as np

from immich2frigate.frigate_client import FrigateTarget
from immich2frigate.expansion import (
    _calibration_groups,
    _deduplicate_candidates,
    _expansion_limit,
    _proposal_improves,
    _scan_new_candidates,
    _validation_scores,
    _inventory_digest,
    check_validation_context,
    verify_validation_context,
)
from immich2frigate.immich_client import FaceCandidate
from immich2frigate.selection import VectorFace
from immich2frigate.validation import event_partition


def test_expansion_rejects_stale_validation_report():
    report = {
        "fresh": True,
        "updated_at": (datetime.now(UTC) - timedelta(hours=7)).isoformat(),
    }

    with pytest.raises(ValueError, match="older than 120 seconds"):
        state = SimpleNamespace(pending=None, roster=[])
        check_validation_context(report, state, SimpleNamespace(), None, None)


class RuntimeFrigate:
    def __init__(self, threshold=0.95, min_faces=3):
        self.threshold = threshold
        self.min_faces = min_faces

    def face_recognition_profile(self):
        return {"enabled": True, "model_size": "large",
                "recognition_threshold": self.threshold, "min_faces": self.min_faces}

    def inventory(self):
        return {}

    def verify_target(self):
        return FrigateTarget("http://frigate:5000", "0.18.0", "large")


def _validation_context(frigate):
    roster = [{"person_id": f"person-{index}", "name": f"Person {index}",
               "enrolled_at": 1000 + index, "profile": {"version": 1}}
              for index in range(3)]
    state = SimpleNamespace(roster=roster, pending=None)
    pipeline = SimpleNamespace(profile=SimpleNamespace(metadata=lambda: {"profile_id": "test"}))
    teacher = SimpleNamespace(profile=lambda: {"model": "buffalo_l"})
    profile = {"faces": {row["person_id"]: row["profile"] for row in roster},
               "pipeline": {"profile_id": "test"}, "teacher": {"model": "buffalo_l"},
               "frigate_face_recognition": frigate.face_recognition_profile(),
               "frigate_target": {"origin": "http://frigate:5000", "version": "0.18.0",
                                  "model_size": "large"}}
    digest = _inventory_digest({})
    enrolled = sorted(str(row["enrolled_at"]) for row in roster)
    seed = json.dumps(profile, sort_keys=True, default=str, separators=(",", ":"))
    epoch = hashlib.sha256(f"{seed}|{'|'.join(enrolled)}|{digest}".encode()).hexdigest()
    events = {}
    number = 0
    for row in roster:
        count = 0
        while count < 20:
            event_id = f"validation-event-{number}"
            number += 1
            if event_partition(event_id) == "calibration":
                events[event_id] = {"teacher_person_id": row["person_id"],
                                    "category": "agree", "partition": "calibration"}
                count += 1
    report = {"fresh": True, "updated_at": datetime.now(UTC).isoformat(),
              "profile": profile, "inventory_digest": digest, "epoch": epoch,
              "events": events}
    return report, state, pipeline, teacher


def test_validation_epoch_binds_live_frigate_settings_and_expansion_requires_locked_values():
    live_legacy = RuntimeFrigate(threshold=0.9, min_faces=1)
    report, state, pipeline, teacher = _validation_context(live_legacy)

    assert verify_validation_context(report, state, live_legacy, pipeline, teacher)["epoch"] == report["epoch"]
    with pytest.raises(ValueError, match="threshold 0.95 and min_faces 3"):
        check_validation_context(report, state, live_legacy, pipeline, teacher)

    locked = RuntimeFrigate(threshold=0.95, min_faces=3)
    with pytest.raises(ValueError, match="profile differs"):
        verify_validation_context(report, state, locked, pipeline, teacher)

    locked_report, state, pipeline, teacher = _validation_context(locked)
    assert check_validation_context(locked_report, state, locked, pipeline, teacher)["epoch"] == locked_report["epoch"]


def test_expansion_never_reads_holdout_archive_rows(tmp_path):
    class Pipeline:
        calls = 0

        def evaluate_crop(self, body):
            self.calls += 1
            raise AssertionError("holdout data must not be evaluated")

    pipeline = Pipeline()
    report = {
        "events": {
            "holdout-event": {
                "partition": "holdout",
                "category": "agree",
                "teacher_person_id": "person",
                "archive_sha256": "a" * 64,
                "archive_paths": ["a" * 64 + ".webp"],
            }
        }
    }

    assert _calibration_groups(report, tmp_path, pipeline) == {}
    assert pipeline.calls == 0


def test_expansion_caps_each_batch_and_library():
    assert _expansion_limit(5) == 2
    assert _expansion_limit(29) == 1
    assert _expansion_limit(30) == 0
    with pytest.raises(ValueError):
        _expansion_limit(31)


def test_expansion_requires_strict_coverage_improvement_without_more_conflicts():
    before = {"coverage": 0.9, "conflicts": 0, "events": 20}
    assert _proposal_improves(before, {"coverage": 0.95, "conflicts": 0, "events": 20})
    assert not _proposal_improves(before, {"coverage": 0.9, "conflicts": 0, "events": 20})
    assert not _proposal_improves(before, {"coverage": 0.95, "conflicts": 1, "events": 20})
    assert not _proposal_improves(before, {"coverage": 1.0, "conflicts": 0, "events": 19})


def test_expansion_rejects_pending_batch_before_another_prepare():
    state = SimpleNamespace(
        pending=None,
        roster=[{"last_batch": {"status": "pending"}} for _ in range(3)],
    )

    with pytest.raises(ValueError, match="awaiting validation"):
        check_validation_context({"fresh": True, "updated_at": datetime.now(UTC).isoformat()}, state, None, None, None)


def test_expansion_deduplicates_known_asset_checksum_and_near_identical_embedding():
    def item(face_id, asset_id, checksum, embedding):
        source = SimpleNamespace(face_id=face_id, asset_id=asset_id, checksum=checksum, taken_at="2026-01-01")
        vector_face = SimpleNamespace(source=source)
        metrics = SimpleNamespace(ediffiqa=0.8)
        face = SimpleNamespace(embedding=np.asarray(embedding, dtype=np.float32), metrics=metrics)
        return SimpleNamespace(source=vector_face, face=face)

    known = [item("known", "asset-known", "checksum-known", [1, 0])]
    candidates = [
        item("duplicate-asset", "asset-known", "other", [0, 1]),
        item("duplicate-checksum", "asset-new", "checksum-known", [0, 1]),
        item("near-copy", "asset-copy", "checksum-copy", [0.9999, 0.01]),
        item("distinct", "asset-distinct", "checksum-distinct", [0, 1]),
    ]

    result = _deduplicate_candidates(candidates, known)

    assert [candidate.source.source.face_id for candidate in result] == ["distinct"]


def test_whole_batch_validation_counts_cross_person_regressions():
    groups = {}
    for person_id, embedding in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
        for index in range(20):
            groups[f"{person_id}-{index}"] = {
                "person_id": person_id,
                "embeddings": [np.asarray(embedding, dtype=np.float32)] * 3,
                "sharpness": 300.0,
            }
    current = {"a": np.asarray([1.0, 0.0]), "b": np.asarray([0.0, 1.0])}
    trial = {"a": np.asarray([0.0, 1.0]), "b": np.asarray([1.0, 0.0])}

    before = _validation_scores(current, groups)
    after = _validation_scores(trial, groups)

    assert before["coverage"] == 1.0
    assert before["conflicts"] == 0
    assert after["conflicts"] > before["conflicts"]
    assert not _proposal_improves(before, after)


def test_expansion_reconsiders_eligible_historical_examined_sources(monkeypatch):
    person_id = "person"
    candidates = []
    for index in range(7):
        face_id = f"face-{index}"
        source = FaceCandidate(
            person_id=person_id,
            face_id=face_id,
            asset_id=f"asset-{index}",
            taken_at=f"2026-01-{index + 1:02d}",
            checksum=f"checksum-{index}",
            box=(0.1, 0.1, 0.9, 0.9),
            frame=(200, 200),
        )
        candidates.append(VectorFace(source, np.asarray([1.0, index / 10], dtype=np.float32)))

    class Vectors:
        def candidates_for_person(self, person, *, years, include_face_ids):
            assert years == 6
            assert include_face_ids == ("face-0",)
            return candidates

    class Immich:
        previews = []

        def preview(self, asset_id):
            self.previews.append(asset_id)
            return b"preview"

    monkeypatch.setattr("immich2frigate.expansion._one_face_per_asset", lambda items, excluded_assets: (items, []))
    monkeypatch.setattr(
        "immich2frigate.expansion._validated_matrix",
        lambda person, items, attr: np.ones((len(items), 2), dtype=np.float32),
    )
    monkeypatch.setattr("immich2frigate.expansion._row_normalize", lambda matrix: matrix)
    monkeypatch.setattr(
        "immich2frigate.expansion._identity_safe_mask",
        lambda matrix, minimum_keep: np.ones(len(matrix), dtype=bool),
    )
    monkeypatch.setattr(
        "immich2frigate.expansion.prepare_candidate_upload",
        lambda source, preview: SimpleNamespace(encoded=b"upload", face_box=(0, 0, 1, 1)),
    )
    pipeline = SimpleNamespace(evaluate=lambda encoded, box: SimpleNamespace(eligible=True))
    row = {
        "face_ids": ["face-0"],
        "examined_face_ids": [f"face-{index}" for index in range(1, 7)],
        "rejected_face_ids": ["face-6"],
    }
    immich = Immich()

    result = _scan_new_candidates(row, SimpleNamespace(person_id=person_id), immich, Vectors(), pipeline, years=6)

    assert [item.source.source.face_id for item in result] == [f"face-{index}" for index in range(1, 6)]
    assert len(immich.previews) == 5
