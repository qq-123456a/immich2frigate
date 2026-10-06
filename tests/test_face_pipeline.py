from __future__ import annotations

import numpy as np
import pytest

from immich2frigate.face_pipeline import (
    FaceMetrics,
    FacePipeline,
    FaceProfile,
    PreparedCandidate,
    PreparedFace,
    cross_person_margin,
    select_foundation,
)
from immich2frigate.frigate_registration import prepare_registered_face
from immich2frigate.immich_client import FaceCandidate, PersonRecord
from immich2frigate.selection import VectorFace

cv2 = pytest.importorskip("cv2")


def _row(x, y, width, height, confidence=0.9):
    return np.array([x, y, width, height, *([0] * 10), confidence], dtype=np.float32)


class Detector:
    def __init__(self, rows):
        self.rows = rows

    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        return 1, self.rows


class Session:
    def __init__(self, name, shape, output):
        self.input = type("Meta", (), {"name": name, "shape": shape})()
        self.output = type("Meta", (), {"name": "output"})()
        self.result = output
        self.feed = None

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [self.output]

    def run(self, names, feed):
        self.feed = next(iter(feed.values()))
        return [self.result]


class Landmarks:
    def fit(self, gray, rect):
        points = np.zeros((68, 2), dtype=np.float32)
        points[42:48] = (30, 35)
        points[36:42] = (70, 35)
        return True, [points[None]]


def pipeline(rows):
    arc = Session("arc", [1, 3, 112, 112], np.ones((1, 512), dtype=np.float32))
    pose = Session("pose", [1, 3, 224, 224], np.eye(3, dtype=np.float32)[None])
    fiqa = Session("fiqa", [1, 3, 112, 112], np.array([[0.9]], dtype=np.float32))
    profile = FaceProfile("test-profile", {}, cv2.__version__, "test")
    return FacePipeline(Detector(rows), Landmarks(), arc, pose, fiqa, profile), (arc, pose, fiqa)


def test_evaluate_uses_registered_webp_readback_and_rejects_ambiguous_faces():
    rng = np.random.default_rng(9)
    image = rng.integers(50, 180, (120, 120, 3), dtype=np.uint8)
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    engine, sessions = pipeline([_row(10, 10, 100, 100)])

    result = engine.evaluate(encoded, (10, 10, 110, 110))

    assert result.eligible
    assert result.stored_webp[8:12] == b"WEBP"
    assert result.embedding.shape == (512,)
    assert result.metrics.target_iou == 1.0
    assert sessions[1].feed.shape == (1, 3, 224, 224)
    assert sessions[2].feed.shape == (1, 3, 112, 112)

    ambiguous = pipeline_for([_row(10, 10, 100, 100), _row(20, 20, 70, 70)])
    with pytest.raises(ValueError, match="ambiguous"):
        ambiguous.evaluate(encoded, (10, 10, 110, 110))


def pipeline_for(rows):
    return pipeline(rows)[0]


def test_readback_requires_webp_and_iou_rejects_wrong_or_out_of_bounds_target():
    engine, _ = pipeline([_row(10, 10, 100, 100)])
    with pytest.raises(ValueError, match="not WebP"):
        engine.validate_readback(b"not-a-webp-image")
    encoded = cv2.imencode(".jpg", np.zeros((120, 120, 3), dtype=np.uint8))[1].tobytes()
    with pytest.raises(ValueError, match="fit inside"):
        engine.evaluate(encoded, (0, 0, 200, 200))


def test_negative_yunet_coordinates_fail_closed_instead_of_wrapping_uint16():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert prepare_registered_face(image, Detector([_row(-1, 10, 50, 50)])) is None


def test_cross_person_margin_excludes_same_asset_checksum_and_capture_day():
    def candidate(person, index, day, vector):
        source = FaceCandidate(
            person_id=person,
            face_id=f"00000000-0000-4000-8000-{index:012d}",
            asset_id=f"10000000-0000-4000-8000-{index:012d}",
            taken_at=f"2025-01-{day:02d}T12:00:00Z",
            checksum=f"checksum-{index}",
            box=(0, 0, 100, 100),
            frame=(100, 100),
        )
        return type("Item", (), {
            "source": VectorFace(source, np.array([1, 0], np.float32)),
            "face": type("Prepared", (), {"embedding": np.array(vector, np.float32)})(),
        })()

    target = candidate("person-a", 1, 1, [1, 0])
    own = candidate("person-a", 2, 2, [1, 0])
    same_day = candidate("person-b", 3, 1, [1, 0])
    other = candidate("person-b", 4, 3, [0, 1])
    assert cross_person_margin(target, {"person-a": [own], "person-b": [same_day, other]}) == pytest.approx(1.0)


def test_foundation_retries_after_teacher_rejection():
    person = PersonRecord("00000000-0000-4000-8000-000000000001", "Person")
    metrics = FaceMetrics(100, 100, 300, 40, 100, .9, 0, 0, 0, 1)
    candidates = []
    for index in range(1, 8):
        angle = index * .01
        candidate = FaceCandidate(
            person_id=person.person_id,
            face_id=f"00000000-0000-4000-8000-{index:012d}",
            asset_id=f"10000000-0000-4000-8000-{index:012d}",
            taken_at=f"2025-01-{index:02d}T12:00:00Z",
            checksum=f"checksum-{index}",
            box=(0, 0, 100, 100),
            frame=(100, 100),
        )
        vector = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        source = VectorFace(candidate, vector)
        frigate_embedding = np.zeros(512, dtype=np.float32)
        frigate_angle = index * .24
        frigate_embedding[:2] = [np.cos(frigate_angle), np.sin(frigate_angle)]
        prepared = PreparedFace(b"WEBP", frigate_embedding, metrics, True)
        candidates.append(PreparedCandidate(source, b"upload", prepared, identity_safe=True))

    calls = 0
    rejected_ids = set()

    def teacher(candidate):
        nonlocal calls
        calls += 1
        if calls == 1:
            rejected_ids.add(candidate.source.source.face_id)
        return calls != 1

    result = select_foundation(person, candidates, teacher=teacher)

    assert len(result) == 5
    assert calls == 6
    assert rejected_ids.isdisjoint(item.source.source.face_id for item in result)
