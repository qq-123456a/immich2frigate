from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from immich2frigate.frigate_registration import (
    align_face_bgr,
    laplacian_variance_bgr,
    prepare_registered_face,
    prepare_registered_upload,
)


class FakeYuNet:
    def __init__(self, rows):
        self.rows = rows
        self.input_size = None

    def setInputSize(self, size):
        self.input_size = size

    def detect(self, image):
        return 1, self.rows


def row(x, y, width, height, score=0.9):
    return np.array([x, y, width, height, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, score])


def test_registration_scales_detects_largest_and_encodes_quality_100_webp():
    image = np.zeros((2160, 1000, 3), dtype=np.uint8)
    image[40:200, 20:220] = (20, 80, 140)
    detector = FakeYuNet(np.array([row(10, 20, 100, 80), row(120, 20, 80, 60)]))

    result = prepare_registered_face(image, detector)

    assert detector.input_size == (500, 1080)
    assert result.box == (20, 40, 220, 200)
    assert result.crop_bgr.shape == (160, 200, 3)
    assert result.stored_webp[:4] == b"RIFF"
    assert result.stored_webp[8:12] == b"WEBP"
    decoded = cv2.imdecode(np.frombuffer(result.stored_webp, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == result.crop_bgr.shape


def test_registration_uses_first_face_on_equal_area_and_ignores_low_confidence():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    detector = FakeYuNet(
        np.array([row(10, 10, 40, 40, 0.49), row(60, 30, 50, 50), row(150, 40, 50, 50)])
    )

    result = prepare_registered_face(image, detector)

    assert result.box == (60, 30, 110, 80)


def test_final_upload_bytes_are_decoded_before_resimulation():
    source = np.full((64, 80, 3), 90, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", source)
    assert ok
    detector = FakeYuNet(np.array([row(5, 6, 30, 25)]))

    result = prepare_registered_upload(encoded.tobytes(), detector)

    assert result is not None
    assert result.box == (5, 6, 35, 31)
    with pytest.raises(ValueError, match="decoded"):
        prepare_registered_upload(b"synthetic-not-an-image", detector)
    assert prepare_registered_upload(encoded.tobytes(), FakeYuNet(None)) is None


class FakeLandmarks:
    def __init__(self):
        self.points = np.zeros((68, 2), dtype=np.float32)
        self.points[42:48] = (30, 35)
        self.points[36:42] = (70, 35)

    def fit(self, gray, rect):
        assert gray.ndim == 2
        assert rect.tolist() == [[0, 0, 100, 80]]
        return True, [self.points[None, ...]]


def test_lbf_alignment_and_unaligned_crop_blur_metric():
    crop = np.full((80, 100, 3), 128, dtype=np.uint8)

    aligned = align_face_bgr(crop, FakeLandmarks())

    assert aligned.shape == crop.shape
    assert aligned.dtype == np.uint8
    assert laplacian_variance_bgr(crop) == 0
