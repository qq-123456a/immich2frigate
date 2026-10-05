from __future__ import annotations

import numpy as np

from immich2frigate.immich_client import FaceCandidate
from immich2frigate.training_quality import assess_training_crop

PERSON = "00000000-0000-4000-8000-000000000001"
FACE = "00000000-0000-4000-8000-000000000002"
ASSET = "10000000-0000-4000-8000-000000000003"


def candidate(box=(25.0, 25.0, 75.0, 75.0), frame=(100, 100)):
    return FaceCandidate(
        person_id=PERSON,
        face_id=FACE,
        asset_id=ASSET,
        taken_at="2025-01-01T00:00:00Z",
        checksum="x",
        box=box,
        frame=frame,
    )


def clear_color_image():
    image = np.empty((100, 100, 3), dtype=np.uint8)
    image[::2, ::2] = (30, 180, 240)
    image[1::2, ::2] = (220, 40, 80)
    image[::2, 1::2] = (220, 40, 80)
    image[1::2, 1::2] = (30, 180, 240)
    return image


def test_clear_color_large_face_is_foundation_eligible():
    item = candidate()
    quality = assess_training_crop(item, clear_color_image(), face_box=item.box)

    assert quality.training_eligible is True
    assert quality.foundation_eligible is True
    assert quality.face_area == 2500
    assert quality.face_area_ratio == 0.25
    assert quality.context_retention == 1.0


def test_grayscale_or_small_face_is_rejected():
    grayscale = np.full((100, 100, 3), 120, dtype=np.uint8)
    item = candidate()
    gray_quality = assess_training_crop(item, grayscale, face_box=item.box)
    assert gray_quality.training_eligible is False

    small = candidate(box=(40.0, 40.0, 60.0, 60.0))
    small_quality = assess_training_crop(small, clear_color_image(), face_box=small.box)
    assert small_quality.training_eligible is False


def test_edge_clipped_face_reports_context_without_becoming_a_hard_rejection():
    item = candidate(box=(0.0, 25.0, 50.0, 75.0))
    quality = assess_training_crop(item, clear_color_image(), face_box=item.box)

    assert quality.training_eligible is True
    assert quality.foundation_eligible is True
    assert quality.context_retention == 0.0


def test_scaled_face_dimensions_and_metrics_ignore_surrounding_context():
    image = clear_color_image()
    image[25:75, 25:75] = 120
    quality = assess_training_crop(
        candidate(box=(1000.0, 1000.0, 1050.0, 1050.0), frame=(6000, 4000)),
        image,
        face_box=(25.0, 25.0, 30.0, 30.0),
    )

    assert quality.face_area == 25
    assert quality.training_eligible is False


def test_face_crop_minimum_dimension_blocks_degenerate_boxes():
    item = candidate(box=(25.0, 25.0, 26.0, 95.0))
    quality = assess_training_crop(item, clear_color_image(), face_box=item.box)

    assert quality.training_eligible is False
