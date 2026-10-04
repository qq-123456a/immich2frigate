from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from immich2frigate.immich_client import FaceCandidate
from immich2frigate.upload_image import prepare_candidate_upload


PERSON = "00000000-0000-4000-8000-000000000001"
ASSET = "00000000-0000-4000-8000-000000000002"


def candidate(box=(20.0, 10.0, 60.0, 50.0), frame=(100, 80)):
    return FaceCandidate(
        person_id=PERSON,
        face_id=None,
        asset_id=ASSET,
        taken_at="2025-01-01T00:00:00Z",
        checksum="checksum",
        box=box,
        frame=frame,
    )


def test_scales_original_box_adds_context_and_encodes_webp_in_memory():
    preview = np.full((160, 200, 3), 128, dtype=np.uint8)

    result = prepare_candidate_upload(candidate(), preview)

    # The candidate is 2x in the preview: (40,20)-(120,100).  A half-box of
    # context on each side produces (0,0)-(160,140) after clipping.
    assert result.crop_box == (0, 0, 160, 140)
    assert result.crop_bgr.shape == (140, 160, 3)
    assert result.content_type == "image/webp"
    assert result.extension == "webp"
    assert result.encoded[:4] == b"RIFF"
    assert result.encoded[8:12] == b"WEBP"
    decoded = cv2.imdecode(np.frombuffer(result.encoded, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == result.crop_bgr.shape


def test_jpeg_encoding_is_supported_for_multipart_uploads():
    preview = np.zeros((80, 100, 3), dtype=np.uint8)

    result = prepare_candidate_upload(candidate(), preview, image_format="jpeg", quality=90)

    assert result.content_type == "image/jpeg"
    assert result.extension == "jpg"
    assert result.encoded[:2] == b"\xff\xd8"
    assert cv2.imdecode(np.frombuffer(result.encoded, np.uint8), cv2.IMREAD_COLOR) is not None


def test_aspect_ratio_mismatch_fails_closed():
    preview = np.zeros((160, 300, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="aspect ratio"):
        prepare_candidate_upload(candidate(), preview)


def test_edge_box_is_clipped_to_a_non_empty_preview_crop():
    preview = np.zeros((80, 100, 3), dtype=np.uint8)
    edge = candidate(box=(0.0, 0.0, 2.0, 2.0))

    result = prepare_candidate_upload(edge, preview)

    x1, y1, x2, y2 = result.crop_box
    assert 0 <= x1 < x2 <= 100
    assert 0 <= y1 < y2 <= 80
    assert result.crop_bgr.size > 0


@pytest.mark.parametrize(
    "bad_box",
    [(0.0, 0.0, 0.0, 2.0), (-1.0, 0.0, 2.0, 2.0), (0.0, 0.0, 101.0, 2.0)],
)
def test_invalid_candidate_boxes_are_rejected(bad_box):
    preview = np.zeros((80, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="candidate box"):
        prepare_candidate_upload(candidate(box=bad_box), preview)


def test_upload_size_limit_is_enforced_after_encoding():
    preview = np.zeros((80, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="byte limit"):
        prepare_candidate_upload(candidate(), preview, max_upload_bytes=1)
