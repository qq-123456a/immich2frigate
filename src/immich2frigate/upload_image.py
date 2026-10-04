"""Prepare an Immich face candidate for Frigate's registration endpoint.

The Immich face coordinates refer to the original asset dimensions while the
preview returned by :class:`ImmichReadOnlyClient` may be resized.  This module
keeps that coordinate conversion explicit and performs all work in memory:
there is no temporary image file or cache involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .immich_client import FaceCandidate

_DEFAULT_CONTEXT = 0.5
_DEFAULT_ASPECT_TOLERANCE = 0.01
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ImageFormat = Literal["jpeg", "jpg", "webp"]


@dataclass(frozen=True, slots=True)
class CandidateUpload:
    """In-memory image request body and the crop used to produce it."""

    encoded: bytes = field(repr=False)
    content_type: str
    extension: str
    crop_box: tuple[int, int, int, int]
    crop_bgr: np.ndarray = field(repr=False)


def prepare_candidate_upload(
    candidate: FaceCandidate,
    preview_bgr: np.ndarray,
    *,
    image_format: ImageFormat = "webp",
    context: float = _DEFAULT_CONTEXT,
    aspect_tolerance: float = _DEFAULT_ASPECT_TOLERANCE,
    quality: int = 100,
    max_upload_bytes: int = _MAX_UPLOAD_BYTES,
) -> CandidateUpload:
    """Crop and encode one Immich candidate from its in-memory preview.

    ``candidate.box`` is in ``candidate.frame`` (original asset) coordinates.
    The box is scaled to the actual preview dimensions only when both axes have
    the same scale within ``aspect_tolerance``.  A half-box of context on each
    side mirrors the context used by the upstream face selector and gives
    Frigate enough surrounding pixels to redetect the face.  The final box is
    clipped to the preview, and the encoded bytes are suitable for a multipart
    request to ``/api/faces/{name}/register``.
    """

    cv2 = _opencv()
    image = _require_bgr(preview_bgr)
    width, height = _require_candidate(candidate)
    context = _require_fraction(context, "context", allow_zero=True)
    aspect_tolerance = _require_fraction(
        aspect_tolerance, "aspect_tolerance", allow_zero=True
    )
    if (
        isinstance(quality, bool)
        or not isinstance(quality, int)
        or not 0 <= quality <= 100
    ):
        raise ValueError("quality must be an integer between 0 and 100")
    if (
        isinstance(max_upload_bytes, bool)
        or not isinstance(max_upload_bytes, int)
        or max_upload_bytes <= 0
    ):
        raise ValueError("max_upload_bytes must be a positive integer")

    preview_height, preview_width = image.shape[:2]
    scale_x = preview_width / width
    scale_y = preview_height / height
    if scale_y <= 0 or abs(scale_x / scale_y - 1.0) > aspect_tolerance:
        raise ValueError(
            "Immich preview aspect ratio does not match the candidate frame"
        )

    x1, y1, x2, y2 = candidate.box
    scaled_x1, scaled_y1 = x1 * scale_x, y1 * scale_y
    scaled_x2, scaled_y2 = x2 * scale_x, y2 * scale_y
    box_width = scaled_x2 - scaled_x1
    box_height = scaled_y2 - scaled_y1
    scaled_x1 -= box_width * context
    scaled_y1 -= box_height * context
    scaled_x2 += box_width * context
    scaled_y2 += box_height * context

    # floor/ceil preserve the entire requested face box.  Clipping happens
    # after rounding so an edge-touching face can never produce an invalid
    # negative slice.
    crop_box = (
        max(0, min(preview_width - 1, math.floor(scaled_x1))),
        max(0, min(preview_height - 1, math.floor(scaled_y1))),
        max(1, min(preview_width, math.ceil(scaled_x2))),
        max(1, min(preview_height, math.ceil(scaled_y2))),
    )
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("candidate crop is empty after scaling and clipping")
    crop = np.ascontiguousarray(image[crop_y1:crop_y2, crop_x1:crop_x2])
    if crop.size == 0:
        raise ValueError("candidate crop is empty after scaling and clipping")

    extension, content_type, params = _encoding_options(cv2, image_format, quality)
    success, encoded = cv2.imencode(extension, crop, params)
    if not success or encoded is None:
        raise ValueError("candidate preview could not be encoded")
    body = encoded.tobytes()
    if not body:
        raise ValueError("candidate preview encoded to an empty body")
    if len(body) > max_upload_bytes:
        raise ValueError("candidate upload exceeds the configured byte limit")
    return CandidateUpload(
        encoded=body,
        content_type=content_type,
        extension=extension[1:],
        crop_box=crop_box,
        crop_bgr=crop,
    )


def _require_candidate(candidate: FaceCandidate) -> tuple[int, int]:
    if not isinstance(candidate, FaceCandidate):
        raise ValueError("candidate must be a FaceCandidate")
    frame_width, frame_height = candidate.frame
    if (
        isinstance(frame_width, bool)
        or isinstance(frame_height, bool)
        or not isinstance(frame_width, int)
        or not isinstance(frame_height, int)
        or frame_width <= 0
        or frame_height <= 0
    ):
        raise ValueError("candidate frame must contain positive integer dimensions")
    if len(candidate.box) != 4 or not all(_finite(value) for value in candidate.box):
        raise ValueError("candidate box must contain four finite coordinates")
    x1, y1, x2, y2 = candidate.box
    if not (0 <= x1 < x2 <= frame_width and 0 <= y1 < y2 <= frame_height):
        raise ValueError("candidate box is outside its frame")
    return frame_width, frame_height


def _require_bgr(image: np.ndarray) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] <= 0
        or image.shape[1] <= 0
    ):
        raise ValueError("preview must be a non-empty uint8 BGR image")
    return image


def _require_fraction(value: float, label: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{label} must be a finite non-negative number")
    return value


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _encoding_options(cv2, image_format: str, quality: int):
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in {"jpeg", "webp"}:
        raise ValueError("image_format must be jpeg, jpg, or webp")
    if image_format == "webp":
        return ".webp", "image/webp", [cv2.IMWRITE_WEBP_QUALITY, quality]
    return ".jpg", "image/jpeg", [cv2.IMWRITE_JPEG_QUALITY, quality]


def _opencv():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to encode candidate previews") from error
    return cv2
