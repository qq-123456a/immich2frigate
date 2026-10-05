"""Frigate-aligned, non-ML quality checks for training candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frigate_registration import laplacian_variance_bgr
from .immich_client import FaceCandidate

MIN_FACE_AREA = 750
MIN_FACE_DIMENSION = 20
MIN_TRAINING_SHARPNESS = 200.0
MIN_FOUNDATION_SHARPNESS = 250.0
MIN_COLOR_SPREAD = 3.0
MIN_MEAN_LUMA = 30.0
MAX_MEAN_LUMA = 225.0
_CONTEXT_RATIO = 0.5


@dataclass(frozen=True, slots=True)
class TrainingQuality:
    """Simple photometric and crop-safety checks derived from Frigate guidance."""

    training_eligible: bool
    foundation_eligible: bool
    sharpness: float
    face_area: float
    face_area_ratio: float
    context_retention: float
    color_spread: float
    mean_luma: float


def assess_training_crop(
    candidate: FaceCandidate,
    crop_bgr: np.ndarray,
    *,
    face_box: tuple[float, float, float, float],
) -> TrainingQuality:
    """Assess whether a crop is suitable for foundation or expansion training.

    This deliberately does not run a pose, expression, scene, or identity model.
    Sharpness uses the same Laplacian variance metric Frigate uses for its blur
    confidence filter. The remaining checks reject very small, effectively
    grayscale, or strongly under/over-exposed faces. When supplied, ``face_box``
    measures these checks on the scaled Immich face region so surrounding
    context cannot hide a poor or downscaled face.
    """

    image = _require_bgr(crop_bgr)
    x1, y1, x2, y2 = candidate.box
    frame_width, frame_height = candidate.frame
    face_width = x2 - x1
    face_height = y2 - y1
    source_face_area = float(face_width * face_height)
    face_area_ratio = source_face_area / float(frame_width * frame_height)
    context_retention = _context_retention(candidate)
    if len(face_box) != 4 or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        for value in face_box
    ):
        raise ValueError("face_box must contain four finite coordinates")
    face_x1, face_y1, face_x2, face_y2 = face_box
    if not (0 <= face_x1 < face_x2 <= image.shape[1] and 0 <= face_y1 < face_y2 <= image.shape[0]):
        raise ValueError("face_box must fit inside the training crop")
    quality_image = image[
        int(np.floor(face_y1)) : int(np.ceil(face_y2)),
        int(np.floor(face_x1)) : int(np.ceil(face_x2)),
    ]
    if quality_image.size == 0:
        raise ValueError("face_box produced an empty training crop")
    measured_width = face_x2 - face_x1
    measured_height = face_y2 - face_y1
    face_area = float(measured_width * measured_height)
    sharpness = laplacian_variance_bgr(quality_image)

    pixels = quality_image.astype(np.float32, copy=False)
    channel_spread = float(
        np.mean(np.max(pixels, axis=2) - np.min(pixels, axis=2), dtype=np.float64)
    )
    mean_luma = float(
        np.mean(
            pixels[:, :, 0] * 0.114
            + pixels[:, :, 1] * 0.587
            + pixels[:, :, 2] * 0.299,
            dtype=np.float64,
        )
    )

    common = (
        face_area >= MIN_FACE_AREA
        and min(measured_width, measured_height) >= MIN_FACE_DIMENSION
        and channel_spread >= MIN_COLOR_SPREAD
        and MIN_MEAN_LUMA <= mean_luma <= MAX_MEAN_LUMA
    )
    training_eligible = common and sharpness >= MIN_TRAINING_SHARPNESS
    foundation_eligible = common and sharpness >= MIN_FOUNDATION_SHARPNESS

    return TrainingQuality(
        training_eligible=training_eligible,
        foundation_eligible=foundation_eligible,
        sharpness=sharpness,
        face_area=face_area,
        face_area_ratio=face_area_ratio,
        context_retention=context_retention,
        color_spread=channel_spread,
        mean_luma=mean_luma,
    )


def _context_retention(candidate: FaceCandidate) -> float:
    x1, y1, x2, y2 = candidate.box
    frame_width, frame_height = candidate.frame
    face_width = x2 - x1
    face_height = y2 - y1
    desired_x = face_width * _CONTEXT_RATIO
    desired_y = face_height * _CONTEXT_RATIO
    if desired_x <= 0 or desired_y <= 0:
        return 0.0
    ratios = (
        x1 / desired_x,
        (frame_width - x2) / desired_x,
        y1 / desired_y,
        (frame_height - y2) / desired_y,
    )
    return float(max(0.0, min(1.0, min(ratios))))


def _require_bgr(image: np.ndarray) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] <= 0
        or image.shape[1] <= 0
    ):
        raise ValueError("training crop must be a non-empty uint8 BGR image")
    return image
