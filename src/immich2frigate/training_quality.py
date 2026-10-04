"""Frigate-aligned, non-ML quality checks for training candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frigate_registration import laplacian_variance_bgr
from .immich_client import FaceCandidate

MIN_FACE_AREA = 750
MIN_TRAINING_SHARPNESS = 200.0
MIN_FOUNDATION_SHARPNESS = 250.0
MIN_COLOR_SPREAD = 3.0
MIN_MEAN_LUMA = 30.0
MAX_MEAN_LUMA = 225.0


@dataclass(frozen=True, slots=True)
class TrainingQuality:
    """Simple photometric checks derived from Frigate's published guidance."""

    training_eligible: bool
    foundation_eligible: bool
    sharpness: float
    face_area: float
    color_spread: float
    mean_luma: float


def assess_training_crop(
    candidate: FaceCandidate,
    crop_bgr: np.ndarray,
) -> TrainingQuality:
    """Assess whether a crop is suitable for foundation or expansion training.

    This deliberately does not run a pose, expression, scene, or identity model.
    Sharpness uses the same Laplacian variance metric Frigate uses for its blur
    confidence filter. The remaining checks only reject very small, effectively
    grayscale, or strongly under/over-exposed crops.
    """

    image = _require_bgr(crop_bgr)
    x1, y1, x2, y2 = candidate.box
    face_area = float((x2 - x1) * (y2 - y1))
    sharpness = laplacian_variance_bgr(image)

    pixels = image.astype(np.float32, copy=False)
    channel_spread = float(
        np.mean(np.max(pixels, axis=2) - np.min(pixels, axis=2), dtype=np.float64)
    )
    # BGR luminance approximation; only used to reject extreme exposure.
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
        color_spread=channel_spread,
        mean_luma=mean_luma,
    )


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
