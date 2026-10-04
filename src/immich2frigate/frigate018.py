"""Pure ArcFace compatibility primitives derived from Frigate v0.18.0.

This module performs no network or filesystem operations. It is not by itself
an end-to-end enrollment compatibility guarantee.
"""

from __future__ import annotations

import math
import re

import numpy as np
from PIL import Image

PIPELINE_ID = "frigate-0.18.0-large-yunet-v1"
TARGET_VERSION = "0.18.0"
TARGET_MODEL_SIZE = "large"
TARGET_COMMIT = "77a66e75c61862b048a07c1295877f4b31343504"


def require_target(version: str, model_size: str) -> None:
    """Fail closed when the runtime is outside the source version we inspected."""
    version_matches = re.fullmatch(r"0\.18\.0(?:-([0-9a-fA-F]{7,40}))?", version)
    commit_matches = version_matches and (
        version_matches.group(1) is None
        or TARGET_COMMIT.startswith(version_matches.group(1).lower())
    )
    if not commit_matches or model_size != TARGET_MODEL_SIZE:
        raise ValueError(
            f"This compatibility profile requires Frigate {TARGET_VERSION} with the "
            f"{TARGET_MODEL_SIZE!r} face model (got {version!r}, {model_size!r})"
        )


def arcface_preprocess(bgr: np.ndarray) -> np.ndarray:
    """Reproduce Frigate v0.18 large-model ArcFace input preprocessing.

    Input is an OpenCV BGR crop. Output is a float32 NCHW batch in RGB order,
    aspect-fit to 112×112, centered on black, and normalized to [-1, 1].
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.shape[0] or not bgr.shape[1]:
        raise ValueError("bgr must be a non-empty H×W×3 image")
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    image = Image.fromarray(rgb)
    width, height = image.size
    if (width, height) != (112, 112):
        if width > height:
            image = image.resize((112, int(height / width * 112 // 4 * 4)))
        else:
            image = image.resize((int(width / height * 112 // 4 * 4), 112))
    pixels = np.asarray(image, dtype=np.float32)
    frame = np.zeros((112, 112, 3), dtype=np.float32)
    y = (112 - pixels.shape[0]) // 2
    x = (112 - pixels.shape[1]) // 2
    frame[y : y + pixels.shape[0], x : x + pixels.shape[1]] = pixels
    return ((frame / 127.5 - 1.0).transpose(2, 0, 1))[None]


def _trim_mean(embeddings: np.ndarray, proportion: float = 0.15) -> np.ndarray:
    values = np.sort(embeddings, axis=0)
    cut = int(proportion * len(values))
    retained = values[cut : len(values) - cut] if cut else values
    return retained.mean(axis=0)


def build_class_mean(
    embeddings: list[np.ndarray],
    trim: float = 0.15,
    outlier_threshold: float = 0.30,
    min_keep_frac: float = 0.7,
    max_iters: int = 3,
) -> np.ndarray:
    """Match Frigate v0.18's vector outlier filter and per-dimension trim mean."""
    if not embeddings or not 0 <= trim < 0.5 or not 0 <= min_keep_frac <= 1:
        raise ValueError("embeddings must be non-empty and trim parameters must be valid")
    values = np.stack(embeddings, axis=0)
    if (
        values.ndim != 2
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise ValueError("embeddings must be a finite 2D floating-point array")
    if len(values) < 5:
        return _trim_mean(values, trim)

    keep = np.ones(len(values), dtype=bool)
    floor = max(5, int(np.ceil(min_keep_frac * len(values))))
    for _ in range(max_iters):
        center = _trim_mean(values[keep], trim)
        center /= np.linalg.norm(center) + 1e-9
        units = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)
        cosine = units @ center
        next_keep = cosine >= outlier_threshold
        if next_keep.sum() < floor:
            next_keep = np.zeros(len(values), dtype=bool)
            next_keep[np.argsort(-cosine)[:floor]] = True
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    return _trim_mean(values[keep], trim)


def similarity_to_confidence(cosine_similarity: float) -> float:
    """Match Frigate v0.18 ArcFace's default sigmoid confidence mapping."""
    return 1.0 / (1.0 + math.exp(-20.0 * (cosine_similarity - 0.3)))


def blur_confidence_reduction(laplacian_variance: float, enabled: bool = True) -> float:
    """Return the v0.18 blur penalty from the crop's Laplacian variance."""
    if not enabled:
        return 0.0
    if laplacian_variance < 120:
        return 0.06
    if laplacian_variance < 160:
        return 0.04
    if laplacian_variance < 200:
        return 0.02
    if laplacian_variance < 250:
        return 0.01
    return 0.0


def reported_confidence(cosine_similarity: float, laplacian_variance: float) -> float:
    """Match Frigate v0.18's one-shot score after blur penalty and rounding."""
    score = similarity_to_confidence(cosine_similarity)
    reduction = blur_confidence_reduction(laplacian_variance)
    return max(0.0, round(score - reduction, 2))
