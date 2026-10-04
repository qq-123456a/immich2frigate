"""Local simulation of Frigate 0.18 face registration and alignment.

OpenCV and the Frigate YuNet/LBF model assets are optional runtime inputs. This
module does not load models, access a service, or write files. Callers must pass
the same verified detector/model configuration used by the Frigate instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_MAX_DETECTION_HEIGHT = 1080
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_REGISTER_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class RegisteredFace:
    """The exact WebP Frigate derives from an uploaded BGR image."""

    stored_webp: bytes = field(repr=False)
    crop_bgr: np.ndarray = field(repr=False)
    box: tuple[int, int, int, int]


def prepare_registered_face(image_bgr: np.ndarray, detector) -> RegisteredFace | None:
    """Simulate Frigate's largest-face crop and quality-100 WebP registration.

    `detector` must implement Frigate's OpenCV YuNet interface: `setInputSize`
    and `detect`, with `detect` returning `(status, rows)` where each row has
    `[x, y, width, height, ..., confidence]` in the detector input coordinates.
    The caller remains responsible for confirming detector/model provenance.
    """
    cv2 = _opencv()
    image = _require_bgr(image_bgr)
    height, width = image.shape[:2]
    scale = 1.0
    detect_image = image
    if height > _MAX_DETECTION_HEIGHT:
        scale = _MAX_DETECTION_HEIGHT / height
        scaled_width = int(scale * width)
        detect_image = cv2.resize(image, (scaled_width, _MAX_DETECTION_HEIGHT))

    detector.setInputSize((detect_image.shape[1], detect_image.shape[0]))
    detection = detector.detect(detect_image)
    rows = detection[1] if isinstance(detection, tuple) and len(detection) > 1 else None
    if rows is None:
        return None

    largest_box: tuple[int, int, int, int] | None = None
    largest_area = -1
    for row in rows:
        row = np.asarray(row)
        if row.ndim != 1 or row.size < 5 or float(row[-1]) < _REGISTER_THRESHOLD:
            continue
        raw_box = row[:4].astype(np.uint16)
        x = int(max(raw_box[0], 0) / scale)
        y = int(max(raw_box[1], 0) / scale)
        box_width = int(raw_box[2] / scale)
        box_height = int(raw_box[3] / scale)
        box = (x, y, x + box_width, y + box_height)
        area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        if largest_box is None or area > largest_area:
            largest_box = box
            largest_area = area

    if largest_box is None:
        return None
    x1, y1, x2, y2 = largest_box
    crop = image[y1:y2, x1:x2]
    success, encoded = cv2.imencode(
        ".webp", crop, [cv2.IMWRITE_WEBP_QUALITY, 100]
    )
    if not success or encoded is None:
        raise ValueError("Frigate-compatible face crop could not be encoded as WebP")
    return RegisteredFace(
        stored_webp=encoded.tobytes(),
        crop_bgr=np.ascontiguousarray(crop),
        box=largest_box,
    )


def prepare_registered_upload(upload_bytes: bytes, detector) -> RegisteredFace | None:
    """Decode and simulate the final bytes Frigate receives on its API route."""
    if not isinstance(upload_bytes, bytes) or not upload_bytes:
        raise ValueError("upload_bytes must be a non-empty byte string")
    if len(upload_bytes) > _MAX_UPLOAD_BYTES:
        raise ValueError("upload exceeds Frigate's 20 MiB request-body limit")
    cv2 = _opencv()
    encoded = np.frombuffer(upload_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("upload bytes could not be decoded as a color image")
    return prepare_registered_face(image, detector)


def align_face_bgr(image_bgr: np.ndarray, landmark_detector) -> np.ndarray:
    """Apply Frigate 0.18's LBF eye-alignment transform to a BGR face crop."""
    cv2 = _opencv()
    image = _require_bgr(image_bgr)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = np.asarray([(0, 0, width, height)])
    success, landmarks = landmark_detector.fit(gray, rect)
    if not success or landmarks is None or len(landmarks) == 0:
        raise ValueError("Frigate-compatible landmark alignment failed")
    points = np.asarray(landmarks[0][0])
    if points.ndim != 2 or points.shape[0] < 48 or points.shape[1] != 2:
        raise ValueError("Landmark model did not return the expected 68 points")

    left_center = points[42:48].mean(axis=0).astype(int)
    right_center = points[36:42].mean(axis=0).astype(int)
    delta_y = right_center[1] - left_center[1]
    delta_x = right_center[0] - left_center[0]
    eye_distance = float(np.sqrt(delta_x**2 + delta_y**2))
    if eye_distance == 0:
        raise ValueError("Landmark eye centers have zero distance")
    angle = np.degrees(np.arctan2(delta_y, delta_x)) - 180
    scale = ((0.65 - 0.35) * width) / eye_distance
    eyes_center = (
        int((left_center[0] + right_center[0]) // 2),
        int((left_center[1] + right_center[1]) // 2),
    )
    transform = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    transform[0, 2] += width * 0.5 - eyes_center[0]
    transform[1, 2] += height * 0.35 - eyes_center[1]
    return cv2.warpAffine(
        image, transform, (width, height), flags=cv2.INTER_CUBIC
    )


def laplacian_variance_bgr(image_bgr: np.ndarray) -> float:
    """Compute the blur metric Frigate applies to the unaligned face crop."""
    cv2 = _opencv()
    image = _require_bgr(image_bgr)
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _require_bgr(image: np.ndarray) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("image must be a non-empty uint8 BGR image")
    return image


def _opencv():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for Frigate registration compatibility operations"
        ) from error
    return cv2
