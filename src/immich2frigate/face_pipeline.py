"""Strict Frigate 0.18 face preparation and candidate ranking."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from .frigate018 import PIPELINE_ID, arcface_preprocess, build_class_mean
from .frigate_registration import align_face_bgr, laplacian_variance_bgr, prepare_registered_upload
from .immich_client import PersonRecord
from .selection import VectorFace, _identity_safe_mask, _row_normalize, _validated_matrix
from .training_quality import MIN_COLOR_SPREAD, MIN_MEAN_LUMA, MAX_MEAN_LUMA
from .upload_image import prepare_candidate_upload

PROFILE_ID = "frigate-0.18.0-strict-face-v1"
MIN_STORED_SIDE = 80
MIN_SHARPNESS = 250.0
MIN_FIQA = 0.30
MAX_POSE_DEGREES = 15.0
MIN_FACE_IOU = 0.5
MIN_CROSS_PERSON_MARGIN = 0.10
_MODEL_CACHE = Path("/models/frigate/facedet")
_QUALITY_MODELS = Path("/models/quality")
_PINNED_HASHES = {
    "arcface": "ec639a0429b4819130d1405a2d3b38beaa4cc4a6c5bd9cf48b94fdf65461de83",
    "yunet": "321aa5a6afabf7ecc46a3d06bfab2b579dc96eb5c3be7edd365fa04502ad9294",
    "lbf": "70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b",
    "pose": "1e902872868e483bd0e4f8f4a8ff2a4d61c2ccbca9dadf748e5479b5cc86a9e9",
    "ediffiqa": "9426c899cc0f01665240cb7d9e7f98e18e24e456c178326c771a43da289bfc6a",
}


@dataclass(frozen=True, slots=True)
class FaceProfile:
    profile_id: str
    model_hashes: Mapping[str, str]
    opencv_version: str
    onnxruntime_version: str

    def metadata(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "frigate_pipeline": PIPELINE_ID,
            "opencv_version": self.opencv_version,
            "onnxruntime_version": self.onnxruntime_version,
            "model_sha256": dict(sorted(self.model_hashes.items())),
            "thresholds": {
                "stored_side_px": MIN_STORED_SIDE,
                "sharpness": MIN_SHARPNESS,
                "color_spread": MIN_COLOR_SPREAD,
                "mean_luma": [MIN_MEAN_LUMA, MAX_MEAN_LUMA],
                "ediffiqa": MIN_FIQA,
                "pose_degrees": MAX_POSE_DEGREES,
                "target_box_iou": MIN_FACE_IOU,
                "cross_person_margin": MIN_CROSS_PERSON_MARGIN,
                "duplicate_cosine": 0.98,
            },
        }


@dataclass(frozen=True, slots=True)
class FaceMetrics:
    width: int
    height: int
    sharpness: float
    color_spread: float
    mean_luma: float
    ediffiqa: float
    pitch: float
    yaw: float
    roll: float
    target_iou: float


@dataclass(frozen=True, slots=True)
class PreparedFace:
    stored_webp: bytes = field(repr=False)
    embedding: np.ndarray = field(repr=False)
    metrics: FaceMetrics
    eligible: bool
    rejected_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RankedFace:
    candidate: VectorFace = field(repr=False)
    prepared: PreparedFace = field(repr=False)
    score: float


@dataclass(frozen=True, slots=True)
class PreparedCandidate:
    source: VectorFace = field(repr=False)
    upload_bytes: bytes = field(repr=False)
    face: PreparedFace = field(repr=False)
    identity_safe: bool = False


class FacePipeline:
    """One pinned profile; model files are external, read-only inputs."""

    def __init__(self, detector, landmark_detector, arcface_session, pose_session, fiqa_session, profile: FaceProfile):
        self.detector = detector
        self.landmark_detector = landmark_detector
        self.arcface_session = arcface_session
        self.pose_session = pose_session
        self.fiqa_session = fiqa_session
        self.profile = profile

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> FacePipeline:
        env = os.environ if env is None else env
        try:
            import cv2
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("OpenCV and onnxruntime are required by the strict face profile") from error
        if cv2.__version__.split(".")[:2] != ["4", "11"]:
            raise RuntimeError(f"strict face profile requires OpenCV 4.11.x (got {cv2.__version__})")
        if ort.__version__ != "1.24.4":
            raise RuntimeError(f"strict face profile requires onnxruntime 1.24.4 (got {ort.__version__})")

        cache = Path(env.get("IMMICH2FRIGATE_MODEL_CACHE", str(_MODEL_CACHE)))
        paths = {
            "arcface": Path(env.get("IMMICH2FRIGATE_ARCFACE_MODEL", str(cache / "arcface.onnx"))),
            "yunet": Path(env.get("IMMICH2FRIGATE_YUNET_MODEL", str(cache / "facedet.onnx"))),
            "lbf": Path(env.get("IMMICH2FRIGATE_LBF_MODEL", str(cache / "landmarkdet.yaml"))),
            "pose": Path(env.get("IMMICH2FRIGATE_POSE_MODEL", str(_QUALITY_MODELS / "mobilenetv2.onnx"))),
            "ediffiqa": Path(env.get("IMMICH2FRIGATE_EDIFFIQA_MODEL", str(_QUALITY_MODELS / "ediffiqa_tiny_jun2024.onnx"))),
        }
        hashes: dict[str, str] = {}
        for name, path in paths.items():
            expected = _PINNED_HASHES[name]
            actual = _sha256(path)
            if actual != expected.lower():
                raise ValueError(f"{name} model SHA-256 does not match the strict profile")
            hashes[name] = actual

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        arcface = ort.InferenceSession(str(paths["arcface"]), sess_options=options, providers=["CPUExecutionProvider"])
        pose = ort.InferenceSession(str(paths["pose"]), sess_options=options, providers=["CPUExecutionProvider"])
        fiqa = ort.InferenceSession(str(paths["ediffiqa"]), sess_options=options, providers=["CPUExecutionProvider"])
        detector = cv2.FaceDetectorYN_create(str(paths["yunet"]), "", (320, 320), 0.5, 0.3, 5000)
        landmark = cv2.face.createFacemarkLBF()
        landmark.loadModel(str(paths["lbf"]))
        profile = FaceProfile(PROFILE_ID, hashes, cv2.__version__, ort.__version__)
        return cls(detector, landmark, arcface, pose, fiqa, profile)

    @staticmethod
    def validate_readback(stored_webp: bytes) -> np.ndarray:
        """Decode exactly what Frigate stores; reject non-WebP or corrupt bytes."""
        if not isinstance(stored_webp, bytes) or len(stored_webp) < 12:
            raise ValueError("stored face must be non-empty WebP bytes")
        if stored_webp[:4] != b"RIFF" or stored_webp[8:12] != b"WEBP":
            raise ValueError("stored face is not WebP")
        cv2 = _opencv()
        image = cv2.imdecode(np.frombuffer(stored_webp, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("stored WebP could not be decoded as a color image")
        return np.ascontiguousarray(image)

    def evaluate(
        self,
        upload_bytes: bytes,
        expected_box: tuple[float, float, float, float],
    ) -> PreparedFace:
        cv2 = _opencv()
        decoded = cv2.imdecode(np.frombuffer(upload_bytes, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("upload bytes could not be decoded as a color image")
        if len(expected_box) != 4 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in expected_box
        ):
            raise ValueError("expected_box must contain four finite coordinates")
        if not (0 <= expected_box[0] < expected_box[2] <= decoded.shape[1] and 0 <= expected_box[1] < expected_box[3] <= decoded.shape[0]):
            raise ValueError("expected_box must fit inside the uploaded image")
        registered = self.simulate_upload(upload_bytes)
        iou = _box_iou(registered.box, expected_box)
        if iou < MIN_FACE_IOU:
            raise ValueError("Frigate's selected face does not match the Immich target box")

        return self.evaluate_crop(registered.stored_webp, target_iou=iou)

    def simulate_upload(self, upload_bytes: bytes):
        registered = prepare_registered_upload(upload_bytes, self.detector)
        if registered is None:
            raise ValueError("Frigate YuNet found no registrable face")
        if registered.accepted_detection_count != 1:
            raise ValueError("multiple Frigate faces make the upload ambiguous")
        return registered

    def evaluate_crop(self, stored_webp: bytes, *, target_iou: float = 1.0) -> PreparedFace:
        """Evaluate the image bytes Frigate read back from its face store."""
        if not math.isfinite(target_iou) or not 0 <= target_iou <= 1:
            raise ValueError("target_iou must be between zero and one")
        if target_iou < MIN_FACE_IOU:
            raise ValueError("Frigate's selected face does not match the Immich target box")
        stored = self.validate_readback(stored_webp)
        height, width = stored.shape[:2]
        aligned = align_face_bgr(stored, self.landmark_detector)
        lap = laplacian_variance_bgr(stored)
        pixels = stored.astype(np.float32, copy=False)
        spread = float(np.mean(np.max(pixels, 2) - np.min(pixels, 2), dtype=np.float64))
        luma = float(np.mean(pixels[:, :, 0] * .114 + pixels[:, :, 1] * .587 + pixels[:, :, 2] * .299))
        fiqa = _fiqa_score(self.fiqa_session, aligned)
        pose = _head_pose(self.pose_session, stored)
        metrics = FaceMetrics(width, height, lap, spread, luma, fiqa, *pose, target_iou)

        rejected = []
        if min(width, height) < MIN_STORED_SIDE:
            rejected.append("stored_face_too_small")
        if lap < MIN_SHARPNESS:
            rejected.append("blur")
        if spread < MIN_COLOR_SPREAD:
            rejected.append("low_color_spread")
        if not MIN_MEAN_LUMA <= luma <= MAX_MEAN_LUMA:
            rejected.append("exposure")
        if fiqa < MIN_FIQA:
            rejected.append("ediffiqa")
        if any(abs(angle) > MAX_POSE_DEGREES for angle in pose):
            rejected.append("pose")
        embedding = _arcface_embedding(self.arcface_session, aligned)
        return PreparedFace(stored_webp, embedding, metrics, not rejected, tuple(rejected))


def scan_person(
    immich,
    vectors,
    person: PersonRecord,
    pipeline: FacePipeline,
    *,
    years: int = 100,
) -> list[PreparedCandidate]:
    """Read, prepare, and score the complete available person candidate set."""
    embedded = vectors.candidates_for_person(person, years=years, include_face_ids=())
    loaded_count = len(embedded)
    from .enrollment import _one_face_per_asset

    embedded, _ = _one_face_per_asset(embedded, excluded_assets=set())
    matrix = _row_normalize(_validated_matrix(person, embedded, attr="face_embedding"))
    safe = _identity_safe_mask(matrix, minimum_keep=5)
    embedded = [item for index, item in enumerate(embedded) if safe[index]]
    identity_safe_count = len(embedded)
    if len(embedded) < 5:
        raise ValueError(f"fewer than five Immich identity-safe candidates remain for {person.name!r}")
    prepared: list[PreparedCandidate] = []
    rejected: dict[str, int] = {}
    invalid_crop = 0
    eligible = 0
    print(json.dumps({"preparation": "scan_progress", "processed": 0, "loaded": loaded_count, "identity_safe": identity_safe_count}, separators=(",", ":")), flush=True)
    for processed, source in enumerate(embedded, 1):
        try:
            upload = prepare_candidate_upload(source.source, immich.preview(source.source.asset_id))
            face = pipeline.evaluate(upload.encoded, upload.face_box)
        except Exception as error:
            invalid_crop += 1
            reason = "invalid_crop" if isinstance(error, ValueError) else "preview_or_pipeline_error"
            rejected[reason] = rejected.get(reason, 0) + 1
        else:
            if face.eligible:
                prepared.append(PreparedCandidate(source, upload.encoded, face, identity_safe=True))
                eligible += 1
            else:
                for reason in face.rejected_by or ("quality",):
                    rejected[reason] = rejected.get(reason, 0) + 1
        if processed % 100 == 0:
            print(json.dumps({"preparation": "scan_progress", "processed": processed, "loaded": loaded_count, "identity_safe": identity_safe_count}, separators=(",", ":")), flush=True)
    print(json.dumps({
        "preparation": "scan_complete",
        "loaded": loaded_count,
        "identity_safe": identity_safe_count,
        "eligible": eligible,
        "invalid_crop": invalid_crop,
        "rejected_by": rejected,
    }, sort_keys=True, separators=(",", ":")), flush=True)
    return prepared


def select_foundation(
    person: PersonRecord,
    candidates: Sequence[PreparedCandidate],
    *,
    teacher: Callable[[PreparedCandidate], bool] | None = None,
    references_by_person: Mapping[str, Sequence[PreparedCandidate]] | None = None,
) -> list[PreparedCandidate]:
    """Rank the strict pool and refill rejected teacher candidates until five remain."""
    pool = [candidate for candidate in candidates if candidate.identity_safe and candidate.face.eligible]
    if references_by_person is not None:
        pool = [
            candidate for candidate in pool
            if (margin := cross_person_margin(candidate, references_by_person)) is not None
            and margin >= MIN_CROSS_PERSON_MARGIN
        ]
    while len(pool) >= 5:
        by_face_id = {item.source.source.face_id: item.face for item in pool}
        selected = select_prepared_candidates(
            person,
            [item.source for item in pool],
            [item.source for item in pool],
            by_face_id,
        )
        chosen = [next(item for item in pool if item.source.source.face_id == row.candidate.source.face_id) for row in selected[:5]]
        rejected = next((item for item in chosen if teacher is not None and not teacher(item)), None)
        if rejected is None:
            if references_by_person is not None:
                margin = _prototype_margin(chosen, references_by_person)
                if margin is None or margin < MIN_CROSS_PERSON_MARGIN:
                    weakest = min(
                        chosen,
                        key=lambda item: cross_person_margin(item, references_by_person) or -1.0,
                    )
                    pool.remove(weakest)
                    continue
            return chosen
        pool.remove(rejected)
    raise ValueError(f"fewer than five strict, identity-safe candidates remain for {person.name!r}")


def select_prepared_candidates(
    person: PersonRecord,
    foundation_candidates: list[VectorFace],
    expansion_candidates: list[VectorFace],
    prepared_by_face_id: Mapping[str, PreparedFace],
) -> list[RankedFace]:
    """Apply Immich identity safety to the full pool, then rank on Frigate vectors."""
    def eligible(items: list[VectorFace]) -> list[VectorFace]:
        seen_assets: set[str] = set()
        seen_checksums: set[str] = set()
        result = []
        order = [
            item for item in items
            if (prepared := prepared_by_face_id.get(item.source.face_id or "")) is not None
            and prepared.eligible
        ]
        order.sort(key=lambda item: (
            -prepared_by_face_id[item.source.face_id or ""].metrics.ediffiqa,
            -prepared_by_face_id[item.source.face_id or ""].metrics.sharpness,
            item.source.taken_at,
            item.source.asset_id,
            item.source.face_id or "",
        ))
        for item in order:
            prepared = prepared_by_face_id.get(item.source.face_id or "")
            assert prepared is not None
            checksum = item.source.checksum.casefold()
            if item.source.asset_id in seen_assets or checksum in seen_checksums:
                continue
            embedding = _unit(prepared.embedding)
            if any(float(np.dot(embedding, _unit(prepared_by_face_id[prior.source.face_id or ""].embedding))) >= 0.98 for prior in result):
                continue
            seen_assets.add(item.source.asset_id)
            seen_checksums.add(checksum)
            result.append(VectorFace(item.source, item.face_embedding, None))
        return result

    expansion = eligible(expansion_candidates)
    foundation_ids = {item.source.face_id for item in eligible(foundation_candidates)}
    foundation = [item for item in expansion if item.source.face_id in foundation_ids]
    if len(foundation) < 5:
        raise ValueError("at least five strict foundation candidates are required")
    _validated_matrix(person, expansion, attr="face_embedding")
    safe_expansion = expansion
    embeddings = np.stack([_unit(prepared_by_face_id[item.source.face_id or ""].embedding) for item in safe_expansion])
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    center = _unit(build_class_mean([embedding for embedding in embeddings]))
    if len(safe_expansion) > 1:
        np.fill_diagonal(similarity, -np.inf)
        neighbor_count = min(5, len(safe_expansion) - 1)
        density = np.mean(np.partition(similarity, -neighbor_count, axis=1)[:, -neighbor_count:], axis=1)
    else:
        density = np.ones(1, dtype=np.float32)
    ranked = []
    for index, candidate in enumerate(safe_expansion):
        prepared = prepared_by_face_id[candidate.source.face_id or ""]
        q = prepared.metrics.ediffiqa
        score = float(1.0 - np.dot(embeddings[index], center) + 1.0 - density[index] - 0.05 * q)
        ranked.append(RankedFace(candidate, prepared, score))
    return sorted(ranked, key=lambda item: (item.score, item.candidate.source.taken_at, item.candidate.source.asset_id))


def _arcface_embedding(session, image_bgr: np.ndarray) -> np.ndarray:
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    vector = np.asarray(session.run([output_meta.name], {input_meta.name: arcface_preprocess(image_bgr)})[0], np.float32).reshape(-1)
    if vector.size != 512 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("Frigate ArcFace returned an invalid 512-dimensional embedding")
    return np.ascontiguousarray(vector)


def _fiqa_score(session, image_bgr: np.ndarray) -> float:
    cv2 = _opencv()
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (112, 112)).astype(np.float32)
    tensor = (((image / 255.0) - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    scores = np.asarray(session.run([output_meta.name], {input_meta.name: tensor})[0], np.float32).reshape(-1)
    if scores.size != 1 or not np.isfinite(scores[0]):
        raise ValueError("eDifFIQA returned an invalid score")
    return float(scores[0])


def _head_pose(session, image_bgr: np.ndarray) -> tuple[float, float, float]:
    cv2 = _opencv()
    meta = session.get_inputs()[0]
    shape = meta.shape
    if len(shape) != 4 or not all(isinstance(value, int) and value > 0 for value in shape[2:]):
        raise ValueError("head-pose model must have static BCHW spatial dimensions")
    height, width = shape[2:]
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height)).astype(np.float32) / 255.0
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    tensor = ((image - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    outputs = session.run(None, {meta.name: tensor})
    rotation = np.asarray(outputs[0], np.float64)
    if rotation.shape != (1, 3, 3) or not np.isfinite(rotation).all():
        raise ValueError("head-pose model returned an invalid rotation matrix")
    matrix = rotation[0]
    sy = math.sqrt(float(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    if sy < 1e-6:
        pitch = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = math.atan2(-matrix[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(-matrix[2, 0], sy)
        roll = math.atan2(matrix[1, 0], matrix[0, 0])
    return tuple(float(np.degrees(value)) for value in (pitch, yaw, roll))


def _box_iou(a, b) -> float:
    if len(b) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in b):
        raise ValueError("expected_box must contain four finite coordinates")
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if not (bx1 < bx2 and by1 < by2):
        raise ValueError("expected_box must have positive area")
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return float(intersection / union) if union > 0 else 0.0


def cross_person_margin(
    candidate: PreparedCandidate,
    references_by_person: Mapping[str, Sequence[PreparedCandidate]],
) -> float | None:
    """Return the best same-person minus other-person raw cosine score.

    References from the same asset, duplicate checksum, or calendar day are
    excluded. A missing comparison set is unknown and must not be treated as a
    passing margin.
    """
    source = candidate.source.source
    capture_day = source.taken_at[:10]
    query = _unit(candidate.face.embedding)
    scores: dict[str, list[np.ndarray]] = {}
    for person_id, references in references_by_person.items():
        rows = scores.setdefault(person_id, [])
        for reference in references:
            ref_source = reference.source.source
            if ref_source.person_id != person_id:
                raise ValueError("reference person ID does not match its candidate")
            if (
                ref_source.asset_id == source.asset_id
                or ref_source.checksum.casefold() == source.checksum.casefold()
                or ref_source.taken_at[:10] == capture_day
            ):
                continue
            rows.append(reference.face.embedding)
    own = scores.get(source.person_id, [])
    other = [rows for person_id, rows in scores.items() if person_id != source.person_id and rows]
    if not own or not other:
        return None
    own_score = float(np.dot(query, _unit(build_class_mean(own))))
    other_score = max(float(np.dot(query, _unit(build_class_mean(rows)))) for rows in other)
    return own_score - other_score


def _prototype_margin(
    selected: Sequence[PreparedCandidate],
    references_by_person: Mapping[str, Sequence[PreparedCandidate]],
) -> float | None:
    if len(selected) != 5:
        return None
    sources = [item.source.source for item in selected]
    person_id = sources[0].person_id
    if any(source.person_id != person_id for source in sources):
        return None
    center = _unit(build_class_mean([item.face.embedding for item in selected]))
    scores: dict[str, list[np.ndarray]] = {}
    for reference_person, references in references_by_person.items():
        rows = scores.setdefault(reference_person, [])
        for reference in references:
            source = reference.source.source
            if source.person_id != reference_person:
                raise ValueError("reference person ID does not match its candidate")
            if any(
                source.asset_id == chosen.asset_id
                or source.checksum.casefold() == chosen.checksum.casefold()
                or source.taken_at[:10] == chosen.taken_at[:10]
                for chosen in sources
            ):
                continue
            rows.append(reference.face.embedding)
    own = scores.get(person_id, [])
    other = [rows for key, rows in scores.items() if key != person_id and rows]
    if not own or not other:
        return None
    own_score = float(np.dot(center, _unit(build_class_mean(own))))
    other_score = max(float(np.dot(center, _unit(build_class_mean(rows)))) for rows in other)
    return own_score - other_score


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not math.isfinite(norm) or norm == 0:
        raise ValueError("Frigate embedding must be finite and non-zero")
    return vector / norm


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"model file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as model:
        for chunk in iter(lambda: model.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opencv():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for the strict face profile") from error
    return cv2
