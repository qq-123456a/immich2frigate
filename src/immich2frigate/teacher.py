"""Read-only Immich face confirmation using its own detector and face index."""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urlsplit

from .immich_client import ImmichReadOnlyClient
from .immich_vectors import ImmichVectorStoreError
from .settings import ImmichDatabaseSettings, ImmichSettings

_PREDICT_ENTRIES = {
    "facial-recognition": {
        "detection": {"modelName": "buffalo_l", "options": {"minScore": 0.7}},
        "recognition": {"modelName": "buffalo_l"},
    }
}
_MAX_IMAGE_BYTES = 16 * 1024 * 1024
_PERSON_CACHE_SECONDS = 60
# ponytail: cap nearest rows at 2000; use per-person indexed searches if that hides real support.
_PADDING_PROFILE = {"id": "neutral-border-v1", "fraction": 0.5, "value": 127, "format": "png"}


@dataclass(frozen=True, slots=True)
class TeacherResult:
    status: str
    person_id: str | None = None
    name: str | None = None
    distance: float | None = None
    support: int = 0
    reason: str | None = None

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"


class ImmichTeacher:
    """Confirm a crop against named Immich people without uploading an asset."""

    def __init__(self, immich: ImmichSettings, database: ImmichDatabaseSettings, *, ml_url: str,
                 session=None, ml_session=None, connect=None):
        parts = urlsplit(ml_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
                or parts.password or parts.path not in {"", "/"} or parts.query or parts.fragment):
            raise ValueError("IMMICH_ML_URL must be an origin URL")
        self.immich = immich
        self.database = database
        self.ml_url = f"{parts.scheme}://{parts.netloc}"
        if session is None or ml_session is None:
            import requests
            session = session or requests.Session()
            ml_session = ml_session or requests.Session()
        session.max_redirects = ml_session.max_redirects = 0
        session.headers.update({"x-api-key": immich.immich_api_key, "Accept": "application/json"})
        self._session = session
        self._ml_session = ml_session
        self._connect = connect
        self._people: dict[str, str] | None = None
        self._people_loaded_at = 0.0

    @classmethod
    def from_env(cls, environ=None, *, session=None, connect=None) -> "ImmichTeacher":
        env = os.environ if environ is None else environ
        immich = ImmichSettings.from_env(env)
        database = ImmichDatabaseSettings.from_env(env)
        ml_url = env.get("IMMICH_ML_URL", "http://immich-machine-learning:3003")
        return cls(immich, database, ml_url=ml_url, session=session, connect=connect)

    def profile(self) -> dict[str, object]:
        """Fail closed unless the live Immich config matches the approved profile."""
        base = self.immich.immich_url.removesuffix("/api")
        response = self._request("GET", f"{base}/api/admin/config", json_response=True)
        try:
            config = response["machineLearning"]["facialRecognition"]
            profile = {
                "enabled": config["enabled"], "modelName": config["modelName"],
                "minScore": float(config["minScore"]),
                "maxDistance": float(config["maxDistance"]), "minFaces": int(config["minFaces"]),
                "detectionModel": _PREDICT_ENTRIES["facial-recognition"]["detection"]["modelName"],
                "detectionMinScore": _PREDICT_ENTRIES["facial-recognition"]["detection"]["options"]["minScore"],
                "inputPadding": _PADDING_PROFILE,
            }
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("Immich face-recognition config was unavailable") from None
        if profile != {
            "enabled": True, "modelName": "buffalo_l", "minScore": 0.7,
            "maxDistance": 0.5, "minFaces": 3,
            "detectionModel": "buffalo_l", "detectionMinScore": 0.7,
            "inputPadding": _PADDING_PROFILE,
        }:
            raise RuntimeError("Immich face-recognition profile does not match the required settings")
        return profile

    def confirm(self, image: bytes, exclude_asset_id: str | None = None,
                exclude_checksum: str | None = None) -> TeacherResult:
        if not isinstance(image, bytes) or not image or len(image) > _MAX_IMAGE_BYTES:
            return TeacherResult("unconfirmed", reason="invalid_image")
        try:
            padded, source_box, padded_size = _pad_prediction_image(image)
            self.profile()
            predictions = self._request(
                "POST", f"{self.ml_url}/predict", multipart={
                    "entries": (None, json.dumps(_PREDICT_ENTRIES, separators=(",", ":")), "application/json"),
                    "image": ("crop.png", padded, "image/png"),
                }, json_response=True, machine_learning=True,
            )
            if (predictions.get("imageWidth"), predictions.get("imageHeight")) != padded_size:
                return TeacherResult("unconfirmed", reason="invalid_prediction_frame")
            faces = predictions.get("facial-recognition")
            if not isinstance(faces, list) or len(faces) != 1:
                return TeacherResult("unconfirmed", reason="no_face_or_ambiguous")
            face = faces[0]
            if not isinstance(face, dict) or not _valid_embedding(face.get("embedding")):
                return TeacherResult("unconfirmed", reason="invalid_embedding")
            if (not isinstance(face.get("score"), (int, float))
                    or not math.isfinite(face["score"]) or face["score"] < 0.7):
                return TeacherResult("unconfirmed", reason="detection_below_threshold")
            if not _overlaps_source(face.get("boundingBox"), source_box, padded_size):
                return TeacherResult("unconfirmed", reason="face_outside_source_crop")
            rows = self._match(face["embedding"], exclude_asset_id, exclude_checksum)
        except ValueError:
            return TeacherResult("unconfirmed", reason="invalid_image")
        except Exception as error:
            return TeacherResult("unconfirmed", reason=f"service_error:{type(error).__name__}")
        if not rows:
            return TeacherResult("unconfirmed", reason="insufficient_support")
        try:
            names = self._person_names()
        except Exception as error:
            return TeacherResult("unconfirmed", reason=f"service_error:{type(error).__name__}")
        grouped: dict[str, list[float]] = defaultdict(list)
        for person_id, distance in rows:
            distance = float(distance)
            if distance <= 0.5:
                grouped[person_id].append(distance)
        ranked = sorted(
            ((sum(sorted(ds)[:3]) / 3, person_id) for person_id, ds in grouped.items() if len(ds) >= 3),
            key=lambda item: (item[0], item[1]),
        )
        if not ranked:
            return TeacherResult("unconfirmed", reason="insufficient_support")
        distance, person_id = ranked[0]
        if person_id not in names:
            return TeacherResult("unconfirmed", distance=distance, support=len(grouped[person_id]), reason="unnamed_person")
        try:
            second = self._nearest_other(face["embedding"], person_id,
                                         exclude_asset_id, exclude_checksum)
        except Exception as error:
            return TeacherResult("unconfirmed", distance=distance, support=len(grouped[person_id]),
                                 reason=f"service_error:{type(error).__name__}")
        if second is None:
            return TeacherResult("unconfirmed", distance=distance, support=len(grouped[person_id]),
                                 reason="no_competing_reference")
        if second - distance < 0.05:
            return TeacherResult("unconfirmed", distance=distance, support=len(grouped[person_id]), reason="ambiguous_people")
        return TeacherResult("confirmed", person_id, names[person_id], distance,
                             len(grouped[person_id]))

    def close(self) -> None:
        self._session.close()
        self._ml_session.close()

    def _person_names(self) -> dict[str, str]:
        now = time.monotonic()
        if self._people is None or now - self._people_loaded_at >= _PERSON_CACHE_SECONDS:
            with ImmichReadOnlyClient(self.immich) as client:
                self._people = {person.person_id: person.name for person in client.people()}
            self._people_loaded_at = now
        return self._people

    def _match(self, embedding: str, exclude_asset_id: str | None,
               exclude_checksum: str | None) -> list[tuple[str, float]]:
        query = """
            WITH nearby AS MATERIALIZED (
                SELECT af."personGroupId"::text AS person_id,
                       af."assetId"::text AS asset_id,
                       encode(a.checksum, 'hex') AS checksum,
                       fs.embedding <=> %s::vector AS distance
                FROM face_search AS fs
                JOIN asset_face AS af ON af.id = fs."faceId"
                JOIN asset AS a ON a.id = af."assetId"
                WHERE af."deletedAt" IS NULL AND af."isVisible" IS TRUE
                  AND a."deletedAt" IS NULL AND a.type = 'IMAGE'
                  AND a.checksum IS NOT NULL
                  AND af."personGroupId" IS NOT NULL
                  AND (%s::uuid IS NULL OR af."assetId" <> %s::uuid)
                  AND (%s::text IS NULL OR encode(a.checksum, 'hex') <> lower(%s::text))
                ORDER BY fs.embedding <=> %s::vector
                LIMIT 2000
            ), ranked AS (
                SELECT person_id, asset_id, distance,
                       ROW_NUMBER() OVER (
                           PARTITION BY person_id, checksum ORDER BY distance
                       ) AS asset_rank
                FROM nearby
            )
            SELECT person_id, distance FROM ranked
            WHERE asset_rank = 1
            ORDER BY distance
        """
        try:
            if self._connect is None:
                import psycopg

                connect = psycopg.connect
            else:
                connect = self._connect
            with connect(self.database.database_url, connect_timeout=10, autocommit=True,
                         options="-c default_transaction_read_only=on") as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, (embedding, exclude_asset_id, exclude_asset_id,
                                           exclude_checksum, exclude_checksum, embedding))
                    rows = cursor.fetchall()
        except Exception as error:
            raise ImmichVectorStoreError(f"Immich match read failed ({type(error).__name__})") from None
        if not isinstance(rows, list) or len(rows) > 2000:
            raise ImmichVectorStoreError("Immich match query returned an unexpected result")
        return [(row[0], float(row[1])) for row in rows
                if isinstance(row, (tuple, list)) and len(row) == 2
                and isinstance(row[0], str) and isinstance(row[1], (int, float))]

    def _nearest_other(self, embedding: str, winner_person_id: str,
                       exclude_asset_id: str | None,
                       exclude_checksum: str | None) -> float | None:
        """Exact nearest other-person distance, independent of the bounded global KNN pool."""
        query = """
            SELECT MIN(fs.embedding <=> %s::vector) AS distance
            FROM face_search AS fs
            JOIN asset_face AS af ON af.id = fs."faceId"
            JOIN asset AS a ON a.id = af."assetId"
            WHERE af."deletedAt" IS NULL AND af."isVisible" IS TRUE
              AND a."deletedAt" IS NULL AND a.type = 'IMAGE'
              AND a.checksum IS NOT NULL
              AND af."personGroupId" IS NOT NULL
              AND af."personGroupId" <> %s::uuid
              AND (%s::uuid IS NULL OR af."assetId" <> %s::uuid)
              AND (%s::text IS NULL OR encode(a.checksum, 'hex') <> lower(%s::text))
        """
        try:
            if self._connect is None:
                import psycopg

                connect = psycopg.connect
            else:
                connect = self._connect
            with connect(self.database.database_url, connect_timeout=10, autocommit=True,
                         options="-c default_transaction_read_only=on") as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, (embedding, winner_person_id, exclude_asset_id, exclude_asset_id,
                                           exclude_checksum, exclude_checksum))
                    row = cursor.fetchone()
        except Exception as error:
            raise ImmichVectorStoreError(f"Immich competitor read failed ({type(error).__name__})") from None
        if not isinstance(row, (tuple, list)) or len(row) != 1:
            raise ImmichVectorStoreError("Immich competitor query returned an unexpected result")
        distance = row[0]
        if distance is None:
            return None
        if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
            raise ImmichVectorStoreError("Immich competitor query returned an invalid distance")
        return float(distance)

    def _request(self, method: str, url: str, *, json_response: bool = False, multipart=None,
                 machine_learning: bool = False):
        try:
            session = self._ml_session if machine_learning else self._session
            response = session.request(method, url, timeout=(10, 60), allow_redirects=False,
                                             files=multipart, stream=True)
            if 300 <= response.status_code < 400:
                raise RuntimeError("Immich redirected a credentialed request")
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"Immich returned HTTP {response.status_code}")
            body = bytearray()
            for chunk in response.iter_content(64 * 1024):
                body.extend(chunk)
                if len(body) > 2 * 1024 * 1024:
                    raise RuntimeError("Immich response exceeded the size limit")
        except Exception as error:
            raise RuntimeError(f"Immich request failed ({type(error).__name__})") from None
        finally:
            if "response" in locals():
                response.close()
        if not json_response:
            return bytes(body)
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RuntimeError("Immich returned invalid JSON") from None


def _pad_prediction_image(image: bytes) -> tuple[bytes, tuple[int, int, int, int], tuple[int, int]]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("OpenCV is required for Immich teacher padding") from error
    decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError("image could not be decoded")
    height, width = decoded.shape[:2]
    border_x, border_y = max(1, round(width * 0.5)), max(1, round(height * 0.5))
    padded = cv2.copyMakeBorder(decoded, border_y, border_y, border_x, border_x,
                                cv2.BORDER_CONSTANT, value=(127, 127, 127))
    ok, encoded = cv2.imencode(".png", padded)
    if not ok:
        raise RuntimeError("padded image could not be encoded")
    return (encoded.tobytes(), (border_x, border_y, border_x + width, border_y + height),
            (width + border_x * 2, height + border_y * 2))


def _valid_embedding(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 8192:
        return False
    text = value.strip()
    if len(text) < 4 or text[0] != "[" or text[-1] != "]":
        return False
    try:
        values = [float(part.strip()) for part in text[1:-1].split(",")]
    except (OverflowError, ValueError):
        return False
    return len(values) == 512 and all(math.isfinite(item) for item in values) and any(values)


def _overlaps_source(box: object, source: tuple[int, int, int, int], frame: tuple[int, int]) -> bool:
    if not isinstance(box, dict):
        return False
    try:
        x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if (not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            or not (0 <= x1 < x2 <= frame[0] and 0 <= y1 < y2 <= frame[1])):
        return False
    sx1, sy1, sx2, sy2 = source
    center_inside = sx1 <= (x1 + x2) / 2 <= sx2 and sy1 <= (y1 + y2) / 2 <= sy2
    intersection = max(0, min(x2, sx2) - max(x1, sx1)) * max(0, min(y2, sy2) - max(y1, sy1))
    area = (x2 - x1) * (y2 - y1)
    return center_inside and area > 0 and intersection / area >= 0.5
