"""Read-only access to Immich's persisted face and smart-search vectors.

This adapter intentionally reads Immich's existing database products instead
of reproducing its face-recognition or CLIP inference. The database account
must be read-only. No SQL write statement exists in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
from uuid import UUID

import numpy as np

from .immich_client import FaceCandidate, PersonRecord
from .selection import MAX_SELECTION_CANDIDATES, VectorFace

_MAX_SQL_CANDIDATES = min(500, MAX_SELECTION_CANDIDATES)
_MAX_RECENT_SQL_CANDIDATES = 200


class ImmichVectorStoreError(RuntimeError):
    """A bounded error that does not include credentials or vector contents."""


@dataclass(frozen=True, slots=True)
class ImmichVectorStore:
    """Fetch persisted vectors for one already-known Immich person."""

    database_url: str = field(repr=False)
    connect_timeout: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.database_url, str) or not self.database_url.strip():
            raise ValueError("IMMICH_DATABASE_URL is required")
        if isinstance(self.connect_timeout, bool) or not isinstance(self.connect_timeout, int):
            raise ValueError("connect_timeout must be an integer")
        if not 1 <= self.connect_timeout <= 60:
            raise ValueError("connect_timeout must be between 1 and 60 seconds")

    def vectors_for_person(
        self,
        person: PersonRecord,
        candidates: Iterable[FaceCandidate],
    ) -> list[VectorFace]:
        """Return candidates with face embeddings and optional scene embeddings."""

        _uuid(person.person_id, "person ID")
        candidate_by_face: dict[str, FaceCandidate] = {}
        for candidate in candidates:
            if candidate.person_id != person.person_id:
                raise ValueError("candidate belongs to a different Immich person")
            if candidate.face_id is None:
                continue
            _uuid(candidate.face_id, "face ID")
            if candidate.face_id in candidate_by_face:
                raise ValueError("duplicate face ID in candidate inventory")
            candidate_by_face[candidate.face_id] = candidate
        if not candidate_by_face:
            return []

        rows = self._read_rows(person.person_id, tuple(candidate_by_face))
        results: list[VectorFace] = []
        for face_id, asset_id, face_vector, scene_vector in rows:
            candidate = candidate_by_face.get(face_id)
            if candidate is None or candidate.asset_id != asset_id:
                continue
            results.append(
                VectorFace(
                    source=candidate,
                    face_embedding=_parse_vector(face_vector, "face"),
                    scene_embedding=(
                        None if scene_vector is None else _parse_vector(scene_vector, "scene")
                    ),
                )
            )
        results.sort(key=lambda item: (item.source.taken_at, item.source.asset_id, item.source.face_id or ""))
        return results

    def candidates_for_person(
        self,
        person: PersonRecord,
        *,
        years: int = 100,
        include_face_ids: tuple[str, ...] = (),
    ) -> list[VectorFace]:
        """Read boxes and vectors in one bounded query, sampling across time.

        The Immich HTTP search API can require one `/faces` request per image.
        Reading its existing face, asset, and embedding rows avoids that N+1
        behavior and fetches previews only for the bounded candidate set.
        """
        _uuid(person.person_id, "person ID")
        if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= 100:
            raise ValueError("years must be between 1 and 100")
        for face_id in include_face_ids:
            _uuid(face_id, "face ID")
        query = """
            WITH valid AS MATERIALIZED (
                SELECT
                    af.id::text AS face_id,
                    af."assetId"::text AS asset_id,
                    COALESCE(a."fileCreatedAt", a."localDateTime") AS taken_at
                FROM asset_face AS af
                INNER JOIN asset AS a ON a.id = af."assetId"
                INNER JOIN face_search AS fs ON fs."faceId" = af.id
                WHERE af."personGroupId" = %s::uuid
                  AND af."deletedAt" IS NULL
                  AND af."isVisible" IS TRUE
                  AND a."deletedAt" IS NULL
                  AND a.type = 'IMAGE'
                  AND a.checksum IS NOT NULL
                  AND COALESCE(a."fileCreatedAt", a."localDateTime") >=
                      CURRENT_TIMESTAMP - (%s * INTERVAL '1 year')
                  AND 0 <= af."boundingBoxX1"
                  AND af."boundingBoxX1" < af."boundingBoxX2"
                  AND af."boundingBoxX2" <= af."imageWidth"
                  AND 0 <= af."boundingBoxY1"
                  AND af."boundingBoxY1" < af."boundingBoxY2"
                  AND af."boundingBoxY2" <= af."imageHeight"
            ), bucketed AS (
                SELECT valid.*, NTILE(%s) OVER (
                    ORDER BY taken_at, asset_id, face_id
                ) AS sample_bucket
                FROM valid
            ), sampled AS (
                SELECT bucketed.*, ROW_NUMBER() OVER (
                    PARTITION BY sample_bucket ORDER BY taken_at, asset_id, face_id
                ) AS sample_slot
                FROM bucketed
            ), recent AS (
                SELECT face_id, asset_id, taken_at
                FROM valid
                ORDER BY taken_at DESC, asset_id, face_id
                LIMIT %s
            ), chosen AS (
                SELECT face_id, asset_id, taken_at
                FROM sampled
                WHERE sample_slot = 1
                UNION
                SELECT face_id, asset_id, taken_at
                FROM recent
                UNION
                SELECT face_id, asset_id, taken_at
                FROM valid
                WHERE face_id = ANY(%s::text[])
            )
            SELECT
                af.id::text AS face_id,
                af."assetId"::text AS asset_id,
                COALESCE(a."fileCreatedAt", a."localDateTime")::text AS taken_at,
                encode(a.checksum, 'hex') AS checksum,
                af."boundingBoxX1" AS x1,
                af."boundingBoxY1" AS y1,
                af."boundingBoxX2" AS x2,
                af."boundingBoxY2" AS y2,
                af."imageWidth" AS frame_width,
                af."imageHeight" AS frame_height,
                fs.embedding::text AS face_embedding,
                ss.embedding::text AS scene_embedding
            FROM chosen AS c
            INNER JOIN asset_face AS af ON af.id::text = c.face_id
            INNER JOIN asset AS a ON a.id = af."assetId"
            INNER JOIN face_search AS fs ON fs."faceId" = af.id
            LEFT JOIN smart_search AS ss ON ss."assetId" = af."assetId"
            ORDER BY c.taken_at, c.asset_id, c.face_id
        """
        try:
            import psycopg

            with psycopg.connect(
                self.database_url,
                connect_timeout=self.connect_timeout,
                autocommit=True,
                options="-c default_transaction_read_only=on",
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        query,
                        (
                            person.person_id,
                            years,
                            _MAX_SQL_CANDIDATES,
                            _MAX_RECENT_SQL_CANDIDATES,
                            list(include_face_ids),
                        ),
                    )
                    rows = cursor.fetchall()
        except Exception as error:
            raise ImmichVectorStoreError(
                f"Immich candidate read failed ({type(error).__name__})"
            ) from None

        results: list[VectorFace] = []
        for row in rows:
            if not isinstance(row, tuple) or len(row) != 12:
                raise ImmichVectorStoreError("Immich candidate query returned an unexpected row")
            face_id, asset_id, taken_at, checksum, x1, y1, x2, y2, width, height, face_vector, scene_vector = row
            if not all(isinstance(value, str) and value for value in (face_id, asset_id, taken_at, checksum)):
                continue
            try:
                candidate = FaceCandidate(
                    person_id=person.person_id,
                    face_id=face_id,
                    asset_id=asset_id,
                    taken_at=taken_at,
                    checksum=checksum,
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    frame=(int(width), int(height)),
                )
                _uuid(candidate.face_id, "face ID")
                _uuid(candidate.asset_id, "asset ID")
                face_embedding = _parse_vector(face_vector, "face")
                scene_embedding = None if scene_vector is None else _parse_vector(scene_vector, "scene")
            except (TypeError, ValueError, OverflowError):
                continue
            results.append(VectorFace(candidate, face_embedding, scene_embedding))
        return results

    def _read_rows(
        self,
        person_id: str,
        face_ids: tuple[str, ...],
    ) -> list[tuple[str, str, object, object | None]]:
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Install the database extra to read Immich vectors") from error

        query = """
            SELECT
                af.id::text,
                af."assetId"::text,
                fs.embedding::text,
                ss.embedding::text
            FROM asset_face AS af
            INNER JOIN face_search AS fs ON fs."faceId" = af.id
            LEFT JOIN smart_search AS ss ON ss."assetId" = af."assetId"
            WHERE af."personGroupId" = %s::uuid
              AND af.id = ANY(%s::uuid[])
              AND af."deletedAt" IS NULL
              AND af."isVisible" IS TRUE
            ORDER BY af.id
        """
        try:
            with psycopg.connect(
                self.database_url,
                connect_timeout=self.connect_timeout,
                autocommit=True,
                options="-c default_transaction_read_only=on",
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, (person_id, list(face_ids)))
                    rows = cursor.fetchall()
        except Exception as error:
            name = type(error).__name__
            raise ImmichVectorStoreError(f"Immich vector read failed ({name})") from None

        output: list[tuple[str, str, object, object | None]] = []
        for row in rows:
            if not isinstance(row, tuple) or len(row) != 4:
                raise ImmichVectorStoreError("Immich vector query returned an unexpected row")
            face_id, asset_id, face_vector, scene_vector = row
            if not isinstance(face_id, str) or not isinstance(asset_id, str):
                raise ImmichVectorStoreError("Immich vector query returned invalid IDs")
            output.append((face_id, asset_id, face_vector, scene_vector))
        return output


def _parse_vector(value: object, label: str) -> np.ndarray:
    if not isinstance(value, str):
        raise ImmichVectorStoreError(f"Immich {label} embedding was not text")
    text = value.strip()
    valid_brackets = len(text) >= 2 and (text[0], text[-1]) in {("[", "]"), ("{", "}")}
    if not valid_brackets:
        raise ImmichVectorStoreError(f"Immich {label} embedding had an unexpected format")
    try:
        components = [component.strip() for component in text[1:-1].split(",")]
        if not components or any(not component for component in components):
            raise ValueError
        vector = np.asarray([float(component) for component in components], dtype=np.float32)
    except (OverflowError, TypeError, ValueError):
        raise ImmichVectorStoreError(f"Immich {label} embedding could not be parsed") from None
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ImmichVectorStoreError(f"Immich {label} embedding was invalid")
    if not np.any(vector):
        raise ImmichVectorStoreError(f"Immich {label} embedding was zero")
    return np.ascontiguousarray(vector)


def _uuid(value: str, label: str) -> None:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise ValueError(f"{label} must be a canonical UUID")
