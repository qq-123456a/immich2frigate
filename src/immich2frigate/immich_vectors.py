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
from .selection import VectorFace


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
