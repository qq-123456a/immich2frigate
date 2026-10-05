from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from immich2frigate.immich_client import FaceCandidate, PersonRecord
from immich2frigate.immich_vectors import ImmichVectorStore, ImmichVectorStoreError

PERSON = "00000000-0000-4000-8000-000000000001"
FACE = "00000000-0000-4000-8000-000000000002"
ASSET = "00000000-0000-4000-8000-000000000003"


def candidate():
    return FaceCandidate(
        person_id=PERSON,
        face_id=FACE,
        asset_id=ASSET,
        taken_at="2025-01-01T00:00:00Z",
        checksum="x",
        box=(0, 0, 10, 10),
        frame=(10, 10),
    )


class Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = None
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None
    def execute(self, query, params):
        self.executed = (query, params)
    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self, cursor):
        self._cursor = cursor
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None
    def cursor(self):
        return self._cursor


def install_psycopg(monkeypatch, rows):
    module = ModuleType("psycopg")
    cursor = Cursor(rows)
    calls = {}
    def connect(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return Connection(cursor)
    module.connect = connect
    monkeypatch.setitem(sys.modules, "psycopg", module)
    return cursor, calls


def test_vector_store_is_read_only_and_returns_both_vectors(monkeypatch):
    cursor, calls = install_psycopg(
        monkeypatch,
        [(FACE, ASSET, "[1,0,0]", "[0,1,0]")],
    )
    store = ImmichVectorStore("postgresql://reader:placeholder@immich-db/immich")

    result = store.vectors_for_person(PersonRecord(PERSON, "Amy"), [candidate()])

    assert len(result) == 1
    assert np.allclose(result[0].face_embedding, [1, 0, 0])
    assert np.allclose(result[0].scene_embedding, [0, 1, 0])
    assert calls["kwargs"]["autocommit"] is True
    assert "default_transaction_read_only=on" in calls["kwargs"]["options"]
    assert cursor.executed[1][0] == PERSON
    assert cursor.executed[1][1] == [FACE]
    assert "SELECT" in cursor.executed[0]
    assert "LEFT JOIN smart_search" in cursor.executed[0]
    assert all(word not in cursor.executed[0].upper() for word in ["INSERT ", "UPDATE ", "DELETE "])


def test_vector_store_keeps_face_when_smart_search_vector_is_missing(monkeypatch):
    install_psycopg(monkeypatch, [(FACE, ASSET, "[1,0,0]", None)])
    store = ImmichVectorStore("postgresql://reader:placeholder@immich-db/immich")

    result = store.vectors_for_person(PersonRecord(PERSON, "Amy"), [candidate()])

    assert len(result) == 1
    assert np.allclose(result[0].face_embedding, [1, 0, 0])
    assert result[0].scene_embedding is None


def test_vector_store_rejects_invalid_vectors(monkeypatch):
    install_psycopg(monkeypatch, [(FACE, ASSET, "[0,0]", "[0,1]")])
    store = ImmichVectorStore("postgresql://reader:placeholder@immich-db/immich")
    with pytest.raises(ImmichVectorStoreError, match="zero"):
        store.vectors_for_person(PersonRecord(PERSON, "Amy"), [candidate()])


@pytest.mark.parametrize("malformed", ["[1,2,garbage]", "[1,,2]", "{1,2,garbage}"])
def test_vector_store_rejects_malformed_vector_text(monkeypatch, malformed):
    install_psycopg(monkeypatch, [(FACE, ASSET, malformed, None)])
    store = ImmichVectorStore("postgresql://reader:placeholder@immich-db/immich")

    with pytest.raises(ImmichVectorStoreError):
        store.vectors_for_person(PersonRecord(PERSON, "Amy"), [candidate()])


def test_vector_store_accepts_postgres_array_vector_text(monkeypatch):
    install_psycopg(monkeypatch, [(FACE, ASSET, "{1,0,0}", None)])
    store = ImmichVectorStore("postgresql://reader:placeholder@immich-db/immich")

    result = store.vectors_for_person(PersonRecord(PERSON, "Amy"), [candidate()])

    assert np.allclose(result[0].face_embedding, [1, 0, 0])
