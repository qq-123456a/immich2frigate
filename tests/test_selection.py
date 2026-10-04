from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from immich2frigate.immich_client import FaceCandidate, PersonRecord
from immich2frigate.selection import EmbeddedFace, select_diverse_faces

PERSON = "00000000-0000-4000-8000-000000000001"


def embedded(asset_id: str, vector: np.ndarray) -> EmbeddedFace:
    source = FaceCandidate(
        person_id=PERSON,
        face_id=None,
        asset_id=asset_id,
        taken_at="2025-01-01T00:00:00Z",
        checksum="synthetic-checksum",
        box=(0.0, 0.0, 10.0, 10.0),
        frame=(10, 10),
    )
    return EmbeddedFace(source, np.zeros((10, 10, 3), np.uint8), vector)


def install_fake_selector(monkeypatch, selected_slice=slice(None)):
    package = ModuleType("if_curator")
    package.__path__ = []
    module = ModuleType("if_curator.selection")

    class Candidate:
        def __init__(self, asset_id, **kwargs):
            self.asset_id = asset_id
            self.__dict__.update(kwargs)

    class Job:
        def __init__(self, person, count, candidates):
            self.person = person
            self.count = count
            self.candidates = candidates
            self.selected = []
            self.recognized = None

    def select(jobs, threshold):
        job = jobs[0]
        assert job.person == {"id": PERSON, "name": "Synthetic Person"}
        assert threshold == 0.3
        job.selected = job.candidates[: job.count][selected_slice]
        job.recognized = "legacy 0.17 score must be ignored"

    module.Candidate = Candidate
    module.Job = Job
    module.select = select
    monkeypatch.setitem(sys.modules, "if_curator", package)
    monkeypatch.setitem(sys.modules, "if_curator.selection", module)


def test_selector_maps_results_by_candidate_identity_and_ignores_old_score(monkeypatch):
    install_fake_selector(monkeypatch)
    person = PersonRecord(PERSON, "Synthetic Person")
    candidates = [
        embedded("00000000-0000-4000-8000-000000000002", np.array([1, 0], np.float32)),
        embedded("00000000-0000-4000-8000-000000000003", np.array([0, 1], np.float32)),
    ]

    result = select_diverse_faces(person, candidates, count=1)

    assert result == candidates[:1]
    assert "legacy" not in repr(result)


def test_selector_rejects_invalid_count_shape_zero_or_mismatched_vectors(monkeypatch):
    install_fake_selector(monkeypatch)
    person = PersonRecord(PERSON, "Synthetic Person")
    valid = embedded("00000000-0000-4000-8000-000000000002", np.array([1, 0], np.float32))
    with pytest.raises(ValueError, match="positive"):
        select_diverse_faces(person, [valid], count=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        select_diverse_faces(person, [embedded(valid.source.asset_id, np.ones((1, 2), np.float32))])
    with pytest.raises(ValueError, match="non-zero"):
        select_diverse_faces(person, [embedded(valid.source.asset_id, np.zeros(2, np.float32))])
    other = embedded("00000000-0000-4000-8000-000000000003", np.ones(3, np.float32))
    with pytest.raises(ValueError, match="same dimension"):
        select_diverse_faces(person, [valid, other])


def test_selector_rejects_ambiguous_same_asset_and_wrong_person(monkeypatch):
    install_fake_selector(monkeypatch)
    person = PersonRecord(PERSON, "Synthetic Person")
    one = embedded("00000000-0000-4000-8000-000000000002", np.array([1, 0], np.float32))
    same_asset = embedded(one.source.asset_id, np.array([0, 1], np.float32))
    with pytest.raises(ValueError, match="ambiguous"):
        select_diverse_faces(person, [one, same_asset])

    wrong = FaceCandidate(
        person_id="00000000-0000-4000-8000-000000000009",
        face_id=None,
        asset_id="00000000-0000-4000-8000-000000000003",
        taken_at="2025-01-01T00:00:00Z",
        checksum="synthetic-checksum",
        box=(0, 0, 10, 10),
        frame=(10, 10),
    )
    with pytest.raises(ValueError, match="different Immich person"):
        select_diverse_faces(person, [EmbeddedFace(wrong, one.image_bgr, one.embedding)])
