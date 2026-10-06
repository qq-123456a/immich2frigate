import json
import time
import cv2
import numpy as np

from immich2frigate.settings import ImmichDatabaseSettings, ImmichSettings
from immich2frigate.teacher import ImmichTeacher, _overlaps_source, _pad_prediction_image

PERSON = "a" * 36
EMBEDDING = "[" + ",".join(["0.1"] * 512) + "]"


class Response:
    status_code = 200

    def __init__(self, value):
        self.body = json.dumps(value).encode()

    def iter_content(self, size):
        yield self.body

    def close(self):
        pass


class Session:
    def __init__(self, values):
        self.values = iter(values)
        self.headers = {}
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return Response(next(self.values))

    def close(self):
        pass


class Cursor:
    def __init__(self, rows, other_rows=None):
        self.rows = rows
        self.other_rows = rows if other_rows is None else other_rows
        self.calls = []
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, params):
        self.calls.append((query, params))
        self.params = params

    def fetchall(self):
        return self.rows

    def fetchone(self):
        winner_id = self.params[1]
        distances = [distance for person_id, distance in self.other_rows if person_id != winner_id]
        return (min(distances) if distances else None,)


class Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def cursor(self):
        return self._cursor


def make_teacher(rows, monkeypatch, other_rows=None):
    person = PERSON
    config = {"machineLearning": {"facialRecognition": {
        "enabled": True, "modelName": "buffalo_l", "minScore": 0.7,
        "maxDistance": 0.5, "minFaces": 3,
    }}}
    ml = {"imageWidth": 4, "imageHeight": 4, "facial-recognition": [{
        "embedding": EMBEDDING, "score": 0.9, "boundingBox": {"x1": 1, "y1": 1, "x2": 3, "y2": 3},
    }]}
    api_session, ml_session = Session([config]), Session([ml])
    cursor = Cursor(rows, other_rows)
    connect_calls = []

    def connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        return Connection(cursor)

    teacher = ImmichTeacher(
        ImmichSettings("http://immich:2283", "secret"),
        ImmichDatabaseSettings("postgresql://reader@db/immich"),
        ml_url="http://ml:3003", session=api_session, ml_session=ml_session, connect=connect,
    )
    teacher._people = {person: "Alice", **{person_id: "Bob" for person_id, _ in (other_rows or rows)
                                            if person_id != person}}
    teacher._people_loaded_at = time.monotonic()
    monkeypatch.setattr("immich2frigate.teacher._pad_prediction_image",
                        lambda image: (b"padded-png", (1, 1, 3, 3), (4, 4)))
    return teacher, person, api_session, ml_session, cursor, connect_calls


def test_confirm_requires_three_different_supporting_assets_and_read_only_database(monkeypatch):
    teacher, person, api, ml, cursor, connects = make_teacher(
        [(PERSON, 0.12)] * 3 + [("b" * 36, 0.3)], monkeypatch)
    result = teacher.confirm(b"crop")

    assert result.confirmed and result.person_id == person and result.name == "Alice"
    assert result.support == 3 and result.distance == 0.12
    assert api.headers["x-api-key"] == "secret"
    assert "x-api-key" not in ml.headers
    assert "INSERT" not in cursor.calls[0][0].upper()
    assert "DISTINCT" not in cursor.calls[0][0].upper()  # window rank keeps one face per asset
    assert "ROW_NUMBER() OVER" in cursor.calls[0][0]
    assert "PARTITION BY PERSON_ID, CHECKSUM" in cursor.calls[0][0].upper()
    assert "ENCODE(A.CHECKSUM, 'HEX')" in cursor.calls[0][0].upper()
    assert connects[0][1]["options"] == "-c default_transaction_read_only=on"
    assert ml.calls[0][1] == "http://ml:3003/predict"


def test_confirm_rejects_insufficient_support_and_profile_mismatch(monkeypatch):
    teacher, person, api, ml, cursor, _ = make_teacher([(PERSON, 0.1), (PERSON, 0.2)], monkeypatch)
    assert teacher.confirm(b"crop").status == "unconfirmed"
    prior_calls = len(ml.calls)
    api.values = iter([{"machineLearning": {"facialRecognition": {
        "enabled": True, "modelName": "buffalo_l", "minScore": 0.7,
        "maxDistance": 0.8, "minFaces": 3,
    }}}])
    assert teacher.confirm(b"crop").reason.startswith("service_error:")
    assert len(ml.calls) == prior_calls


def test_confirm_rejects_multiple_faces_before_querying_database(monkeypatch):
    teacher, person, api, ml, cursor, connects = make_teacher([], monkeypatch)
    ml.values = iter([{"imageWidth": 4, "imageHeight": 4, "facial-recognition": [{}, {}]}])
    assert teacher.confirm(b"crop").reason == "no_face_or_ambiguous"
    assert connects == []


def test_second_person_single_nearest_asset_still_blocks_small_margin(monkeypatch):
    teacher, _, _, _, _, _ = make_teacher(
        [(PERSON, .10), (PERSON, .12), (PERSON, .14), ("b" * 36, .16)], monkeypatch)
    result = teacher.confirm(b"crop")
    assert not result.confirmed and result.reason == "ambiguous_people"


def test_margin_uses_competitor_even_when_its_distance_exceeds_match_limit(monkeypatch):
    teacher, _, _, _, _, _ = make_teacher(
        [(PERSON, .42), (PERSON, .45), (PERSON, .48), ("b" * 36, .51)], monkeypatch)
    assert teacher.confirm(b"crop").confirmed
    teacher, _, _, _, _, _ = make_teacher(
        [(PERSON, .42), (PERSON, .45), (PERSON, .48), ("b" * 36, .49)], monkeypatch)
    result = teacher.confirm(b"crop")
    assert not result.confirmed and result.reason == "ambiguous_people"


def test_missing_competitor_reference_fails_closed(monkeypatch):
    teacher, _, _, _, _, _ = make_teacher([(PERSON, .10), (PERSON, .12), (PERSON, .14)], monkeypatch)
    teacher._people["b" * 36] = "Bob"
    result = teacher.confirm(b"crop")
    assert not result.confirmed and result.reason == "no_competing_reference"


def test_exact_nearest_named_person_search_finds_competitor_beyond_global_pool(monkeypatch):
    rival = "b" * 36
    bounded_pool = [(PERSON, .10 + index / 100000) for index in range(2000)]
    teacher, _, _, _, cursor, _ = make_teacher(
        bounded_pool, monkeypatch, other_rows=[(rival, .18)])

    result = teacher.confirm(b"crop", exclude_asset_id="asset-source", exclude_checksum="aabb")

    assert result.confirmed and result.support == 2000
    assert len(cursor.calls) == 2
    competitor_query, params = cursor.calls[1]
    assert "SELECT MIN(FS.EMBEDDING <=> %S::VECTOR)" in competitor_query.upper()
    assert 'AF."PERSONGROUPID" IS NOT NULL' in competitor_query.upper()
    assert 'AF."PERSONGROUPID" <> %S::UUID' in competitor_query.upper()
    assert params[1] == PERSON
    assert params[2:4] == ("asset-source", "asset-source")
    assert params[4:6] == ("aabb", "aabb")


def test_person_name_cache_expires_after_one_minute(monkeypatch):
    teacher, person, _, _, _, _ = make_teacher([], monkeypatch)
    teacher._people = None
    now = [100.0]
    monkeypatch.setattr("immich2frigate.teacher.time.monotonic", lambda: now[0])
    lookups = []

    class Person:
        def __init__(self, name):
            self.person_id, self.name = person, name

    class PeopleClient:
        def __init__(self, settings):
            self.name = "Alice" if not lookups else "Renamed"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def people(self):
            lookups.append(self.name)
            return [Person(self.name)]

    monkeypatch.setattr("immich2frigate.teacher.ImmichReadOnlyClient", PeopleClient)

    assert teacher._person_names()[person] == "Alice"
    now[0] += 59
    assert teacher._person_names()[person] == "Alice"
    assert len(lookups) == 1
    now[0] += 2
    assert teacher._person_names()[person] == "Renamed"
    assert len(lookups) == 2


def test_neutral_padding_and_detected_box_must_point_back_to_source_crop():
    ok, source = cv2.imencode(".jpg", np.full((8, 10, 3), 30, dtype=np.uint8))
    padded, central, size = _pad_prediction_image(source.tobytes())
    decoded = cv2.imdecode(np.frombuffer(padded, np.uint8), cv2.IMREAD_COLOR)
    assert ok and decoded.shape[:2] == (16, 20)
    assert central == (5, 4, 15, 12) and size == (20, 16)
    assert decoded[0, 0].tolist() == [127, 127, 127]
    assert _overlaps_source({"x1": 5, "y1": 4, "x2": 15, "y2": 12}, central, size)
    assert not _overlaps_source({"x1": 0, "y1": 0, "x2": 4, "y2": 4}, central, size)
