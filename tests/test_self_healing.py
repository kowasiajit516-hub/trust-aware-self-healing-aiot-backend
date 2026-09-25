"""
tests/test_self_healing.py
-----------------------------
Tests for the SELF-HEALING ENGINE (Phase 6):
  - services/self_healing.py (record/list/get fault_events directly)
  - its wiring into services/trust_engine.update_trust (only a real
    status TRANSITION should ever produce a fault_events document -
    not every reading)
  - a full end-to-end run through the real ingestion pipeline
    (POST /readings -> fault detection -> trust scoring -> self-healing
    event), using the same tiny_trained_model convention as
    tests/test_fault_detection.py

Uses the project's real `test_db` / `client` fixtures - same
convention as tests/test_trust.py and tests/test_fault_detection.py.
"""

import random

import joblib
import pytest
import pytest_asyncio
from sklearn.ensemble import IsolationForest

from models.sensor import SensorStatus, now_utc
from services import fault_detection, self_healing, trust_engine
from tests.conftest import sample_sensor_payload

SENSOR_ID = "SH_TEST_TEMP_01"


def _make_sensor_doc(sensor_id: str = SENSOR_ID) -> dict:
    now = now_utc()
    return {
        "sensor_id": sensor_id,
        "name": "Self-Healing Test Sensor",
        "sensor_type": "TEMPERATURE",
        "unit": "°C",
        "location": "Test Lab",
        "node_id": None,
        "normal_min": 15.0,
        "normal_max": 40.0,
        "sampling_interval_seconds": 5,
        "enabled": True,
        "status": SensorStatus.HEALTHY.value,
        "trust_score": 100.0,
        "created_at": now,
        "updated_at": now,
    }


def _make_prediction(is_fault: bool, sensor_id: str = SENSOR_ID) -> dict:
    return {
        "id": "fake_prediction_id",
        "sensor_id": sensor_id,
        "reading_id": "fake_reading_id",
        "value": 999.0 if is_fault else 22.0,
        "timestamp": now_utc(),
        "anomaly_score": -0.5 if is_fault else 0.2,
        "is_fault": is_fault,
        "model_version": "test",
    }


@pytest_asyncio.fixture
async def seeded_sensor(test_db):
    """Insert a real sensor doc directly (unit-level tests, no need to go through the API)."""
    await test_db["sensors"].insert_one(_make_sensor_doc())
    yield SENSOR_ID
    await test_db["sensors"].delete_many({"sensor_id": SENSOR_ID})
    await test_db["sensor_health"].delete_many({"sensor_id": SENSOR_ID})
    await test_db["fault_events"].delete_many({"sensor_id": SENSOR_ID})


# ---------------------------------------------------------------------
# services/self_healing.py - direct unit tests
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_event_recorded_when_status_unchanged(test_db):
    result = await self_healing.record_transition_if_changed(
        test_db,
        sensor_id=SENSOR_ID,
        previous_status=SensorStatus.HEALTHY.value,
        new_status=SensorStatus.HEALTHY.value,
        trust_score=100.0,
        trigger_reading_id="r1",
        timestamp=now_utc(),
    )
    assert result is None
    count = await test_db["fault_events"].count_documents({"sensor_id": SENSOR_ID})
    assert count == 0


@pytest.mark.asyncio
async def test_event_recorded_when_status_changes(test_db):
    result = await self_healing.record_transition_if_changed(
        test_db,
        sensor_id=SENSOR_ID,
        previous_status=SensorStatus.HEALTHY.value,
        new_status=SensorStatus.RECOVERING.value,
        trust_score=80.0,
        trigger_reading_id="r1",
        timestamp=now_utc(),
    )
    assert result is not None
    assert result["previous_status"] == SensorStatus.HEALTHY.value
    assert result["new_status"] == SensorStatus.RECOVERING.value
    assert result["trust_score"] == pytest.approx(80.0)
    assert result["trigger_reading_id"] == "r1"
    assert "id" in result and "created_at" in result

    count = await test_db["fault_events"].count_documents({"sensor_id": SENSOR_ID})
    assert count == 1


@pytest.mark.asyncio
async def test_list_fault_events_filters_by_sensor(test_db):
    await self_healing.record_transition_if_changed(
        test_db, SENSOR_ID, "HEALTHY", "RECOVERING", 80.0, "r1", now_utc()
    )
    await self_healing.record_transition_if_changed(
        test_db, "OTHER_SENSOR", "HEALTHY", "RECOVERING", 80.0, "r2", now_utc()
    )

    results = await self_healing.list_fault_events(test_db, sensor_id=SENSOR_ID)
    assert len(results) == 1
    assert results[0]["sensor_id"] == SENSOR_ID

    await test_db["fault_events"].delete_many({"sensor_id": "OTHER_SENSOR"})


@pytest.mark.asyncio
async def test_get_latest_fault_event_404_before_any_transition(test_db, seeded_sensor):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await self_healing.get_latest_fault_event(test_db, SENSOR_ID)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_latest_fault_event_returns_most_recent(test_db, seeded_sensor):
    await self_healing.record_transition_if_changed(
        test_db, SENSOR_ID, "HEALTHY", "RECOVERING", 80.0, "r1", now_utc()
    )
    await self_healing.record_transition_if_changed(
        test_db, SENSOR_ID, "RECOVERING", "FAULTY", 40.0, "r2", now_utc()
    )

    latest = await self_healing.get_latest_fault_event(test_db, SENSOR_ID)
    assert latest["previous_status"] == "RECOVERING"
    assert latest["new_status"] == "FAULTY"


# ---------------------------------------------------------------------
# Wiring into services/trust_engine.update_trust
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transition_wired_through_trust_engine(test_db, seeded_sensor):
    """A single fault that crosses HEALTHY_THRESHOLD (100 -> 80) must record exactly one event."""
    await trust_engine.update_trust(test_db, _make_prediction(is_fault=True))

    events = await self_healing.list_fault_events(test_db, sensor_id=SENSOR_ID)
    assert len(events) == 1
    assert events[0]["previous_status"] == SensorStatus.HEALTHY.value
    assert events[0]["new_status"] == SensorStatus.RECOVERING.value


@pytest.mark.asyncio
async def test_only_transitions_recorded_not_every_reading(test_db, seeded_sensor):
    """
    Repeated faults while already RECOVERING/FAULTY must NOT produce a
    new event each time - only when the derived status actually
    crosses a boundary. This is the whole point of fault_events being
    a transition log, not a per-reading log (predictions already is
    that, from Phase 3).
    """
    # 100 -> 80 (HEALTHY->RECOVERING, 1 event) -> 60 (RECOVERING->RECOVERING,
    # no event) -> 40 (RECOVERING->FAULTY, 2nd event) -> 20 (FAULTY->FAULTY, no event)
    for _ in range(4):
        await trust_engine.update_trust(test_db, _make_prediction(is_fault=True))

    events = await self_healing.list_fault_events(test_db, sensor_id=SENSOR_ID)
    assert len(events) == 2
    statuses = [(e["previous_status"], e["new_status"]) for e in events]
    # most-recent-first ordering
    assert statuses[1] == (SensorStatus.HEALTHY.value, SensorStatus.RECOVERING.value)
    assert statuses[0] == (SensorStatus.RECOVERING.value, SensorStatus.FAULTY.value)


@pytest.mark.asyncio
async def test_full_self_healing_loop_records_every_transition_in_order(test_db, seeded_sensor):
    """
    Drives the master plan's section 4 demo loop end-to-end at the
    trust_engine level: NORMAL -> FAULT -> RECOVERING -> FAULTY ->
    (healthy readings resume) -> RECOVERING -> HEALTHY, and confirms
    fault_events captures every boundary crossing, in chronological
    order, with no duplicates for readings that don't cross a boundary.
    """
    # Drive down to FAULTY: 100 -> 80 -> 60 -> 40 (3 faults)
    for _ in range(3):
        await trust_engine.update_trust(test_db, _make_prediction(is_fault=True))
    # Drive back up to HEALTHY: 40 -> 45 -> 50 -> 55 -> ... -> 100 (12 healthy, +5 each)
    for _ in range(12):
        await trust_engine.update_trust(test_db, _make_prediction(is_fault=False))

    final_health = await trust_engine.get_health(test_db, SENSOR_ID)
    assert final_health["status"] == SensorStatus.HEALTHY.value

    events = await self_healing.list_fault_events(test_db, sensor_id=SENSOR_ID, limit=100)
    # chronological order (oldest first) for readability
    events = list(reversed(events))
    transitions = [(e["previous_status"], e["new_status"]) for e in events]

    assert transitions == [
        (SensorStatus.HEALTHY.value, SensorStatus.RECOVERING.value),   # 100 -> 80
        (SensorStatus.RECOVERING.value, SensorStatus.FAULTY.value),    # 60 -> 40
        (SensorStatus.FAULTY.value, SensorStatus.RECOVERING.value),    # 45 -> 50
        (SensorStatus.RECOVERING.value, SensorStatus.HEALTHY.value),   # 85 -> 90
    ]


@pytest.mark.asyncio
async def test_self_healing_failure_never_breaks_trust_update(test_db, seeded_sensor, monkeypatch):
    """
    Rule: a failure in the self-healing/event-recording stage must
    never break trust scoring (same best-effort contract every other
    downstream stage in this pipeline follows).
    """
    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated self-healing failure")

    monkeypatch.setattr(self_healing, "record_transition_if_changed", _boom)

    # This crosses HEALTHY -> RECOVERING, which would normally trigger
    # the (now broken) self-healing hook - must still return normally.
    health = await trust_engine.update_trust(test_db, _make_prediction(is_fault=True))
    assert health["status"] == SensorStatus.RECOVERING.value
    assert health["trust_score"] == pytest.approx(80.0)


# ---------------------------------------------------------------------
# End-to-end through the real ingestion pipeline
# ---------------------------------------------------------------------

@pytest.fixture
def tiny_trained_model(tmp_path, monkeypatch):
    """Same convention as tests/test_fault_detection.py's fixture of the same name."""
    rng = random.Random(7)
    normal = [[abs(rng.gauss(0, 0.05)), abs(rng.gauss(0, 1.0)), abs(rng.gauss(0, 0.05)), 0.0]
              for _ in range(200)]
    faulty = [[rng.uniform(1.0, 3.0), rng.uniform(5.0, 10.0), rng.uniform(0.5, 2.0), 0.0]
              for _ in range(20)]

    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(normal + faulty)

    model_path = tmp_path / "isolation_forest.joblib"
    joblib.dump(model, model_path)

    monkeypatch.setattr(fault_detection, "MODEL_PATH", model_path)
    fault_detection.reload_model()
    yield model_path
    fault_detection.reload_model()


@pytest.mark.asyncio
async def test_end_to_end_ingestion_can_produce_a_fault_event(client, test_db, tiny_trained_model):
    """
    POST /readings with a wildly out-of-range value repeatedly, through
    the real API, and confirm a fault_events transition eventually
    appears via GET /fault-events - proving the full chain (fault
    detection -> trust scoring -> self-healing) is wired end-to-end,
    not just at the unit level.
    """
    await client.post("/sensors", json=sample_sensor_payload("SH_E2E_01"))

    saw_transition = False
    for _ in range(5):
        resp = await client.post("/readings", json={"sensor_id": "SH_E2E_01", "value": 9999.0})
        assert resp.status_code == 201  # ingestion contract preserved regardless

        events_resp = await client.get("/fault-events", params={"sensor_id": "SH_E2E_01"})
        assert events_resp.status_code == 200
        if events_resp.json():
            saw_transition = True
            break

    assert saw_transition, "expected at least one status transition after repeated fault readings"


@pytest.mark.asyncio
async def test_end_to_end_ingestion_unaffected_when_status_never_changes(client, test_db, tiny_trained_model):
    """
    A single healthy-looking reading should not cross any status
    boundary (fresh sensor starts at HEALTHY/100), so GET /fault-events
    should stay empty - confirming fault_events truly is a transition
    log, not a per-reading log, at the full API level too.
    """
    await client.post("/sensors", json=sample_sensor_payload("SH_E2E_02"))
    resp = await client.post("/readings", json={"sensor_id": "SH_E2E_02", "value": 25.0})
    assert resp.status_code == 201

    events_resp = await client.get("/fault-events", params={"sensor_id": "SH_E2E_02"})
    assert events_resp.status_code == 200
    assert events_resp.json() == []