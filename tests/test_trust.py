"""
tests/test_trust.py
---------------------
Tests for the DYNAMIC TRUST SCORE stage (services/trust_engine.py).

Uses the project's real `test_db` fixture (a real, empty
AsyncIOMotorDatabase pointed at a test database) - same convention as
tests/test_fault_detection.py.

Covers:
  - fresh sensor starts at trust_score=100.0 / HEALTHY
  - a single fault prediction drops trust by FAULT_PENALTY and can move
    status to RECOVERING
  - repeated faults eventually cross FAULTY_THRESHOLD -> status FAULTY
  - trust never drops below 0
  - a healthy prediction after faults raises trust by RECOVERY_GAIN and
    resets consecutive_faults
  - trust never exceeds 100
  - sensor_health is upserted (one doc per sensor_id, not one per call)
  - the mirrored trust_score/status on the `sensors` collection stays in
    sync with `sensor_health`
  - update_trust raises ValueError for an unknown sensor_id (best-effort
    caller in reading_service.py is expected to catch and log this, not
    propagate it)
"""

import pytest
import pytest_asyncio

from models.sensor import SensorStatus, now_utc
from services import trust_engine


SENSOR_ID = "TRUST_TEST_TEMP_01"


def _make_sensor_doc(sensor_id: str = SENSOR_ID) -> dict:
    now = now_utc()
    return {
        "sensor_id": sensor_id,
        "name": "Trust Test Sensor",
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
    """Insert a real sensor doc directly (no need to go through the API for these unit-level tests)."""
    await test_db["sensors"].insert_one(_make_sensor_doc())
    yield SENSOR_ID
    await test_db["sensors"].delete_many({"sensor_id": SENSOR_ID})
    await test_db["sensor_health"].delete_many({"sensor_id": SENSOR_ID})


@pytest.mark.asyncio
async def test_first_fault_drops_trust_from_100(test_db, seeded_sensor):
    prediction = _make_prediction(is_fault=True)
    health = await trust_engine.update_trust(test_db, prediction)

    assert health["trust_score"] == pytest.approx(100.0 - trust_engine.FAULT_PENALTY)
    assert health["consecutive_faults"] == 1
    assert health["consecutive_healthy"] == 0


@pytest.mark.asyncio
async def test_repeated_faults_reach_faulty_status(test_db, seeded_sensor):
    # 100 -> 80 -> 60 -> 40 (3rd fault crosses FAULTY_THRESHOLD of 50)
    for _ in range(3):
        health = await trust_engine.update_trust(test_db, _make_prediction(is_fault=True))

    assert health["trust_score"] == pytest.approx(100.0 - 3 * trust_engine.FAULT_PENALTY)
    assert health["status"] == SensorStatus.FAULTY.value
    assert health["consecutive_faults"] == 3


@pytest.mark.asyncio
async def test_trust_never_drops_below_zero(test_db, seeded_sensor):
    health = None
    for _ in range(20):  # far more than enough to floor out at 0
        health = await trust_engine.update_trust(test_db, _make_prediction(is_fault=True))

    assert health["trust_score"] == pytest.approx(trust_engine.TRUST_MIN)
    assert health["status"] == SensorStatus.FAULTY.value


@pytest.mark.asyncio
async def test_recovery_after_faults_increases_trust_and_resets_streak(test_db, seeded_sensor):
    for _ in range(3):
        await trust_engine.update_trust(test_db, _make_prediction(is_fault=True))

    trust_after_faults = (await trust_engine.get_health(test_db, SENSOR_ID))["trust_score"]

    health = await trust_engine.update_trust(test_db, _make_prediction(is_fault=False))

    assert health["trust_score"] == pytest.approx(
        trust_after_faults + trust_engine.RECOVERY_GAIN
    )
    assert health["consecutive_faults"] == 0
    assert health["consecutive_healthy"] == 1


@pytest.mark.asyncio
async def test_trust_never_exceeds_100(test_db, seeded_sensor):
    health = None
    for _ in range(20):
        health = await trust_engine.update_trust(test_db, _make_prediction(is_fault=False))

    assert health["trust_score"] == pytest.approx(trust_engine.TRUST_MAX)
    assert health["status"] == SensorStatus.HEALTHY.value


@pytest.mark.asyncio
async def test_sensor_health_is_upserted_not_duplicated(test_db, seeded_sensor):
    for _ in range(5):
        await trust_engine.update_trust(test_db, _make_prediction(is_fault=True))

    count = await test_db["sensor_health"].count_documents({"sensor_id": SENSOR_ID})
    assert count == 1


@pytest.mark.asyncio
async def test_sensors_collection_mirrors_trust_score(test_db, seeded_sensor):
    health = await trust_engine.update_trust(test_db, _make_prediction(is_fault=True))

    sensor_doc = await test_db["sensors"].find_one({"sensor_id": SENSOR_ID})
    assert sensor_doc["trust_score"] == pytest.approx(health["trust_score"])
    assert sensor_doc["status"] == health["status"]


@pytest.mark.asyncio
async def test_update_trust_raises_for_unknown_sensor(test_db):
    with pytest.raises(ValueError):
        await trust_engine.update_trust(
            test_db, _make_prediction(is_fault=True, sensor_id="NO_SUCH_SENSOR")
        )


@pytest.mark.asyncio
async def test_get_health_returns_none_before_any_prediction(test_db, seeded_sensor):
    result = await trust_engine.get_health(test_db, SENSOR_ID)
    assert result is None
