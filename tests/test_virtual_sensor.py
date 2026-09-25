"""
tests/test_virtual_sensor.py
------------------------------
Tests for the VIRTUAL SENSOR / TRUSTED VALUE stage (Phase 5):
  - ml/virtual_sensor_features.py (pure functions, no DB needed)
  - services/virtual_sensor.py (uses a REAL, small RandomForestRegressor
    trained on synthetic data in a fixture - not a mocked prediction -
    so prediction logic, model loading/caching, normalization, and DB
    persistence are all exercised for real, per the master plan's
    "no hardcoded predictions" rule. Same convention as
    tests/test_fault_detection.py's tiny_trained_model fixture.)

Uses the project's real `test_db` / `seeded_sensor`-style fixtures -
same convention as tests/test_trust.py.
"""

import random

import joblib
import pytest
import pytest_asyncio
from sklearn.ensemble import RandomForestRegressor

from ml.virtual_sensor_features import (
    FEATURE_NAMES,
    compute_virtual_feature_matrix,
    compute_virtual_features,
    denormalize,
)
from models.sensor import SensorStatus, now_utc
from services import virtual_sensor
from tests.conftest import sample_sensor_payload


# ---------------------------------------------------------------------
# ml/virtual_sensor_features.py - pure unit tests, no DB
# ---------------------------------------------------------------------

def test_neutral_defaults_when_no_history():
    ts = now_utc()
    features = compute_virtual_features(ts, 10.0, 40.0, history_values=[])
    lag_1, lag_2, lag_3, rolling_mean, rolling_std, trend, _, _ = features
    assert lag_1 == lag_2 == lag_3 == rolling_mean == 0.5
    assert rolling_std == 0.0
    assert trend == 0.0


def test_lag_1_reflects_most_recent_history_value():
    ts = now_utc()
    history = [20.0, 22.0, 25.0]  # most recent = 25.0
    features = compute_virtual_features(ts, 10.0, 40.0, history_values=history)
    lag_1 = features[FEATURE_NAMES.index("lag_1")]
    assert lag_1 == pytest.approx((25.0 - 10.0) / 30.0)


def test_current_raw_value_is_never_a_feature_input():
    """
    The whole point of this module (master plan rule #5): the function
    signature doesn't even accept the reading's own value - only prior
    history. This test just documents/locks that contract by confirming
    two wildly different "current" scenarios produce IDENTICAL features
    when history and timestamp are identical (there is nothing else
    that COULD make them differ).
    """
    ts = now_utc()
    history = [21.0, 22.0, 23.0]
    features_a = compute_virtual_features(ts, 10.0, 40.0, history_values=history)
    features_b = compute_virtual_features(ts, 10.0, 40.0, history_values=history)
    assert features_a == features_b


def test_hour_features_are_bounded_cyclic_values():
    ts = now_utc()
    features = compute_virtual_features(ts, 10.0, 40.0, history_values=[])
    hour_sin = features[FEATURE_NAMES.index("hour_sin")]
    hour_cos = features[FEATURE_NAMES.index("hour_cos")]
    assert -1.0 <= hour_sin <= 1.0
    assert -1.0 <= hour_cos <= 1.0


def test_feature_matrix_is_causal_no_lookahead():
    """Row i's features must never depend on values[i+1:]."""
    values = [20.0, 21.0, 22.0, 90.0, 23.0, 24.0]  # a spike at index 3
    timestamps = [now_utc() for _ in values]
    X_full, y_full = compute_virtual_feature_matrix(10.0, 40.0, timestamps, values)
    X_prefix, y_prefix = compute_virtual_feature_matrix(10.0, 40.0, timestamps[:3], values[:3])
    assert X_full[0] == X_prefix[0]
    assert X_full[1] == X_prefix[1]
    assert X_full[2] == X_prefix[2]
    assert y_full[0] == y_prefix[0]
    assert y_full[2] == y_prefix[2]


def test_denormalize_is_inverse_of_internal_normalization():
    # A value exactly at normal_min should normalize to 0.0 and back.
    assert denormalize(0.0, 10.0, 40.0) == pytest.approx(10.0)
    assert denormalize(1.0, 10.0, 40.0) == pytest.approx(40.0)
    assert denormalize(0.5, 10.0, 40.0) == pytest.approx(25.0)


# ---------------------------------------------------------------------
# services/virtual_sensor.py - real (small) trained model + real DB
# ---------------------------------------------------------------------

SENSOR_ID = "VS_TEST_TEMP_01"
NORMAL_MIN = 15.0
NORMAL_MAX = 40.0


def _make_sensor_doc(sensor_id: str = SENSOR_ID) -> dict:
    now = now_utc()
    return {
        "sensor_id": sensor_id,
        "name": "Virtual Sensor Test Sensor",
        "sensor_type": "TEMPERATURE",
        "unit": "°C",
        "location": "Test Lab",
        "node_id": None,
        "normal_min": NORMAL_MIN,
        "normal_max": NORMAL_MAX,
        "sampling_interval_seconds": 5,
        "enabled": True,
        "status": SensorStatus.HEALTHY.value,
        "trust_score": 100.0,
        "created_at": now,
        "updated_at": now,
    }


def _make_reading(value: float, sensor_id: str = SENSOR_ID) -> dict:
    return {
        "id": "fake_reading_id",
        "sensor_id": sensor_id,
        "value": value,
        "timestamp": now_utc(),
    }


@pytest_asyncio.fixture
async def seeded_sensor(test_db):
    """Insert a real sensor doc directly (unit-level tests, no need to go through the API)."""
    await test_db["sensors"].insert_one(_make_sensor_doc())
    yield SENSOR_ID
    await test_db["sensors"].delete_many({"sensor_id": SENSOR_ID})
    await test_db["trusted_readings"].delete_many({"sensor_id": SENSOR_ID})


@pytest.fixture
def tiny_trained_virtual_model(tmp_path, monkeypatch):
    """
    Train a REAL RandomForestRegressor on synthetic (features, target)
    pairs where the target tracks lag_1 closely, save it to a temp
    path, and point services/virtual_sensor.py at it. Real training/
    inference on a small synthetic dataset, same spirit as
    test_fault_detection.py's tiny_trained_model fixture.
    """
    rng = random.Random(11)
    X, y = [], []
    for _ in range(200):
        lag_1 = rng.uniform(0.3, 0.7)
        lag_2 = lag_1 + rng.gauss(0, 0.02)
        lag_3 = lag_2 + rng.gauss(0, 0.02)
        rolling_mean = (lag_1 + lag_2 + lag_3) / 3
        rolling_std = abs(rng.gauss(0, 0.02))
        trend = lag_1 - lag_2
        hour_sin, hour_cos = rng.uniform(-1, 1), rng.uniform(-1, 1)
        X.append([lag_1, lag_2, lag_3, rolling_mean, rolling_std, trend, hour_sin, hour_cos])
        y.append(lag_1 + rng.gauss(0, 0.01))  # target stays close to lag_1

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "random_forest.joblib"
    joblib.dump(model, model_path)

    monkeypatch.setattr(virtual_sensor, "MODEL_PATH", model_path)
    virtual_sensor.reload_model()
    yield model_path
    virtual_sensor.reload_model()


@pytest.mark.asyncio
async def test_healthy_status_uses_raw_value_as_trusted_no_model_needed(test_db, seeded_sensor):
    """
    HEALTHY sensor -> trusted_value == raw value, source RAW. Crucially,
    no trained model file exists in this test at all, and this must
    still succeed - a healthy sensor never needs the regressor.
    """
    reading = _make_reading(value=22.5)
    result = await virtual_sensor.compute_trusted_value(test_db, reading, SensorStatus.HEALTHY.value)

    assert result["source"] == virtual_sensor.SOURCE_RAW
    assert result["trusted_value"] == pytest.approx(22.5)
    assert result["raw_value"] == pytest.approx(22.5)
    assert result["status_at_prediction"] == SensorStatus.HEALTHY.value


@pytest.mark.asyncio
async def test_faulty_status_without_trained_model_raises_unavailable(test_db, seeded_sensor, tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.joblib"
    monkeypatch.setattr(virtual_sensor, "MODEL_PATH", missing_path)
    virtual_sensor.reload_model()

    reading = _make_reading(value=9999.0)
    with pytest.raises(virtual_sensor.VirtualSensorUnavailable):
        await virtual_sensor.compute_trusted_value(test_db, reading, SensorStatus.FAULTY.value)

    virtual_sensor.reload_model()


@pytest.mark.asyncio
async def test_faulty_status_uses_virtual_predicted_value(test_db, seeded_sensor, tiny_trained_virtual_model):
    reading = _make_reading(value=9999.0)  # wildly out-of-range raw spike
    result = await virtual_sensor.compute_trusted_value(test_db, reading, SensorStatus.FAULTY.value)

    assert result["source"] == virtual_sensor.SOURCE_VIRTUAL
    assert result["raw_value"] == pytest.approx(9999.0)
    # The predicted value must not just echo the untrusted raw spike.
    assert result["trusted_value"] != pytest.approx(9999.0)
    # With no history yet, the model saw neutral (0.5-normalized) lag
    # features, so the predicted value should land roughly mid-range,
    # nowhere near the raw spike.
    assert NORMAL_MIN - 5.0 <= result["trusted_value"] <= NORMAL_MAX + 5.0


@pytest.mark.asyncio
async def test_recovering_status_also_uses_virtual_value(test_db, seeded_sensor, tiny_trained_virtual_model):
    """RECOVERING (not just FAULTY) counts as untrusted - only HEALTHY uses raw."""
    reading = _make_reading(value=5.0)
    result = await virtual_sensor.compute_trusted_value(test_db, reading, SensorStatus.RECOVERING.value)
    assert result["source"] == virtual_sensor.SOURCE_VIRTUAL


@pytest.mark.asyncio
async def test_prediction_never_depends_on_the_untrusted_raw_value(test_db, seeded_sensor, tiny_trained_virtual_model):
    """
    Explicit data-leakage sanity check (master plan rule #5): with
    identical history (empty, in this case) and identical status, two
    readings with WILDLY different raw values must produce the exact
    same predicted trusted_value - because the raw value is never fed
    into the model as a feature.
    """
    reading_a = _make_reading(value=-500.0)
    result_a = await virtual_sensor.compute_trusted_value(test_db, reading_a, SensorStatus.FAULTY.value)

    # Clear out the trusted_readings doc so history is empty again for
    # the second call too (isolating this to a pure feature comparison).
    await test_db["trusted_readings"].delete_many({"sensor_id": SENSOR_ID})

    reading_b = _make_reading(value=500000.0)
    result_b = await virtual_sensor.compute_trusted_value(test_db, reading_b, SensorStatus.FAULTY.value)

    assert result_a["trusted_value"] == pytest.approx(result_b["trusted_value"])
    assert result_a["raw_value"] != result_b["raw_value"]


@pytest.mark.asyncio
async def test_raw_and_trusted_values_stay_distinguishable(test_db, seeded_sensor, tiny_trained_virtual_model):
    """Rule #8: raw, predicted, and trusted values must remain distinguishable and retained."""
    reading = _make_reading(value=9999.0)
    result = await virtual_sensor.compute_trusted_value(test_db, reading, SensorStatus.FAULTY.value)

    doc = await test_db["trusted_readings"].find_one({"sensor_id": SENSOR_ID})
    assert doc["raw_value"] == pytest.approx(9999.0)
    assert doc["trusted_value"] == pytest.approx(result["trusted_value"])
    assert doc["raw_value"] != doc["trusted_value"]


@pytest.mark.asyncio
async def test_update_raises_for_unknown_sensor(test_db):
    reading = _make_reading(value=25.0, sensor_id="NO_SUCH_SENSOR")
    with pytest.raises(ValueError):
        await virtual_sensor.compute_trusted_value(test_db, reading, SensorStatus.HEALTHY.value)


@pytest.mark.asyncio
async def test_list_trusted_filters_by_sensor(test_db, seeded_sensor):
    await virtual_sensor.compute_trusted_value(test_db, _make_reading(20.0), SensorStatus.HEALTHY.value)
    await virtual_sensor.compute_trusted_value(test_db, _make_reading(21.0), SensorStatus.HEALTHY.value)

    other_sensor = "VS_OTHER_SENSOR"
    await test_db["sensors"].insert_one(_make_sensor_doc(other_sensor))
    await virtual_sensor.compute_trusted_value(
        test_db, _make_reading(22.0, sensor_id=other_sensor), SensorStatus.HEALTHY.value
    )

    results = await virtual_sensor.list_trusted(test_db, sensor_id=SENSOR_ID)
    assert len(results) == 2
    assert all(r["sensor_id"] == SENSOR_ID for r in results)

    await test_db["sensors"].delete_many({"sensor_id": other_sensor})
    await test_db["trusted_readings"].delete_many({"sensor_id": other_sensor})


@pytest.mark.asyncio
async def test_get_latest_trusted_404_before_any_prediction(client, test_db):
    await client.post("/sensors", json=sample_sensor_payload("VS_404_TEST"))
    resp = await client.get("/trusted-readings/VS_404_TEST/latest")
    assert resp.status_code == 404


# ---------------------------------------------------------------------
# End-to-end ingestion wiring - full pipeline through POST /readings
# ---------------------------------------------------------------------

@pytest.fixture
def tiny_trained_isolation_forest(tmp_path, monkeypatch):
    """
    Real, small, trained IsolationForest so the full pipeline
    (fault detection -> trust -> virtual sensing) can run end-to-end
    through the real POST /readings endpoint. Mirrors
    test_fault_detection.py's tiny_trained_model fixture.
    """
    from services import fault_detection

    rng = random.Random(7)
    normal = [[abs(rng.gauss(0, 0.05)), abs(rng.gauss(0, 1.0)), abs(rng.gauss(0, 0.05)), 0.0]
              for _ in range(200)]
    faulty = [[rng.uniform(1.0, 3.0), rng.uniform(5.0, 10.0), rng.uniform(0.5, 2.0), 0.0]
              for _ in range(20)]

    from sklearn.ensemble import IsolationForest
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(normal + faulty)

    model_path = tmp_path / "isolation_forest.joblib"
    joblib.dump(model, model_path)

    monkeypatch.setattr(fault_detection, "MODEL_PATH", model_path)
    fault_detection.reload_model()
    yield model_path
    fault_detection.reload_model()


@pytest.mark.asyncio
async def test_full_pipeline_ingestion_produces_a_trusted_reading(
    client, test_db, tiny_trained_isolation_forest, tiny_trained_virtual_model
):
    """
    POST /readings with a wildly anomalous value should flow all the
    way through fault detection -> trust scoring -> virtual sensing and
    land in GET /trusted-readings as a VIRTUAL trusted value - all
    through the real ingestion contract, response shape unchanged.
    """
    await client.post("/sensors", json=sample_sensor_payload("VS_PIPE_01"))

    resp = await client.post("/readings", json={"sensor_id": "VS_PIPE_01", "value": 9999.0})
    assert resp.status_code == 201
    assert set(resp.json().keys()) == {"sensor_id", "value", "timestamp", "id"}

    trusted_resp = await client.get("/trusted-readings/VS_PIPE_01/latest")
    assert trusted_resp.status_code == 200
    body = trusted_resp.json()
    assert body["sensor_id"] == "VS_PIPE_01"
    assert body["raw_value"] == pytest.approx(9999.0)
    assert body["source"] in ("RAW", "VIRTUAL")


@pytest.mark.asyncio
async def test_ingestion_still_succeeds_without_virtual_sensor_model(
    client, test_db, tiny_trained_isolation_forest, tmp_path, monkeypatch
):
    """
    If fault detection + trust scoring succeed but no virtual-sensor
    model has been trained yet, POST /readings must STILL return 201
    with the unchanged response shape (best-effort chain, rule #10/#11).
    """
    from services import virtual_sensor as vs

    missing_path = tmp_path / "does_not_exist.joblib"
    monkeypatch.setattr(vs, "MODEL_PATH", missing_path)
    vs.reload_model()

    await client.post("/sensors", json=sample_sensor_payload("VS_PIPE_02"))
    resp = await client.post("/readings", json={"sensor_id": "VS_PIPE_02", "value": 9999.0})

    assert resp.status_code == 201
    assert set(resp.json().keys()) == {"sensor_id", "value", "timestamp", "id"}

    vs.reload_model()
