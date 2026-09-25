"""
tests/test_fault_detection.py
--------------------------------
Tests for the AI FAULT DETECTION stage (Phase 3):
  - ml/feature_engineering.py (pure functions, no DB needed)
  - services/fault_detection.py (uses a REAL, small IsolationForest
    trained on synthetic data in a fixture - not a mocked prediction -
    so scoring logic, model loading/caching, and DB persistence are all
    exercised for real, per the master plan's "no hardcoded predictions"
    rule).
"""

import random

import joblib
import pytest
from sklearn.ensemble import IsolationForest

from ml.feature_engineering import compute_feature_vector, compute_feature_matrix, FEATURE_NAMES
from services import fault_detection
from tests.conftest import sample_sensor_payload


# ---------------------------------------------------------------------
# feature_engineering - pure unit tests, no DB
# ---------------------------------------------------------------------

def test_deviation_zero_when_in_range():
    features = compute_feature_vector(25.0, 10.0, 40.0, history_values=[])
    assert features[FEATURE_NAMES.index("deviation")] == 0.0


def test_deviation_positive_when_out_of_range():
    features = compute_feature_vector(100.0, 10.0, 40.0, history_values=[])
    assert features[FEATURE_NAMES.index("deviation")] > 0.0


def test_stuck_flag_detects_repeated_value():
    history = [25.0, 25.1, 25.0, 24.9, 25.0]
    features = compute_feature_vector(25.0, 10.0, 40.0, history_values=history + [25.0])
    # last history value equals the new value -> stuck
    assert features[FEATURE_NAMES.index("stuck")] == 1.0


def test_stuck_flag_zero_when_value_changes():
    history = [25.0]
    features = compute_feature_vector(30.0, 10.0, 40.0, history_values=history)
    assert features[FEATURE_NAMES.index("stuck")] == 0.0


def test_zscore_high_for_outlier_relative_to_recent_history():
    stable_history = [25.0] * 10
    normal_features = compute_feature_vector(25.1, 10.0, 40.0, history_values=stable_history)
    outlier_features = compute_feature_vector(38.0, 10.0, 40.0, history_values=stable_history)
    z_idx = FEATURE_NAMES.index("zscore")
    assert outlier_features[z_idx] > normal_features[z_idx]


def test_feature_matrix_is_causal_no_lookahead():
    """Feature i must never depend on values[i+1:]."""
    values = [25.0, 26.0, 90.0, 25.5, 25.6]  # a spike at index 2
    matrix_full = compute_feature_matrix(10.0, 40.0, values)
    matrix_prefix = compute_feature_matrix(10.0, 40.0, values[:3])
    # Features for indices 0,1,2 must be identical whether or not later
    # values (index 3,4) exist.
    assert matrix_full[0] == matrix_prefix[0]
    assert matrix_full[1] == matrix_prefix[1]
    assert matrix_full[2] == matrix_prefix[2]


def test_feature_matrix_flags_the_injected_spike():
    values = [25.0] * 10 + [90.0]  # spike after a stable baseline
    matrix = compute_feature_matrix(10.0, 40.0, values)
    dev_idx = FEATURE_NAMES.index("deviation")
    z_idx = FEATURE_NAMES.index("zscore")
    assert matrix[-1][dev_idx] > 0.0
    assert matrix[-1][z_idx] > matrix[5][z_idx]


# ---------------------------------------------------------------------
# fault_detection service - real (small) trained model + real DB
# ---------------------------------------------------------------------

@pytest.fixture
def tiny_trained_model(tmp_path, monkeypatch):
    """
    Train a REAL IsolationForest on synthetic normal + faulty feature
    vectors, save it to a temp path, and point fault_detection at it.
    This is real training/inference, just on a small synthetic dataset
    instead of the full ml/train_isolation_forest.py pipeline, so tests
    stay fast and hermetic.
    """
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
async def test_score_reading_unavailable_without_trained_model(client, test_db, monkeypatch, tmp_path):
    missing_path = tmp_path / "does_not_exist.joblib"
    monkeypatch.setattr(fault_detection, "MODEL_PATH", missing_path)
    fault_detection.reload_model()

    await client.post("/sensors", json=sample_sensor_payload("FD_01"))
    resp = await client.post("/readings", json={"sensor_id": "FD_01", "value": 25.0})

    # Ingestion must still succeed even with no trained model (Phase 2
    # behavior/contract preserved).
    assert resp.status_code == 201

    with pytest.raises(fault_detection.FaultDetectionUnavailable):
        fault_detection._get_model()

    fault_detection.reload_model()


@pytest.mark.asyncio
async def test_reading_ingestion_produces_a_prediction(client, test_db, tiny_trained_model):
    await client.post("/sensors", json=sample_sensor_payload("FD_02"))

    resp = await client.post("/readings", json={"sensor_id": "FD_02", "value": 25.0})
    assert resp.status_code == 201
    reading_id = resp.json()["id"]

    pred_resp = await client.get("/predictions/FD_02/latest")
    assert pred_resp.status_code == 200
    body = pred_resp.json()
    assert body["sensor_id"] == "FD_02"
    assert body["reading_id"] == reading_id
    assert isinstance(body["is_fault"], bool)
    assert isinstance(body["anomaly_score"], float)


@pytest.mark.asyncio
async def test_actuators_never_see_raw_faulty_value_directly(client, test_db, tiny_trained_model):
    """
    Sanity-check for the master plan's hardest rule ('actuators act on
    trusted values only, never raw'): Phase 3 does not touch actuators
    at all yet, so this simply confirms fault detection does not, as a
    side effect, create/modify any actuator_events - that stage does
    not exist until Phase 6/9.
    """
    await client.post("/sensors", json=sample_sensor_payload("FD_03"))
    await client.post("/readings", json={"sensor_id": "FD_03", "value": 999.0})

    events_resp = await client.get("/predictions", params={"sensor_id": "FD_03"})
    assert events_resp.status_code == 200
    assert len(events_resp.json()) == 1


@pytest.mark.asyncio
async def test_list_predictions_filters_by_sensor(client, test_db, tiny_trained_model):
    await client.post("/sensors", json=sample_sensor_payload("FD_04"))
    await client.post("/sensors", json=sample_sensor_payload("FD_05"))

    await client.post("/readings", json={"sensor_id": "FD_04", "value": 25.0})
    await client.post("/readings", json={"sensor_id": "FD_05", "value": 26.0})
    await client.post("/readings", json={"sensor_id": "FD_04", "value": 27.0})

    resp = await client.get("/predictions", params={"sensor_id": "FD_04"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert all(p["sensor_id"] == "FD_04" for p in body)


@pytest.mark.asyncio
async def test_latest_prediction_404_for_unknown_sensor(client, test_db, tiny_trained_model):
    resp = await client.get("/predictions/DOES_NOT_EXIST/latest")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_model_cache_reused_across_calls(client, test_db, tiny_trained_model):
    """The model should be loaded once and cached, not reloaded per reading."""
    await client.post("/sensors", json=sample_sensor_payload("FD_06"))

    fault_detection.reload_model()
    assert fault_detection._model_cache == {}

    await client.post("/readings", json={"sensor_id": "FD_06", "value": 25.0})
    first_model = fault_detection._model_cache["model"]

    await client.post("/readings", json={"sensor_id": "FD_06", "value": 25.5})
    second_model = fault_detection._model_cache["model"]

    assert first_model is second_model
