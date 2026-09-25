"""
tests/test_settings.py
-------------------------
Tests for GET/PUT /settings, and - critically - that changing a
setting actually changes trust_engine.update_trust()'s behavior on
the very next reading, not just that the setting value is stored.

Run with:
    pytest tests/test_settings.py -v
"""

import random

import joblib
import pytest
from sklearn.ensemble import IsolationForest

from services import fault_detection
from tests.conftest import sample_sensor_payload


@pytest.fixture
def tiny_trained_model(tmp_path, monkeypatch):
    """
    Train a REAL IsolationForest on synthetic normal + faulty feature
    vectors (same convention as tests/test_fault_detection.py) so
    readings actually get scored - the settings tests need a real
    is_fault=True result to prove a changed fault_penalty/threshold
    actually changes trust behavior, not a hardcoded one.
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
async def test_get_settings_returns_real_defaults(client):
    resp = await client.get("/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["fault_penalty"] == 20.0
    assert data["recovery_gain"] == 5.0
    assert data["faulty_threshold"] == 50.0
    assert data["healthy_threshold"] == 90.0
    assert data["trust_max"] == 100.0
    assert data["trust_min"] == 0.0


@pytest.mark.asyncio
async def test_put_settings_partial_update(client):
    resp = await client.put("/settings", json={"fault_penalty": 35.0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["fault_penalty"] == 35.0
    # untouched fields keep their previous real values
    assert data["recovery_gain"] == 5.0

    # persists across a fresh GET
    get_resp = await client.get("/settings")
    assert get_resp.json()["fault_penalty"] == 35.0


@pytest.mark.asyncio
async def test_put_settings_rejects_out_of_bounds(client):
    resp = await client.put("/settings", json={"fault_penalty": 500.0})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_changed_fault_penalty_actually_changes_trust_drop(client, tiny_trained_model):
    """
    The real functional test: PUT a custom fault_penalty, then feed in
    an out-of-range reading, and confirm trust actually dropped by the
    NEW amount - not the old hardcoded 20.0. This proves update_trust()
    is really reading live settings, not just that the API stores them.
    """
    await client.put("/settings", json={"fault_penalty": 40.0})

    await client.post("/sensors", json=sample_sensor_payload("SETTINGS_TRUST_01"))
    # sample_sensor_payload's normal range is 15.0-40.0; way out of range
    # should be flagged as a fault by the tiny trained model.
    resp = await client.post("/readings", json={"sensor_id": "SETTINGS_TRUST_01", "value": 999.0})
    assert resp.status_code == 201

    health_resp = await client.get("/sensor-health/SETTINGS_TRUST_01")
    assert health_resp.status_code == 200
    health = health_resp.json()
    # started at 100.0, one fault at penalty=40.0 -> 60.0
    assert health["trust_score"] == 60.0


@pytest.mark.asyncio
async def test_changed_thresholds_actually_change_derived_status(client, tiny_trained_model):
    """Confirm a custom faulty_threshold actually changes which status a given trust_score maps to."""
    # With a much higher faulty_threshold, even a single fault (100 -> 80
    # at the default penalty) should already read as FAULTY, not RECOVERING.
    await client.put("/settings", json={"faulty_threshold": 85.0})

    await client.post("/sensors", json=sample_sensor_payload("SETTINGS_STATUS_01"))
    resp = await client.post("/readings", json={"sensor_id": "SETTINGS_STATUS_01", "value": 999.0})
    assert resp.status_code == 201

    health_resp = await client.get("/sensor-health/SETTINGS_STATUS_01")
    health = health_resp.json()
    assert health["trust_score"] == 80.0  # default fault_penalty still 20.0
    assert health["status"] == "FAULTY"  # would be RECOVERING under the old default 50.0 threshold


@pytest.mark.asyncio
async def test_settings_failure_falls_back_to_defaults_and_never_breaks_readings(client, monkeypatch, tiny_trained_model):
    """A settings-service failure must never break the real ingestion pipeline."""
    from services import trust_engine

    async def _boom(db):
        raise RuntimeError("simulated settings outage")

    monkeypatch.setattr("services.settings_service.get_settings", _boom)

    await client.post("/sensors", json=sample_sensor_payload("SETTINGS_FAILSAFE_01"))
    resp = await client.post("/readings", json={"sensor_id": "SETTINGS_FAILSAFE_01", "value": 27.0})
    assert resp.status_code == 201  # ingestion still works, using module fallback defaults
