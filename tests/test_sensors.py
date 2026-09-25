"""
tests/test_sensors.py
----------------------
Real tests for the Sensor CRUD API. Run with:

    pytest tests/test_sensors.py -v

Requires MongoDB to be running (uses a real, isolated test database -
see conftest.py). These are not mocked - they exercise the actual
FastAPI routes, service layer, and MongoDB together.
"""

import pytest

from tests.conftest import sample_sensor_payload


@pytest.mark.asyncio
async def test_create_sensor(client):
    payload = sample_sensor_payload("TEMP_TEST_CREATE")
    resp = await client.post("/sensors", json=payload)

    assert resp.status_code == 201
    data = resp.json()
    assert data["sensor_id"] == "TEMP_TEST_CREATE"
    assert data["status"] == "HEALTHY"
    assert data["trust_score"] == 100.0
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_sensor_duplicate_id_fails(client):
    payload = sample_sensor_payload("TEMP_TEST_DUP")
    resp1 = await client.post("/sensors", json=payload)
    assert resp1.status_code == 201

    resp2 = await client.post("/sensors", json=payload)
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_create_sensor_invalid_range_fails(client):
    payload = sample_sensor_payload("TEMP_TEST_BADRANGE")
    payload["normal_min"] = 50.0
    payload["normal_max"] = 10.0  # max < min -> should fail validation

    resp = await client.post("/sensors", json=payload)
    assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_list_sensors(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_TEST_LIST1"))
    await client.post("/sensors", json=sample_sensor_payload("TEMP_TEST_LIST2"))

    resp = await client.get("/sensors")
    assert resp.status_code == 200
    data = resp.json()
    ids = [s["sensor_id"] for s in data]
    assert "TEMP_TEST_LIST1" in ids
    assert "TEMP_TEST_LIST2" in ids


@pytest.mark.asyncio
async def test_get_sensor(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_TEST_GET"))

    resp = await client.get("/sensors/TEMP_TEST_GET")
    assert resp.status_code == 200
    assert resp.json()["sensor_id"] == "TEMP_TEST_GET"


@pytest.mark.asyncio
async def test_get_sensor_not_found(client):
    resp = await client.get("/sensors/DOES_NOT_EXIST")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_sensor(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_TEST_UPDATE"))

    resp = await client.put(
        "/sensors/TEMP_TEST_UPDATE",
        json={"location": "Lab 2", "normal_max": 45.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["location"] == "Lab 2"
    assert data["normal_max"] == 45.0
    # Unchanged fields should remain the same
    assert data["sensor_type"] == "TEMPERATURE"


@pytest.mark.asyncio
async def test_delete_sensor(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_TEST_DELETE"))

    resp = await client.delete("/sensors/TEMP_TEST_DELETE")
    assert resp.status_code == 204

    resp2 = await client.get("/sensors/TEMP_TEST_DELETE")
    assert resp2.status_code == 404


@pytest.mark.asyncio
async def test_disable_and_enable_sensor(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_TEST_TOGGLE"))

    resp_disable = await client.patch("/sensors/TEMP_TEST_TOGGLE/disable")
    assert resp_disable.status_code == 200
    assert resp_disable.json()["enabled"] is False

    resp_enable = await client.patch("/sensors/TEMP_TEST_TOGGLE/enable")
    assert resp_enable.status_code == 200
    assert resp_enable.json()["enabled"] is True


@pytest.mark.asyncio
async def test_list_sensors_filter_by_type(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_TEST_FILTER"))

    gas_payload = sample_sensor_payload("GAS_TEST_FILTER")
    gas_payload["sensor_type"] = "GAS"
    gas_payload["unit"] = "ppm"
    await client.post("/sensors", json=gas_payload)

    resp = await client.get("/sensors", params={"sensor_type": "GAS"})
    assert resp.status_code == 200
    data = resp.json()
    assert all(s["sensor_type"] == "GAS" for s in data)
    assert any(s["sensor_id"] == "GAS_TEST_FILTER" for s in data)
