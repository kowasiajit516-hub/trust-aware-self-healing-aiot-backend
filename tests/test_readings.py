"""
tests/test_readings.py
------------------------
Tests for POST/GET /readings - the ingestion entry point of the core
pipeline (RAW SENSOR -> VALIDATION -> ...).
"""

import pytest

from tests.conftest import sample_sensor_payload


@pytest.mark.asyncio
async def test_create_reading_success(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_R01"))

    resp = await client.post("/readings", json={"sensor_id": "TEMP_R01", "value": 27.4})

    assert resp.status_code == 201
    body = resp.json()
    assert body["sensor_id"] == "TEMP_R01"
    assert body["value"] == 27.4
    assert "id" in body and body["id"]
    assert "timestamp" in body and body["timestamp"]



@pytest.mark.asyncio
async def test_create_reading_defaults_timestamp_when_omitted(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_R02"))

    resp = await client.post("/readings", json={"sensor_id": "TEMP_R02", "value": 22.0})

    assert resp.status_code == 201
    assert resp.json()["timestamp"] is not None



@pytest.mark.asyncio
async def test_create_reading_respects_explicit_timestamp(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_R03"))

    ts = "2026-01-01T00:00:00Z"
    resp = await client.post(
        "/readings",
        json={"sensor_id": "TEMP_R03", "value": 20.0, "timestamp": ts},
    )

    assert resp.status_code == 201
    assert resp.json()["timestamp"].startswith("2026-01-01T00:00:00")



@pytest.mark.asyncio
async def test_create_reading_unknown_sensor_404(client):
    resp = await client.post("/readings", json={"sensor_id": "DOES_NOT_EXIST", "value": 1.0})
    assert resp.status_code == 404



@pytest.mark.asyncio
async def test_list_readings_filtered_by_sensor(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_R04"))
    await client.post("/sensors", json=sample_sensor_payload("TEMP_R05"))

    for v in (10.0, 11.0, 12.0):
        await client.post("/readings", json={"sensor_id": "TEMP_R04", "value": v})
    await client.post("/readings", json={"sensor_id": "TEMP_R05", "value": 99.0})

    resp = await client.get("/readings", params={"sensor_id": "TEMP_R04"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    assert all(r["sensor_id"] == "TEMP_R04" for r in body)



@pytest.mark.asyncio
async def test_list_readings_most_recent_first(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_R06"))

    for v in (1.0, 2.0, 3.0):
        await client.post("/readings", json={"sensor_id": "TEMP_R06", "value": v})

    resp = await client.get("/readings", params={"sensor_id": "TEMP_R06"})
    values = [r["value"] for r in resp.json()]

    assert values[0] == 3.0  # most recently inserted comes first



@pytest.mark.asyncio
async def test_list_readings_respects_limit(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_R07"))
    for v in range(5):
        await client.post("/readings", json={"sensor_id": "TEMP_R07", "value": float(v)})

    resp = await client.get(
        "/readings", params={"sensor_id": "TEMP_R07", "limit": 2}
    )

    assert resp.status_code == 200
    assert len(resp.json()) == 2



@pytest.mark.asyncio
async def test_get_latest_reading(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_R08"))
    await client.post("/readings", json={"sensor_id": "TEMP_R08", "value": 5.0})
    await client.post("/readings", json={"sensor_id": "TEMP_R08", "value": 6.0})

    resp = await client.get("/readings/TEMP_R08/latest")

    assert resp.status_code == 200
    assert resp.json()["value"] == 6.0



@pytest.mark.asyncio
async def test_get_latest_reading_no_data_yet_404(client):
    await client.post("/sensors", json=sample_sensor_payload("TEMP_R09"))

    resp = await client.get("/readings/TEMP_R09/latest")

    assert resp.status_code == 404



@pytest.mark.asyncio
async def test_get_latest_reading_unknown_sensor_404(client):
    resp = await client.get("/readings/DOES_NOT_EXIST/latest")
    assert resp.status_code == 404
