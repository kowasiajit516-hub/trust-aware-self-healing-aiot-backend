"""
tests/test_actuators.py
-------------------------
Real tests for the Actuator CRUD + manual command API. Run with:

    pytest tests/test_actuators.py -v

Requires MongoDB to be running (uses a real, isolated test database -
see conftest.py).
"""

import pytest

from tests.conftest import sample_actuator_payload


@pytest.mark.asyncio
async def test_create_actuator(client):
    payload = sample_actuator_payload("FAN_TEST_CREATE")
    resp = await client.post("/actuators", json=payload)

    assert resp.status_code == 201
    data = resp.json()
    assert data["actuator_id"] == "FAN_TEST_CREATE"
    assert data["state"] == "OFF"
    assert data["mode"] == "AUTO"
    assert data["last_command"] is None


@pytest.mark.asyncio
async def test_create_actuator_duplicate_id_fails(client):
    payload = sample_actuator_payload("FAN_TEST_DUP")
    resp1 = await client.post("/actuators", json=payload)
    assert resp1.status_code == 201

    resp2 = await client.post("/actuators", json=payload)
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_list_actuators(client):
    await client.post("/actuators", json=sample_actuator_payload("FAN_TEST_LIST1"))
    led_payload = sample_actuator_payload("LED_TEST_LIST2")
    led_payload["actuator_type"] = "LED"
    await client.post("/actuators", json=led_payload)

    resp = await client.get("/actuators")
    assert resp.status_code == 200
    ids = [a["actuator_id"] for a in resp.json()]
    assert "FAN_TEST_LIST1" in ids
    assert "LED_TEST_LIST2" in ids


@pytest.mark.asyncio
async def test_get_actuator_not_found(client):
    resp = await client.get("/actuators/DOES_NOT_EXIST")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_actuator(client):
    await client.post("/actuators", json=sample_actuator_payload("FAN_TEST_UPDATE"))

    resp = await client.put(
        "/actuators/FAN_TEST_UPDATE", json={"location": "Lab 3"}
    )
    assert resp.status_code == 200
    assert resp.json()["location"] == "Lab 3"


@pytest.mark.asyncio
async def test_delete_actuator(client):
    await client.post("/actuators", json=sample_actuator_payload("FAN_TEST_DELETE"))

    resp = await client.delete("/actuators/FAN_TEST_DELETE")
    assert resp.status_code == 204

    resp2 = await client.get("/actuators/FAN_TEST_DELETE")
    assert resp2.status_code == 404


@pytest.mark.asyncio
async def test_manual_command_on(client):
    await client.post("/actuators", json=sample_actuator_payload("FAN_TEST_CMD_ON"))

    resp = await client.post(
        "/actuators/FAN_TEST_CMD_ON/command", json={"command": "ON"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "ON"
    assert data["mode"] == "MANUAL"
    assert data["last_command"] == "ON"


@pytest.mark.asyncio
async def test_manual_command_off(client):
    await client.post("/actuators", json=sample_actuator_payload("FAN_TEST_CMD_OFF"))
    await client.post("/actuators/FAN_TEST_CMD_OFF/command", json={"command": "ON"})

    resp = await client.post(
        "/actuators/FAN_TEST_CMD_OFF/command", json={"command": "OFF"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "OFF"
    assert data["mode"] == "MANUAL"


@pytest.mark.asyncio
async def test_manual_command_auto_returns_to_auto_mode(client):
    await client.post("/actuators", json=sample_actuator_payload("FAN_TEST_CMD_AUTO"))
    await client.post("/actuators/FAN_TEST_CMD_AUTO/command", json={"command": "ON"})

    resp = await client.post(
        "/actuators/FAN_TEST_CMD_AUTO/command", json={"command": "AUTO"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "AUTO"


@pytest.mark.asyncio
async def test_command_creates_actuator_event(client, test_db):
    await client.post("/actuators", json=sample_actuator_payload("FAN_TEST_EVENT"))
    await client.post(
        "/actuators/FAN_TEST_EVENT/command", json={"command": "ON"}
    )

    events = await test_db.actuator_events.find(
        {"actuator_id": "FAN_TEST_EVENT"}
    ).to_list(length=10)

    assert len(events) == 1
    assert events[0]["command"] == "ON"
    assert events[0]["previous_state"] == "OFF"
    assert events[0]["new_state"] == "ON"


@pytest.mark.asyncio
async def test_disable_and_enable_actuator(client):
    await client.post("/actuators", json=sample_actuator_payload("FAN_TEST_TOGGLE"))

    resp_disable = await client.patch("/actuators/FAN_TEST_TOGGLE/disable")
    assert resp_disable.status_code == 200
    assert resp_disable.json()["enabled"] is False

    resp_enable = await client.patch("/actuators/FAN_TEST_TOGGLE/enable")
    assert resp_enable.status_code == 200
    assert resp_enable.json()["enabled"] is True
