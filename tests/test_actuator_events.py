"""
tests/test_actuator_events.py
-------------------------------
Tests for the read-only GET /actuator-events endpoint (Phase 7 backend
fix). Run with:

    pytest tests/test_actuator_events.py -v

Requires MongoDB to be running (uses a real, isolated test database -
see conftest.py).
"""

import pytest

from tests.conftest import sample_actuator_payload


@pytest.mark.asyncio
async def test_no_events_before_any_command(client):
    await client.post("/actuators", json=sample_actuator_payload("FAN_EVT_NONE"))

    resp = await client.get("/actuator-events/FAN_EVT_NONE/latest")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_command_creates_event(client):
    await client.post("/actuators", json=sample_actuator_payload("FAN_EVT_CMD"))

    resp = await client.post("/actuators/FAN_EVT_CMD/command", json={"command": "ON"})
    assert resp.status_code == 200
    assert resp.json()["state"] == "ON"

    latest = await client.get("/actuator-events/FAN_EVT_CMD/latest")
    assert latest.status_code == 200
    data = latest.json()
    assert data["actuator_id"] == "FAN_EVT_CMD"
    assert data["command"] == "ON"
    assert data["previous_state"] == "OFF"
    assert data["new_state"] == "ON"
    assert data["mode"] == "MANUAL"


@pytest.mark.asyncio
async def test_list_filters_by_actuator_and_orders_most_recent_first(client):
    led_payload = sample_actuator_payload("LED_EVT_LIST")
    led_payload["actuator_type"] = "LED"

    await client.post("/actuators", json=sample_actuator_payload("FAN_EVT_LIST"))
    await client.post("/actuators", json=led_payload)

    await client.post("/actuators/FAN_EVT_LIST/command", json={"command": "ON"})
    await client.post("/actuators/FAN_EVT_LIST/command", json={"command": "OFF"})
    await client.post("/actuators/LED_EVT_LIST/command", json={"command": "ON"})

    resp = await client.get("/actuator-events", params={"actuator_id": "FAN_EVT_LIST"})
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) == 2
    assert all(e["actuator_id"] == "FAN_EVT_LIST" for e in events)
    # most recent first: last command sent (OFF) should come before the first (ON)
    assert events[0]["command"] == "OFF"
    assert events[1]["command"] == "ON"


@pytest.mark.asyncio
async def test_auto_command_switches_mode_without_state_change_but_still_logs(client):
    await client.post("/actuators", json=sample_actuator_payload("FAN_EVT_AUTO"))
    await client.post("/actuators/FAN_EVT_AUTO/command", json={"command": "ON"})

    resp = await client.post("/actuators/FAN_EVT_AUTO/command", json={"command": "AUTO"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "AUTO"

    latest = await client.get("/actuator-events/FAN_EVT_AUTO/latest")
    assert latest.status_code == 200
    assert latest.json()["command"] == "AUTO"
    assert latest.json()["mode"] == "AUTO"
