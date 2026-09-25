"""
tests/test_automation_rules.py
--------------------------------
Real tests for the Automation Rule CRUD API. Run with:

    pytest tests/test_automation_rules.py -v

Requires MongoDB to be running (uses a real, isolated test database -
see conftest.py). These rules are CRUD-only in Phase 1 - the actual
evaluation engine is Phase 9. We DO test that a rule cannot be created
pointing at a sensor/actuator that doesn't exist.
"""

import pytest

from tests.conftest import sample_sensor_payload, sample_actuator_payload


def sample_rule_payload(
    rule_id: str = "RULE_TEST_01",
    sensor_id: str = "TEMP_TEST_RULE",
    actuator_id: str = "FAN_TEST_RULE",
) -> dict:
    return {
        "rule_id": rule_id,
        "name": "Test fan rule",
        "sensor_id": sensor_id,
        "actuator_id": actuator_id,
        "condition": "GREATER_THAN_OR_EQUAL",
        "threshold": 28.0,
        "action": "ON",
        "reset_threshold": 26.0,
        "reset_action": "OFF",
        "enabled": True,
    }


async def _create_sensor_and_actuator(client, sensor_id: str, actuator_id: str):
    await client.post("/sensors", json=sample_sensor_payload(sensor_id))
    await client.post("/actuators", json=sample_actuator_payload(actuator_id))


@pytest.mark.asyncio
async def test_create_rule(client):
    await _create_sensor_and_actuator(client, "TEMP_TEST_RULE1", "FAN_TEST_RULE1")

    payload = sample_rule_payload("RULE_TEST_CREATE", "TEMP_TEST_RULE1", "FAN_TEST_RULE1")
    resp = await client.post("/automation-rules", json=payload)

    assert resp.status_code == 201
    data = resp.json()
    assert data["rule_id"] == "RULE_TEST_CREATE"
    assert data["threshold"] == 28.0


@pytest.mark.asyncio
async def test_create_rule_fails_if_sensor_missing(client):
    # Only create the actuator, not the sensor
    await client.post("/actuators", json=sample_actuator_payload("FAN_TEST_RULE2"))

    payload = sample_rule_payload("RULE_TEST_NOSENSOR", "TEMP_DOES_NOT_EXIST", "FAN_TEST_RULE2")
    resp = await client.post("/automation-rules", json=payload)

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_rule_fails_if_actuator_missing(client):
    # Only create the sensor, not the actuator
    await client.post("/sensors", json=sample_sensor_payload("TEMP_TEST_RULE3"))

    payload = sample_rule_payload("RULE_TEST_NOACT", "TEMP_TEST_RULE3", "FAN_DOES_NOT_EXIST")
    resp = await client.post("/automation-rules", json=payload)

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_rule_duplicate_id_fails(client):
    await _create_sensor_and_actuator(client, "TEMP_TEST_RULE4", "FAN_TEST_RULE4")
    payload = sample_rule_payload("RULE_TEST_DUP", "TEMP_TEST_RULE4", "FAN_TEST_RULE4")

    resp1 = await client.post("/automation-rules", json=payload)
    assert resp1.status_code == 201

    resp2 = await client.post("/automation-rules", json=payload)
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_list_rules_filtered_by_sensor(client):
    await _create_sensor_and_actuator(client, "TEMP_TEST_RULE5", "FAN_TEST_RULE5")
    payload = sample_rule_payload("RULE_TEST_FILTER", "TEMP_TEST_RULE5", "FAN_TEST_RULE5")
    await client.post("/automation-rules", json=payload)

    resp = await client.get("/automation-rules", params={"sensor_id": "TEMP_TEST_RULE5"})
    assert resp.status_code == 200
    data = resp.json()
    assert all(r["sensor_id"] == "TEMP_TEST_RULE5" for r in data)
    assert any(r["rule_id"] == "RULE_TEST_FILTER" for r in data)


@pytest.mark.asyncio
async def test_update_rule(client):
    await _create_sensor_and_actuator(client, "TEMP_TEST_RULE6", "FAN_TEST_RULE6")
    payload = sample_rule_payload("RULE_TEST_UPDATE", "TEMP_TEST_RULE6", "FAN_TEST_RULE6")
    await client.post("/automation-rules", json=payload)

    resp = await client.put(
        "/automation-rules/RULE_TEST_UPDATE", json={"threshold": 30.0}
    )
    assert resp.status_code == 200
    assert resp.json()["threshold"] == 30.0


@pytest.mark.asyncio
async def test_delete_rule(client):
    await _create_sensor_and_actuator(client, "TEMP_TEST_RULE7", "FAN_TEST_RULE7")
    payload = sample_rule_payload("RULE_TEST_DELETE", "TEMP_TEST_RULE7", "FAN_TEST_RULE7")
    await client.post("/automation-rules", json=payload)

    resp = await client.delete("/automation-rules/RULE_TEST_DELETE")
    assert resp.status_code == 204

    resp2 = await client.get("/automation-rules/RULE_TEST_DELETE")
    assert resp2.status_code == 404
