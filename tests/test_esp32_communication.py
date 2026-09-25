"""
tests/test_esp32_communication.py
------------------------------------
Phase 8: real ESP32 node communication - heartbeat, lazy offline
detection, and the node_commands queue a node polls and acknowledges.

Uses the same test_db/client fixtures and sample_*_payload helpers as
every other test file in this suite (see conftest.py).

CONFIRMED via routes/actuators.py: POST /actuators/{id}/command takes
a JSON body matching models.actuator.ActuatorCommandRequest, i.e.
{"command": "ON"} - not a raw string, not a query param (both of those
were wrong guesses before this file was available). All three call
sites below use that shape now.

NOTE ON HELPERS: sample_actuator_payload/sample_node_payload are
defined locally below rather than imported from conftest.py, because
this project's pytest.ini uses an import mode where `from conftest
import ...` isn't resolvable as a plain module import (conftest.py is
still auto-loaded by pytest itself for its actual fixtures - test_db,
client - just not importable by name). This keeps the file working
regardless of that config. sample_actuator_payload here is identical
to conftest.py's version.
"""

from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.asyncio


def sample_actuator_payload(actuator_id: str = "FAN_TEST_01") -> dict:
    """Local copy of conftest.py's helper - see NOTE ON HELPERS above."""
    return {
        "actuator_id": actuator_id,
        "name": "Test Cooling Fan",
        "actuator_type": "FAN",
        "location": "Test Lab",
        "node_id": None,
        "mode": "AUTO",
        "enabled": True,
    }


def sample_node_payload(node_id: str = "ESP32_TEST_01") -> dict:
    """Local copy of conftest.py's helper - see NOTE ON HELPERS above."""
    return {
        "node_id": node_id,
        "name": "Test Node",
        "location": "Test Lab",
        "ip_address": "192.168.1.99",
        "enabled": True,
    }


async def test_heartbeat_sets_online_and_last_seen(client):
    await client.post("/nodes", json=sample_node_payload())

    resp = await client.post("/nodes/ESP32_TEST_01/heartbeat", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ONLINE"
    assert body["last_seen"] is not None


async def test_heartbeat_can_update_ip_address(client):
    await client.post("/nodes", json=sample_node_payload())

    resp = await client.post(
        "/nodes/ESP32_TEST_01/heartbeat", json={"ip_address": "192.168.1.150"}
    )
    assert resp.status_code == 200
    assert resp.json()["ip_address"] == "192.168.1.150"


async def test_heartbeat_unknown_node_404(client):
    resp = await client.post("/nodes/DOES_NOT_EXIST/heartbeat", json={})
    assert resp.status_code == 404


async def test_pending_commands_empty_initially(client):
    await client.post("/nodes", json=sample_node_payload())

    resp = await client.get("/nodes/ESP32_TEST_01/commands/pending")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_pending_commands_unknown_node_404(client):
    resp = await client.get("/nodes/DOES_NOT_EXIST/commands/pending")
    assert resp.status_code == 404


async def test_manual_command_enqueues_pending_command_for_node(client):
    await client.post("/nodes", json=sample_node_payload())
    actuator = sample_actuator_payload("FAN_NODE_TEST_01")
    actuator["node_id"] = "ESP32_TEST_01"
    await client.post("/actuators", json=actuator)

    resp = await client.post("/actuators/FAN_NODE_TEST_01/command", json={"command": "ON"})
    assert resp.status_code == 200

    pending = await client.get("/nodes/ESP32_TEST_01/commands/pending")
    assert pending.status_code == 200
    commands = pending.json()
    assert len(commands) == 1
    assert commands[0]["actuator_id"] == "FAN_NODE_TEST_01"
    assert commands[0]["command"] == "ON"
    assert commands[0]["status"] == "PENDING"


async def test_actuator_without_node_id_does_not_enqueue_anything(client):
    """
    Regression guard for rule #11 (never break previous phases):
    actuators with node_id=None (the Phase 1-7 default used by every
    other test file's sample_actuator_payload) must behave exactly as
    before - Phase 8 must not require a node to exist for a manual
    command to work.
    """
    actuator = sample_actuator_payload("FAN_NO_NODE_01")
    await client.post("/actuators", json=actuator)

    resp = await client.post("/actuators/FAN_NO_NODE_01/command", json={"command": "ON"})
    assert resp.status_code == 200
    assert resp.json()["state"] == "ON"


async def test_acknowledge_command_marks_it_acked_and_clears_pending(client):
    await client.post("/nodes", json=sample_node_payload())
    actuator = sample_actuator_payload("FAN_NODE_TEST_02")
    actuator["node_id"] = "ESP32_TEST_01"
    await client.post("/actuators", json=actuator)
    await client.post("/actuators/FAN_NODE_TEST_02/command", json={"command": "ON"})

    pending = (await client.get("/nodes/ESP32_TEST_01/commands/pending")).json()
    command_id = pending[0]["id"]

    ack = await client.post(f"/nodes/ESP32_TEST_01/commands/{command_id}/ack")
    assert ack.status_code == 200
    assert ack.json()["status"] == "ACKED"
    assert ack.json()["acked_at"] is not None

    pending_after = (await client.get("/nodes/ESP32_TEST_01/commands/pending")).json()
    assert pending_after == []


async def test_acknowledge_unknown_command_404(client):
    await client.post("/nodes", json=sample_node_payload())
    resp = await client.post(
        "/nodes/ESP32_TEST_01/commands/000000000000000000000000/ack"
    )
    assert resp.status_code == 404


async def test_acknowledge_malformed_command_id_404(client):
    await client.post("/nodes", json=sample_node_payload())
    resp = await client.post("/nodes/ESP32_TEST_01/commands/not-a-valid-object-id/ack")
    assert resp.status_code == 404


async def test_stale_node_marked_offline_on_read(client, test_db):
    await client.post("/nodes", json=sample_node_payload())
    await client.post("/nodes/ESP32_TEST_01/heartbeat", json={})

    # Simulate a node that stopped heartbeating well past OFFLINE_AFTER_SECONDS
    stale_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    await test_db["nodes"].update_one(
        {"node_id": "ESP32_TEST_01"}, {"$set": {"last_seen": stale_time}}
    )

    resp = await client.get("/nodes/ESP32_TEST_01")
    assert resp.status_code == 200
    assert resp.json()["status"] == "OFFLINE"


async def test_recent_heartbeat_stays_online_on_read(client):
    await client.post("/nodes", json=sample_node_payload())
    await client.post("/nodes/ESP32_TEST_01/heartbeat", json={})

    resp = await client.get("/nodes/ESP32_TEST_01")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ONLINE"


async def test_list_nodes_also_applies_offline_check(client, test_db):
    await client.post("/nodes", json=sample_node_payload())
    await client.post("/nodes/ESP32_TEST_01/heartbeat", json={})

    stale_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    await test_db["nodes"].update_one(
        {"node_id": "ESP32_TEST_01"}, {"$set": {"last_seen": stale_time}}
    )

    resp = await client.get("/nodes")
    assert resp.status_code == 200
    node = next(n for n in resp.json() if n["node_id"] == "ESP32_TEST_01")
    assert node["status"] == "OFFLINE"