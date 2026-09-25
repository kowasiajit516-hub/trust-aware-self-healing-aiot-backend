"""
tests/test_nodes.py
---------------------
Real tests for the Node (ESP32) CRUD API. Run with:

    pytest tests/test_nodes.py -v

Requires MongoDB to be running (uses a real, isolated test database -
see conftest.py).
"""

import pytest


def sample_node_payload(node_id: str = "ESP32_TEST_01") -> dict:
    return {
        "node_id": node_id,
        "name": "Test Lab Node",
        "location": "Test Lab",
        "ip_address": "192.168.1.99",
        "enabled": True,
    }


@pytest.mark.asyncio
async def test_create_node(client):
    resp = await client.post("/nodes", json=sample_node_payload("ESP32_TEST_CREATE"))

    assert resp.status_code == 201
    data = resp.json()
    assert data["node_id"] == "ESP32_TEST_CREATE"
    assert data["status"] == "UNKNOWN"
    assert data["last_seen"] is None


@pytest.mark.asyncio
async def test_create_node_duplicate_id_fails(client):
    payload = sample_node_payload("ESP32_TEST_DUP")
    resp1 = await client.post("/nodes", json=payload)
    assert resp1.status_code == 201

    resp2 = await client.post("/nodes", json=payload)
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_list_nodes(client):
    await client.post("/nodes", json=sample_node_payload("ESP32_TEST_LIST1"))
    await client.post("/nodes", json=sample_node_payload("ESP32_TEST_LIST2"))

    resp = await client.get("/nodes")
    assert resp.status_code == 200
    ids = [n["node_id"] for n in resp.json()]
    assert "ESP32_TEST_LIST1" in ids
    assert "ESP32_TEST_LIST2" in ids


@pytest.mark.asyncio
async def test_get_node_not_found(client):
    resp = await client.get("/nodes/DOES_NOT_EXIST")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_node(client):
    await client.post("/nodes", json=sample_node_payload("ESP32_TEST_UPDATE"))

    resp = await client.put(
        "/nodes/ESP32_TEST_UPDATE", json={"ip_address": "192.168.1.100"}
    )
    assert resp.status_code == 200
    assert resp.json()["ip_address"] == "192.168.1.100"


@pytest.mark.asyncio
async def test_delete_node(client):
    await client.post("/nodes", json=sample_node_payload("ESP32_TEST_DELETE"))

    resp = await client.delete("/nodes/ESP32_TEST_DELETE")
    assert resp.status_code == 204

    resp2 = await client.get("/nodes/ESP32_TEST_DELETE")
    assert resp2.status_code == 404
