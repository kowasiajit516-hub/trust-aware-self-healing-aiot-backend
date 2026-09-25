"""
tests/conftest.py
------------------
Shared pytest fixtures for all backend tests.

Uses a SEPARATE test database (trust_aware_iot_test) so tests never
touch or pollute your real development data. The test database is
dropped clean before each test session and after it finishes.
"""

import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from motor.motor_asyncio import AsyncIOMotorClient

from config import settings
from main import app
from database import mongo_conn, create_indexes

TEST_DB_NAME = "trust_aware_iot_test"


@pytest_asyncio.fixture(scope="function")
async def test_db():
    """
    Provide a clean, isolated test database for each test function.
    Points the app's shared mongo_conn at the test DB for the duration
    of the test, then drops all test collections afterward.
    """
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[TEST_DB_NAME]

    # Point the app's global connection at the test database
    mongo_conn.client = client
    mongo_conn.db = db
    await create_indexes()

    yield db

    # Cleanup: drop every collection used by the app so tests stay isolated
    collections = [
        "sensors", "actuators", "nodes", "sensor_readings", "sensor_health",
        "predictions", "trusted_readings", "fault_events", "actuator_events",
        "automation_rules", "users", "password_reset_tokens", "node_commands",
    ]
    for name in collections:
        await db[name].delete_many({})

    client.close()


@pytest_asyncio.fixture(scope="function")
async def client(test_db):
    """
    Provide an async HTTP client wired directly to the FastAPI app
    (no real network socket needed) for calling endpoints in tests.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def sample_sensor_payload(sensor_id: str = "TEMP_TEST_01") -> dict:
    """Reusable valid sensor payload for tests."""
    return {
        "sensor_id": sensor_id,
        "name": "Test Room Temperature",
        "sensor_type": "TEMPERATURE",
        "unit": "°C",
        "location": "Test Lab",
        "node_id": None,
        "normal_min": 15.0,
        "normal_max": 40.0,
        "sampling_interval_seconds": 5,
        "enabled": True,
    }


def sample_actuator_payload(actuator_id: str = "FAN_TEST_01") -> dict:
    """Reusable valid actuator payload for tests."""
    return {
        "actuator_id": actuator_id,
        "name": "Test Cooling Fan",
        "actuator_type": "FAN",
        "location": "Test Lab",
        "node_id": None,
        "mode": "AUTO",
        "enabled": True,
    }


def sample_user_payload(email: str = "test.user@example.com") -> dict:
    """Reusable valid registration payload for auth tests."""
    return {
        "full_name": "Test User",
        "email": email,
        "password": "StrongPass123",
    }


def sample_node_payload(node_id: str = "ESP32_TEST_01") -> dict:
    """Reusable valid node payload for tests (Phase 8)."""
    return {
        "node_id": node_id,
        "name": "Test Node",
        "location": "Test Lab",
        "ip_address": "192.168.1.99",
        "enabled": True,
    }
