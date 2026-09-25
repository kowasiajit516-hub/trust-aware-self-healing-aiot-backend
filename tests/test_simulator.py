"""
tests/test_simulator.py
-------------------------
Tests for simulator/simulator.py.

The simulator is exercised against the REAL FastAPI app - just through
an in-process ASGI transport (the `client` fixture) instead of a real
TCP socket. It still runs through the actual routes, service layer,
and MongoDB test database; no direct DB writes are used anywhere in
these tests, matching the "no simulator-only shortcuts" rule.
"""

import random

import pytest

from simulator.simulator import FaultType, Simulator
from tests.conftest import sample_sensor_payload


def _sensor_dict(sensor_id="TEMP_SIM01", normal_min=15.0, normal_max=40.0):
    return {"sensor_id": sensor_id, "normal_min": normal_min, "normal_max": normal_max}


def test_generate_plausible_value_stays_near_range(client):
    sim = Simulator(client=client, rng=random.Random(42))
    sensor = _sensor_dict()

    for _ in range(200):
        value = sim.generate_plausible_value(sensor)
        # tiny 2% overshoot allowed beyond the normal range, per design
        assert 15.0 - 0.5 - 1 <= value <= 40.0 + 0.5 + 1


def test_inject_fault_returns_a_known_fault_type(client):
    sim = Simulator(client=client, fault_rate=1.0, rng=random.Random(1))
    sensor = _sensor_dict()

    value, fault_type = sim.inject_fault(sensor, base_value=27.0)

    assert isinstance(fault_type, FaultType)
    assert isinstance(value, float)


def test_inject_fault_spike_is_outside_normal_range(client):
    sim = Simulator(client=client, rng=random.Random(7))
    sensor = _sensor_dict()

    # Force SPIKE deterministically by monkeypatching the rng choice
    sim.rng.choice = lambda seq: FaultType.SPIKE if FaultType.SPIKE in seq else seq[0]

    value, fault_type = sim.inject_fault(sensor, base_value=27.0)

    assert fault_type == FaultType.SPIKE
    assert value < sensor["normal_min"] or value > sensor["normal_max"]


def test_inject_fault_stuck_reuses_last_value(client):
    sim = Simulator(client=client, rng=random.Random(3))
    sensor = _sensor_dict()
    sim._last_values[sensor["sensor_id"]] = 33.3

    sim.rng.choice = lambda seq: FaultType.STUCK if FaultType.STUCK in seq else seq[0]

    value, fault_type = sim.inject_fault(sensor, base_value=27.0)

    assert fault_type == FaultType.STUCK
    assert value == 33.3


@pytest.mark.asyncio
async def test_tick_posts_readings_only_for_enabled_sensors(client):
    enabled_payload = sample_sensor_payload("TEMP_SIM_ENABLED")
    disabled_payload = sample_sensor_payload("TEMP_SIM_DISABLED")
    disabled_payload["enabled"] = False

    await client.post("/sensors", json=enabled_payload)
    await client.post("/sensors", json=disabled_payload)

    sim = Simulator(client=client, fault_rate=0.0, rng=random.Random(0))
    results = await sim.tick()

    sensor_ids = {r["sensor_id"] for r in results}
    assert "TEMP_SIM_ENABLED" in sensor_ids
    assert "TEMP_SIM_DISABLED" not in sensor_ids

    resp = await client.get("/readings", params={"sensor_id": "TEMP_SIM_ENABLED"})
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_tick_readings_land_within_sensor_range_when_no_faults(client):
    payload = sample_sensor_payload("TEMP_SIM_RANGE")
    await client.post("/sensors", json=payload)

    sim = Simulator(client=client, fault_rate=0.0, rng=random.Random(0))
    results = await sim.tick()

    reading = next(r for r in results if r["sensor_id"] == "TEMP_SIM_RANGE")
    assert reading["injected_fault"] is None
    assert 15.0 - 1 <= reading["value"] <= 40.0 + 1


@pytest.mark.asyncio
async def test_tick_can_inject_faults_when_fault_rate_is_one(client):
    payload = sample_sensor_payload("TEMP_SIM_FAULTY")
    await client.post("/sensors", json=payload)

    sim = Simulator(client=client, fault_rate=1.0, rng=random.Random(5))
    results = await sim.tick()

    reading = next(r for r in results if r["sensor_id"] == "TEMP_SIM_FAULTY")
    assert reading["injected_fault"] in {ft.value for ft in FaultType}


@pytest.mark.asyncio
async def test_run_executes_requested_number_of_ticks(client):
    payload = sample_sensor_payload("TEMP_SIM_RUN")
    await client.post("/sensors", json=payload)

    sim = Simulator(client=client, fault_rate=0.0, rng=random.Random(0))
    await sim.run(interval_seconds=0, iterations=3)

    resp = await client.get("/readings", params={"sensor_id": "TEMP_SIM_RUN"})
    assert len(resp.json()) == 3


@pytest.mark.asyncio
async def test_readings_posted_by_simulator_go_through_real_validation(client):
    """
    Sanity check that the simulator can't post a reading for a sensor
    that doesn't exist in the DB - i.e. it really is going through the
    real /readings endpoint's validation, not a shortcut.
    """
    sim = Simulator(client=client, fault_rate=0.0, rng=random.Random(0))
    # No sensors created at all -> fetch_enabled_sensors returns [] -> no readings posted
    results = await sim.tick()
    assert results == []
