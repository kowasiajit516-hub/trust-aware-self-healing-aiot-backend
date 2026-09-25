"""
services/fault_injection_service.py
--------------------------------------
Backing logic for the dashboard's Fault Injection panel.

Reuses the exact fault-shape math from simulator/simulator.py's
Simulator.inject_fault for SPIKE/STUCK/DRIFT, driven only by the
sensor's own normal_min/normal_max (generic, per rule #6 - never a
per-sensor-type table), and extends it with three more real,
deterministic fault shapes (SUDDEN_OFFSET, FROZEN, COMMUNICATION).
MISSING is handled separately in inject_fault() since it submits no
reading at all - see models/fault_injection.py's FaultType docstring
for that type's real, documented limitation.

Every value-producing type is handed to
services/reading_service.create_reading() - the same function
POST /readings uses - so it goes through the real pipeline
(validation -> AI fault detection -> trust scoring -> virtual
sensing), never a shortcut straight into the database.
"""

from __future__ import annotations

import random

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.fault_injection import FaultType
from models.reading import SensorReadingCreate
from services import reading_service
from utils.validators import ensure_exists

SENSORS_COLLECTION = "sensors"

# Fixed sentinel used for COMMUNICATION faults - deliberately far outside
# any plausible sensor range, distinct from SPIKE (random large deviation)
# so it reads as "the sensor sent garbage/an error code", not "a real
# extreme reading".
COMMUNICATION_SENTINEL = -999.0

_rng = random.Random()


def generate_fault_value(sensor: dict, fault_type: FaultType, offset: float | None) -> float:
    """
    Compute a deliberately corrupted value for `sensor`. Only called
    for types that actually submit a reading (MISSING is handled
    separately in inject_fault()).
    """
    lo, hi = sensor["normal_min"], sensor["normal_max"]
    span = hi - lo if hi > lo else 1.0
    mid = (lo + hi) / 2

    if fault_type == FaultType.SPIKE:
        # sudden large deviation past a range edge
        direction = _rng.choice([-1, 1])
        edge = hi if direction > 0 else lo
        magnitude = offset if offset is not None else span * _rng.uniform(0.5, 2.0)
        value = edge + direction * abs(magnitude)

    elif fault_type == FaultType.SUDDEN_OFFSET:
        # smaller, more "believable" constant shift from the expected
        # midpoint - a calibration-error style fault, distinct from
        # SPIKE's wild deviation
        direction = _rng.choice([-1, 1])
        magnitude = offset if offset is not None else span * _rng.uniform(0.15, 0.35)
        value = mid + direction * abs(magnitude)

    elif fault_type in (FaultType.STUCK, FaultType.FROZEN):
        # both mean the same real-world symptom: the sensor stops
        # changing and repeats its last known value (or the midpoint
        # if no prior reading exists)
        value = sensor.get("_last_known_value", mid)

    elif fault_type == FaultType.DRIFT:
        direction = _rng.choice([-1, 1])
        magnitude = offset if offset is not None else span * _rng.uniform(0.3, 0.8)
        value = mid + direction * abs(magnitude)

    else:  # COMMUNICATION
        value = offset if offset is not None else COMMUNICATION_SENTINEL

    return round(value, 3)


async def inject_fault(
    db: AsyncIOMotorDatabase, sensor_id: str, fault_type: FaultType, offset: float | None
) -> tuple[float | None, dict | None, str | None]:
    """
    Look up the real sensor and either:
      - compute a corrupted value and post it through the real
        ingestion pipeline (all types except MISSING), or
      - submit nothing at all (MISSING - simulates a dropout).

    Returns (injected_value, reading_response, note). note is only
    set for MISSING, explaining the real limitation rather than
    implying the dropout visibly changed anything.
    """
    sensor = await ensure_exists(db, SENSORS_COLLECTION, "sensor_id", sensor_id)

    if fault_type == FaultType.MISSING:
        return (
            None,
            None,
            (
                "No reading was submitted for this sensor, simulating a dropout. "
                "This project has no background staleness/timeout watcher, so the "
                "sensor's status and trust score will NOT change from this alone - "
                "they only update in reaction to a new reading being scored."
            ),
        )

    last = await db["sensor_readings"].find_one(
        {"sensor_id": sensor_id}, sort=[("timestamp", -1), ("_id", -1)]
    )
    if last is not None:
        sensor["_last_known_value"] = last["value"]

    value = generate_fault_value(sensor, fault_type, offset)

    reading = await reading_service.create_reading(
        db, SensorReadingCreate(sensor_id=sensor_id, value=value)
    )
    return value, reading, None
