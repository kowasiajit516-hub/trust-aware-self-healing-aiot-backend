"""
services/self_healing.py
---------------------------
SELF-HEALING ENGINE (Phase 6) - the observability layer over the loop
already wired by Phases 3-5:

    RAW SENSOR -> AI FAULT DETECTION -> DYNAMIC TRUST SCORE
        -> VIRTUAL SENSOR -> TRUSTED VALUE

Phases 3/4/5 already compute every value in that chain
(services/fault_detection.py, services/trust_engine.py,
services/virtual_sensor.py, all called from
services/reading_service.py). This module does NOT redefine or
re-run any of that - it only detects when a sensor's SensorStatus
actually CHANGES (e.g. HEALTHY -> RECOVERING, RECOVERING -> FAULTY,
FAULTY -> RECOVERING, RECOVERING -> HEALTHY) and persists that
transition as one document in the generic `fault_events` collection,
turning the master plan's section 4 self-healing demo loop into
something queryable instead of only inferable by diffing
`sensor_health` snapshots over time.

Called from services/trust_engine.py, at the exact point it already
computes `previous_status` vs `new_status` to decide whether to log a
status change. Best-effort, same convention as every other downstream
stage in this pipeline: a failure here must never break trust scoring
or, transitively, POST /readings.

Does NOT touch automation_rules or actuators - that remains Phase 9's
scope, per the master plan's phase table.
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.fault_event import now_utc
from utils.validators import ensure_exists, serialize_mongo_doc

logger = logging.getLogger(__name__)

SENSORS_COLLECTION = "sensors"
FAULT_EVENTS_COLLECTION = "fault_events"


async def record_transition_if_changed(
    db: AsyncIOMotorDatabase,
    sensor_id: str,
    previous_status: str,
    new_status: str,
    trust_score: float,
    trigger_reading_id: str | None,
    timestamp,
) -> dict | None:
    """
    Persist one fault_events document IF AND ONLY IF the sensor's
    status actually changed. Returns the persisted event dict, or
    None if previous_status == new_status (the common case - most
    readings don't cross a status boundary).

    This is intentionally a pure "record what already happened"
    function - it never recomputes trust_score or status itself, it
    only logs the transition trust_engine.update_trust already
    derived, so there is exactly one source of truth for the trust
    model (rule #6 spirit: no duplicated business logic).
    """
    if previous_status == new_status:
        return None

    doc = {
        "sensor_id": sensor_id,
        "previous_status": previous_status,
        "new_status": new_status,
        "trust_score": trust_score,
        "trigger_reading_id": trigger_reading_id,
        "timestamp": timestamp,
        "created_at": now_utc(),
    }

    result = await db[FAULT_EVENTS_COLLECTION].insert_one(doc)
    doc["id"] = str(result.inserted_id)

    logger.info(
        "Fault event recorded sensor=%s %s -> %s (trust_score=%.1f)",
        sensor_id, previous_status, new_status, trust_score,
    )
    return serialize_mongo_doc(doc)


async def list_fault_events(
    db: AsyncIOMotorDatabase, sensor_id: str | None = None, limit: int = 100
) -> list[dict]:
    """List fault events, most recent first, optionally filtered by sensor_id."""
    query: dict = {}
    if sensor_id is not None:
        query["sensor_id"] = sensor_id

    cursor = (
        db[FAULT_EVENTS_COLLECTION]
        .find(query)
        .sort([("timestamp", -1), ("_id", -1)])
        .limit(limit)
    )
    results = []
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        results.append(serialize_mongo_doc(doc))
    return results


async def get_latest_fault_event(db: AsyncIOMotorDatabase, sensor_id: str) -> dict:
    """Get the most recent fault event for a given sensor. 404 if none exist yet."""
    from fastapi import HTTPException, status as http_status

    await ensure_exists(db, SENSORS_COLLECTION, "sensor_id", sensor_id)
    doc = await db[FAULT_EVENTS_COLLECTION].find_one(
        {"sensor_id": sensor_id}, sort=[("timestamp", -1), ("_id", -1)]
    )
    if doc is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"No fault events found for sensor_id='{sensor_id}'",
        )
    doc["id"] = str(doc["_id"])
    return serialize_mongo_doc(doc)