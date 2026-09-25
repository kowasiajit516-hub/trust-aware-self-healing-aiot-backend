"""
services/reading_service.py
----------------------------
Business logic for raw sensor readings - the entry point of the core
pipeline:

    RAW SENSOR -> VALIDATION -> AI FAULT DETECTION -> DYNAMIC TRUST SCORE -> ...

Phase 2 scope: validate against a real, existing sensor and store the
raw reading exactly as received. Per the master plan's non-negotiable
rules, raw readings are never modified or deleted, even after a sensor
is later found faulty - they stay the permanent, distinguishable "raw"
record.

Phase 3 added AI FAULT DETECTION as a best-effort step after storage.
Phase 4 added DYNAMIC TRUST SCORE as a further best-effort step, run
only when fault detection actually produced a prediction. Phase 5 adds
VIRTUAL SENSOR / TRUSTED VALUE as a third best-effort step, run only
when trust scoring actually produced a health snapshot. Phase 9 adds
the AUTOMATION ENGINE as a fourth best-effort step, run only when a
trusted value was actually produced - closing the full core pipeline:

    RAW SENSOR -> VALIDATION -> AI FAULT DETECTION -> DYNAMIC TRUST SCORE
        -> VIRTUAL SENSOR -> TRUSTED VALUE -> AUTOMATION ENGINE -> ACTUATOR

All four downstream stages follow the same rule: a failure here must
NEVER break `POST /readings` or change its response shape - the
ingestion contract established in Phase 2 must survive every later
phase intact.
"""

import logging

from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from models.reading import SensorReadingCreate, now_utc
from utils.validators import ensure_exists, serialize_mongo_doc

logger = logging.getLogger(__name__)

COLLECTION = "sensor_readings"
SENSORS_COLLECTION = "sensors"


async def create_reading(db: AsyncIOMotorDatabase, payload: SensorReadingCreate) -> dict:
    """
    Store one raw sensor reading.

    Validates that sensor_id refers to a real, existing sensor (created
    via POST /sensors) before accepting the reading. This keeps the
    pipeline's entry point honest and prevents orphaned reading data,
    and is why the simulator (and real ESP32 nodes) must create
    sensors through the API before it can post readings for them.

    After the raw reading is safely stored, it is handed to the AI
    FAULT DETECTION stage (services/fault_detection.py) and, if that
    succeeds, on to the DYNAMIC TRUST SCORE stage
    (services/trust_engine.py) and, if THAT succeeds, on to the
    VIRTUAL SENSOR / TRUSTED VALUE stage (services/virtual_sensor.py)
    and, if THAT succeeds, on to the AUTOMATION ENGINE (Phase 9,
    services/automation.py). All four are downstream, best-effort
    steps - if no model has been trained yet, no automation rules
    exist for this sensor, or any step fails for any reason, ingestion
    still succeeds and returns the raw reading unchanged (Phase 2's
    response contract is preserved). This keeps "RAW SENSOR ->
    VALIDATION" honest and decoupled from every later stage in the
    master plan's core pipeline.
    """
    await ensure_exists(db, SENSORS_COLLECTION, "sensor_id", payload.sensor_id)

    doc = payload.model_dump()
    doc["timestamp"] = payload.timestamp or now_utc()

    result = await db[COLLECTION].insert_one(doc)
    doc["id"] = str(result.inserted_id)

    logger.info(
        "Reading stored for sensor %s: value=%s", payload.sensor_id, payload.value
    )
    reading = serialize_mongo_doc(doc)

    await _try_score_reading(db, reading)

    return reading


async def _try_score_reading(db: AsyncIOMotorDatabase, reading: dict) -> None:
    """
    Best-effort fault-detection scoring, followed by best-effort trust
    scoring if (and only if) a prediction was actually produced. Never
    breaks ingestion - see create_reading's docstring.
    """
    from services import fault_detection

    try:
        prediction = await fault_detection.score_reading(db, reading)
    except fault_detection.FaultDetectionUnavailable as exc:
        logger.warning("Skipping fault detection: %s", exc)
        return
    except Exception:
        logger.exception(
            "Fault detection scoring failed for sensor %s; reading was still stored.",
            reading.get("sensor_id"),
        )
        return

    await _try_update_trust(db, reading, prediction)


async def _try_update_trust(db: AsyncIOMotorDatabase, reading: dict, prediction: dict) -> None:
    """
    Best-effort dynamic trust scoring, followed by best-effort virtual
    sensing / trusted-value computation if (and only if) a health
    snapshot was actually produced. Never breaks ingestion.
    """
    from services import trust_engine

    try:
        health = await trust_engine.update_trust(db, prediction)
    except Exception:
        logger.exception(
            "Trust scoring failed for sensor %s; prediction was still stored.",
            prediction.get("sensor_id"),
        )
        return

    await _try_compute_trusted_value(db, reading, health)


async def _try_compute_trusted_value(db: AsyncIOMotorDatabase, reading: dict, health: dict) -> None:
    """
    Best-effort virtual sensing / trusted-value computation, followed
    by best-effort automation-rule evaluation (Phase 9) if (and only
    if) a trusted value was actually produced. Never breaks ingestion.
    """
    from services import virtual_sensor

    try:
        trusted_reading = await virtual_sensor.compute_trusted_value(db, reading, health["status"])
    except virtual_sensor.VirtualSensorUnavailable as exc:
        logger.warning("Skipping virtual sensing: %s", exc)
        return
    except Exception:
        logger.exception(
            "Virtual sensing failed for sensor %s; trust update was still stored.",
            reading.get("sensor_id"),
        )
        return

    await _try_evaluate_automation(db, trusted_reading)


async def _try_evaluate_automation(db: AsyncIOMotorDatabase, trusted_reading: dict) -> None:
    """
    Best-effort automation-rule evaluation (Phase 9) - the final stage
    of the core pipeline. Reads ONLY the just-computed trusted_value,
    never the raw reading (master plan rule #2). Never breaks
    ingestion - a failed or missing automation rule must not prevent
    the trusted value from having already been safely stored above.
    """
    from services import automation

    try:
        await automation.evaluate_and_apply(db, trusted_reading)
    except Exception:
        logger.exception(
            "Automation evaluation failed for sensor %s; trusted value was still stored.",
            trusted_reading.get("sensor_id"),
        )


async def list_readings(
    db: AsyncIOMotorDatabase,
    sensor_id: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """List raw readings, most recent first, optionally filtered by sensor_id."""
    query: dict = {}
    if sensor_id is not None:
        query["sensor_id"] = sensor_id

    # Secondary sort on _id (monotonically increasing within a process) as
    # a tiebreaker for readings inserted fast enough to share a timestamp.
    cursor = (
        db[COLLECTION]
        .find(query)
        .sort([("timestamp", -1), ("_id", -1)])
        .limit(limit)
    )

    results = []
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        results.append(serialize_mongo_doc(doc))
    return results


async def get_latest_reading(db: AsyncIOMotorDatabase, sensor_id: str) -> dict:
    """Get the most recent raw reading stored for a given sensor."""
    await ensure_exists(db, SENSORS_COLLECTION, "sensor_id", sensor_id)

    doc = await db[COLLECTION].find_one(
        {"sensor_id": sensor_id}, sort=[("timestamp", -1), ("_id", -1)]
    )
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No readings found for sensor_id='{sensor_id}'",
        )

    doc["id"] = str(doc["_id"])
    return serialize_mongo_doc(doc)
