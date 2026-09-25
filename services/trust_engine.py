"""
services/trust_engine.py
--------------------------
DYNAMIC TRUST SCORE stage of the core pipeline:

    RAW SENSOR -> VALIDATION -> AI FAULT DETECTION -> DYNAMIC TRUST SCORE
        -> VIRTUAL SENSOR -> ...

Consumes Phase 3's fault-detection predictions (`is_fault`,
`anomaly_score`) and maintains a running, per-sensor `trust_score`
(0-100) that decays when faults are detected and recovers when healthy
readings resume again - the "SENSOR FAULT -> AI DETECTION -> TRUST
DECREASE -> ... -> SENSOR RECOVERY -> TRUST RECOVERY" loop from the
master plan's self-healing demo (section 4).

This stage does NOT do virtual sensing (Phase 5) and does NOT touch
actuators (Phase 6/9). It only:
  - updates the `trust_score` / `status` fields already present on the
    `sensors` collection (Phase 1 schema - previously static at
    100.0 / HEALTHY, never updated until now)
  - writes/upserts one live snapshot document per sensor in the generic
    `sensor_health` collection (indexed since Phase 1, unused until now)

Called as a best-effort step, after fault detection has already
succeeded, from services/reading_service.py. A trust-scoring failure
must never break `POST /readings` - same design principle Phase 3
established for fault detection itself (see reading_service docstring).
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.sensor import SensorStatus, now_utc
from utils.validators import serialize_mongo_doc

logger = logging.getLogger(__name__)

SENSORS_COLLECTION = "sensors"
SENSOR_HEALTH_COLLECTION = "sensor_health"

# ---------------------------------------------------------------------
# Default trust-model parameters. These are the FALLBACK values used
# only if services/settings_service.get_settings() can't be reached
# for some reason - the real, live values are read from the
# system_settings collection at the top of update_trust() every call,
# so an operator can tune these via PUT /settings without redeploying.
# Deliberately generic - never keyed by sensor_type, per the master
# plan's "no per-type hardcoded logic" rule.
# ---------------------------------------------------------------------
FAULT_PENALTY = 20.0
RECOVERY_GAIN = 5.0
TRUST_MIN = 0.0
TRUST_MAX = 100.0
FAULTY_THRESHOLD = 50.0
HEALTHY_THRESHOLD = 90.0


def _derive_status(trust_score: float, faulty_threshold: float, healthy_threshold: float) -> SensorStatus:
    if trust_score < faulty_threshold:
        return SensorStatus.FAULTY
    if trust_score < healthy_threshold:
        return SensorStatus.RECOVERING
    return SensorStatus.HEALTHY


async def update_trust(db: AsyncIOMotorDatabase, prediction: dict) -> dict:
    """
    Apply one fault-detection prediction to its sensor's running trust
    score, persist the new snapshot to `sensor_health`, and mirror the
    result onto the `sensors` collection's `trust_score`/`status`
    fields (so existing SensorResponse consumers see live values with
    no new endpoint required).

    `prediction` is the dict returned by
    services.fault_detection.score_reading - must contain at least
    sensor_id, is_fault, anomaly_score, reading_id, timestamp, id.

    Raises ValueError if the sensor no longer exists. Caller
    (reading_service._try_update_trust) treats this as best-effort and
    only logs it - it must never affect the POST /readings response.
    """
    sensor_id = prediction["sensor_id"]
    is_fault = bool(prediction["is_fault"])

    sensor = await db[SENSORS_COLLECTION].find_one({"sensor_id": sensor_id})
    if sensor is None:
        raise ValueError(f"Cannot update trust: unknown sensor_id='{sensor_id}'")

    # Live, operator-tunable parameters (PUT /settings) - falls back to
    # the module defaults above if settings can't be read for any reason,
    # so a settings-service hiccup can never break trust scoring.
    try:
        from services import settings_service

        settings = await settings_service.get_settings(db)
        fault_penalty = float(settings.get("fault_penalty", FAULT_PENALTY))
        recovery_gain = float(settings.get("recovery_gain", RECOVERY_GAIN))
        trust_min = float(settings.get("trust_min", TRUST_MIN))
        trust_max = float(settings.get("trust_max", TRUST_MAX))
        faulty_threshold = float(settings.get("faulty_threshold", FAULTY_THRESHOLD))
        healthy_threshold = float(settings.get("healthy_threshold", HEALTHY_THRESHOLD))
    except Exception:
        logger.exception("Failed to load live settings; using module defaults.")
        fault_penalty, recovery_gain = FAULT_PENALTY, RECOVERY_GAIN
        trust_min, trust_max = TRUST_MIN, TRUST_MAX
        faulty_threshold, healthy_threshold = FAULTY_THRESHOLD, HEALTHY_THRESHOLD

    health = await db[SENSOR_HEALTH_COLLECTION].find_one({"sensor_id": sensor_id})

    if health is None:
        # First prediction ever seen for this sensor - start from the
        # sensor's current trust_score (100.0 for a brand-new sensor,
        # per models/sensor.py's default).
        current_trust = float(sensor.get("trust_score", trust_max))
        consecutive_faults = 0
        consecutive_healthy = 0
        previous_status = sensor.get("status", SensorStatus.HEALTHY.value)
    else:
        current_trust = float(health.get("trust_score", trust_max))
        consecutive_faults = int(health.get("consecutive_faults", 0))
        consecutive_healthy = int(health.get("consecutive_healthy", 0))
        previous_status = health.get("status")

    if is_fault:
        new_trust = max(trust_min, current_trust - fault_penalty)
        consecutive_faults += 1
        consecutive_healthy = 0
    else:
        new_trust = min(trust_max, current_trust + recovery_gain)
        consecutive_healthy += 1
        consecutive_faults = 0

    new_status = _derive_status(new_trust, faulty_threshold, healthy_threshold)
    now = now_utc()

    health_doc = {
        "sensor_id": sensor_id,
        "trust_score": new_trust,
        "status": new_status.value,
        "consecutive_faults": consecutive_faults,
        "consecutive_healthy": consecutive_healthy,
        "last_prediction_id": prediction.get("id"),
        "last_is_fault": is_fault,
        "last_anomaly_score": prediction.get("anomaly_score"),
        "last_reading_timestamp": prediction.get("timestamp"),
        "updated_at": now,
    }

    await db[SENSOR_HEALTH_COLLECTION].update_one(
        {"sensor_id": sensor_id},
        {"$set": health_doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )

    # Mirror onto the sensors collection so the existing SensorResponse
    # (trust_score/status fields, live since Phase 1) reflects reality
    # without requiring every consumer to also hit /sensor-health.
    await db[SENSORS_COLLECTION].update_one(
        {"sensor_id": sensor_id},
        {"$set": {"trust_score": new_trust, "status": new_status.value, "updated_at": now}},
    )

    if previous_status != new_status.value:
        logger.info(
            "Sensor %s status changed: %s -> %s (trust_score=%.1f)",
            sensor_id, previous_status, new_status.value, new_trust,
        )
        await _try_record_fault_event(
            db, sensor_id, previous_status, new_status.value, new_trust, prediction
        )

    logger.info(
        "Trust updated sensor=%s is_fault=%s trust_score=%.1f status=%s",
        sensor_id, is_fault, new_trust, new_status.value,
    )

    return serialize_mongo_doc(health_doc)


async def _try_record_fault_event(
    db: AsyncIOMotorDatabase,
    sensor_id: str,
    previous_status: str,
    new_status: str,
    trust_score: float,
    prediction: dict,
) -> None:
    """
    Best-effort call into the Phase 6 self-healing engine to persist
    this status transition to `fault_events`. Never raises - a failure
    here must not break trust scoring, same rule every other
    downstream stage in this pipeline follows (see
    reading_service.py's _try_* helpers).
    """
    from services import self_healing

    try:
        await self_healing.record_transition_if_changed(
            db,
            sensor_id=sensor_id,
            previous_status=previous_status,
            new_status=new_status,
            trust_score=trust_score,
            trigger_reading_id=prediction.get("reading_id"),
            timestamp=prediction.get("timestamp") or now_utc(),
        )
    except Exception:
        logger.exception(
            "Self-healing fault-event recording failed for sensor %s; trust update was still stored.",
            sensor_id,
        )


async def get_health(db: AsyncIOMotorDatabase, sensor_id: str) -> dict | None:
    """Fetch the current sensor_health snapshot for one sensor, or None if it has no predictions yet."""
    doc = await db[SENSOR_HEALTH_COLLECTION].find_one({"sensor_id": sensor_id})
    if doc is None:
        return None
    return serialize_mongo_doc(doc)


async def list_health(db: AsyncIOMotorDatabase, limit: int = 100) -> list[dict]:
    """List sensor_health snapshots, most recently updated first."""
    cursor = (
        db[SENSOR_HEALTH_COLLECTION]
        .find({})
        .sort("updated_at", -1)
        .limit(limit)
    )
    return [serialize_mongo_doc(doc) async for doc in cursor]