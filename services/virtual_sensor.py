"""
services/virtual_sensor.py
----------------------------
VIRTUAL SENSOR + TRUSTED VALUE stages of the core pipeline:

    ... -> DYNAMIC TRUST SCORE -> VIRTUAL SENSOR -> TRUSTED VALUE -> ...

Consumes Phase 4's derived SensorStatus for a reading and produces the
TRUSTED VALUE for it, written to the generic `trusted_readings`
collection (schema/index defined since Phase 1, unused until now):

  - status == HEALTHY
      -> the raw reading IS the trusted value (source="RAW"). The
         sensor has already proven itself trustworthy; nothing to
         predict.
  - status != HEALTHY (RECOVERING or FAULTY)
      -> the raw value is NOT trusted. A REAL, trained sklearn
         RandomForestRegressor (ml/train_random_forest.py) predicts a
         plausible virtual value to stand in for it (source="VIRTUAL").

No data leakage (master plan rule #5): the model NEVER sees the
current, untrusted raw value as an input feature - only that sensor's
own prior TRUSTED history (pulled from THIS collection, so a fault's
predicted values feed the *next* prediction's lag features instead of
letting the raw spike propagate) plus generic time-of-day features.
See ml/virtual_sensor_features.py for the full explanation.

This stage does NOT wire up automation/actuators (Phase 6/9) - it only
produces and persists the trusted value for each reading.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from motor.motor_asyncio import AsyncIOMotorDatabase

from ml.virtual_sensor_features import compute_virtual_features, denormalize
from models.sensor import SensorStatus
from models.trusted_reading import now_utc
from utils.validators import ensure_exists, serialize_mongo_doc

logger = logging.getLogger(__name__)

SENSORS_COLLECTION = "sensors"
TRUSTED_READINGS_COLLECTION = "trusted_readings"

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "models" / "random_forest.joblib"

# How many prior TRUSTED readings to pull for lag/rolling features
# (must be >= ml.virtual_sensor_features.WINDOW; a small buffer is fine).
HISTORY_LOOKBACK = 20

SOURCE_RAW = "RAW"
SOURCE_VIRTUAL = "VIRTUAL"


class VirtualSensorUnavailable(RuntimeError):
    """Raised when the RandomForestRegressor has not been trained/saved yet."""


_model_cache: dict[str, object] = {}


def _get_model():
    """
    Lazily load and cache the trained model. Cached by file mtime so a
    freshly retrained model is picked up without restarting the process.
    Mirrors services/fault_detection.py's _get_model exactly.
    """
    if not MODEL_PATH.exists():
        raise VirtualSensorUnavailable(
            f"No trained model found at {MODEL_PATH}. "
            "Run: python ml/train_random_forest.py"
        )

    mtime = MODEL_PATH.stat().st_mtime
    cached = _model_cache.get("model")
    cached_mtime = _model_cache.get("mtime")

    if cached is None or cached_mtime != mtime:
        model = joblib.load(MODEL_PATH)
        _model_cache["model"] = model
        _model_cache["mtime"] = mtime
        logger.info("Loaded RandomForestRegressor model from %s", MODEL_PATH)

    return _model_cache["model"]


def reload_model() -> None:
    """Clear the cached model (used by tests to point at a fresh model file)."""
    _model_cache.clear()


async def _fetch_trusted_history(
    db: AsyncIOMotorDatabase, sensor_id: str, before_timestamp, limit: int = HISTORY_LOOKBACK
) -> list[float]:
    """
    Fetch up to `limit` prior TRUSTED values for this sensor, strictly
    older than `before_timestamp`, in chronological order (oldest
    first) - deliberately reads from `trusted_readings` (not raw
    `sensor_readings`), so a previous fault's predicted virtual value -
    not the raw spike it replaced - is what feeds the next prediction.
    """
    cursor = (
        db[TRUSTED_READINGS_COLLECTION]
        .find({"sensor_id": sensor_id, "timestamp": {"$lt": before_timestamp}})
        .sort([("timestamp", -1), ("_id", -1)])
        .limit(limit)
    )
    values = [doc["trusted_value"] async for doc in cursor]
    values.reverse()  # now oldest-first
    return values


async def compute_trusted_value(db: AsyncIOMotorDatabase, reading: dict, status: str) -> dict:
    """
    Core Phase 5 entrypoint. `reading` is the raw stored reading dict
    (has id, sensor_id, value, timestamp - the same shape
    services/fault_detection.py consumes). `status` is the sensor's
    CURRENT SensorStatus value string, as just derived by Phase 4's
    trust_engine.update_trust for this exact reading.

    Persists and returns the resulting trusted_readings document.

    Raises ValueError for an unknown sensor_id, and
    VirtualSensorUnavailable if the sensor is untrusted but no trained
    model exists yet - callers (reading_service) decide whether that
    should fail the request or just be logged, same pattern as
    fault_detection.FaultDetectionUnavailable.
    """
    sensor_id = reading["sensor_id"]
    sensor = await db[SENSORS_COLLECTION].find_one({"sensor_id": sensor_id})
    if sensor is None:
        raise ValueError(f"Cannot compute trusted value: unknown sensor_id='{sensor_id}'")

    is_trusted = status == SensorStatus.HEALTHY.value

    if is_trusted:
        # Sensor has proven itself healthy - the raw value already IS
        # the trusted value. No model needed at all.
        trusted_value = reading["value"]
        source = SOURCE_RAW
    else:
        model = _get_model()  # raises VirtualSensorUnavailable if untrained
        history = await _fetch_trusted_history(db, sensor_id, reading["timestamp"])
        features = compute_virtual_features(
            reading["timestamp"], sensor["normal_min"], sensor["normal_max"], history
        )
        predicted_normalized = float(model.predict([features])[0])
        trusted_value = denormalize(predicted_normalized, sensor["normal_min"], sensor["normal_max"])
        source = SOURCE_VIRTUAL

    doc = {
        "sensor_id": sensor_id,
        "reading_id": reading["id"],
        "raw_value": reading["value"],
        "trusted_value": trusted_value,
        "source": source,
        "status_at_prediction": status,
        "timestamp": reading["timestamp"],
        "created_at": now_utc(),
    }

    result = await db[TRUSTED_READINGS_COLLECTION].insert_one(doc)
    doc["id"] = str(result.inserted_id)

    logger.info(
        "Trusted value sensor=%s status=%s source=%s raw=%s trusted=%.4f",
        sensor_id, status, source, reading["value"], trusted_value,
    )
    return serialize_mongo_doc(doc)


async def list_trusted(
    db: AsyncIOMotorDatabase, sensor_id: str | None = None, limit: int = 100
) -> list[dict]:
    query: dict = {}
    if sensor_id is not None:
        query["sensor_id"] = sensor_id

    cursor = (
        db[TRUSTED_READINGS_COLLECTION]
        .find(query)
        .sort([("timestamp", -1), ("_id", -1)])
        .limit(limit)
    )
    results = []
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        results.append(serialize_mongo_doc(doc))
    return results


async def get_latest_trusted(db: AsyncIOMotorDatabase, sensor_id: str) -> dict:
    from fastapi import HTTPException, status as http_status

    await ensure_exists(db, SENSORS_COLLECTION, "sensor_id", sensor_id)
    doc = await db[TRUSTED_READINGS_COLLECTION].find_one(
        {"sensor_id": sensor_id}, sort=[("timestamp", -1), ("_id", -1)]
    )
    if doc is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"No trusted readings found for sensor_id='{sensor_id}'",
        )
    doc["id"] = str(doc["_id"])
    return serialize_mongo_doc(doc)
