"""
services/fault_detection.py
------------------------------
AI FAULT DETECTION stage of the core pipeline:

    RAW SENSOR -> VALIDATION -> AI FAULT DETECTION -> DYNAMIC TRUST SCORE -> ...

Loads the REAL, already-trained Isolation Forest saved by
ml/train_isolation_forest.py (joblib file) and uses it to score each
new raw reading as it arrives. No hardcoded predictions - if the model
file has not been trained yet, this service raises a clear error
rather than silently faking a result.

Trust scoring (Phase 4) and virtual sensing (Phase 5) are explicitly
NOT done here - this stage only produces a fault-detection score and
stores it in the generic `predictions` collection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from motor.motor_asyncio import AsyncIOMotorDatabase

from ml.feature_engineering import compute_feature_vector
from models.reading import now_utc
from utils.validators import ensure_exists, serialize_mongo_doc

logger = logging.getLogger(__name__)

READINGS_COLLECTION = "sensor_readings"
SENSORS_COLLECTION = "sensors"
PREDICTIONS_COLLECTION = "predictions"

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "models" / "isolation_forest.joblib"

# How many prior readings to pull for the causal rolling-window features
# (must be >= ml.feature_engineering.WINDOW; a small buffer is fine).
HISTORY_LOOKBACK = 20


class FaultDetectionUnavailable(RuntimeError):
    """Raised when the Isolation Forest model has not been trained/saved yet."""


_model_cache: dict[str, object] = {}


def _get_model():
    """
    Lazily load and cache the trained model. Cached by file mtime so a
    freshly retrained model is picked up without restarting the process.
    """
    if not MODEL_PATH.exists():
        raise FaultDetectionUnavailable(
            f"No trained model found at {MODEL_PATH}. "
            "Run: python ml/train_isolation_forest.py"
        )

    mtime = MODEL_PATH.stat().st_mtime
    cached = _model_cache.get("model")
    cached_mtime = _model_cache.get("mtime")

    if cached is None or cached_mtime != mtime:
        model = joblib.load(MODEL_PATH)
        _model_cache["model"] = model
        _model_cache["mtime"] = mtime
        logger.info("Loaded Isolation Forest model from %s", MODEL_PATH)

    return _model_cache["model"]


def reload_model() -> None:
    """Clear the cached model (used by tests to point at a fresh model file)."""
    _model_cache.clear()


async def _fetch_history_values(
    db: AsyncIOMotorDatabase, sensor_id: str, before_id: str, limit: int = HISTORY_LOOKBACK
) -> list[float]:
    """
    Fetch up to `limit` prior readings for this sensor, strictly older
    than `before_id` (the reading currently being scored), in
    chronological order (oldest first) - the same causal shape the
    training script uses.
    """
    from bson import ObjectId

    cursor = (
        db[READINGS_COLLECTION]
        .find({"sensor_id": sensor_id, "_id": {"$lt": ObjectId(before_id)}})
        .sort([("timestamp", -1), ("_id", -1)])
        .limit(limit)
    )
    values = [doc["value"] async for doc in cursor]
    values.reverse()  # now oldest-first
    return values


async def score_reading(db: AsyncIOMotorDatabase, reading: dict) -> dict:
    """
    Score one already-stored raw reading (as returned by
    reading_service.create_reading, i.e. it has an "id" field) and
    persist the result to the `predictions` collection.

    Raises FaultDetectionUnavailable if no trained model exists yet -
    callers decide whether that should fail the request or just be
    logged (see reading_service.create_reading for the ingestion path).
    """
    model = _get_model()  # raises FaultDetectionUnavailable if untrained

    sensor = await ensure_exists(db, SENSORS_COLLECTION, "sensor_id", reading["sensor_id"])
    history = await _fetch_history_values(db, reading["sensor_id"], reading["id"])

    features = compute_feature_vector(
        reading["value"], sensor["normal_min"], sensor["normal_max"], history
    )

    anomaly_score = float(model.decision_function([features])[0])
    is_fault = bool(model.predict([features])[0] == -1)

    doc = {
        "sensor_id": reading["sensor_id"],
        "reading_id": reading["id"],
        "value": reading["value"],
        "timestamp": reading["timestamp"],
        "anomaly_score": anomaly_score,
        "is_fault": is_fault,
        "model_version": _model_cache.get("mtime_iso") or str(MODEL_PATH.stat().st_mtime),
        "created_at": now_utc(),
    }

    result = await db[PREDICTIONS_COLLECTION].insert_one(doc)
    doc["id"] = str(result.inserted_id)

    logger.info(
        "Scored reading sensor=%s value=%s anomaly_score=%.4f is_fault=%s",
        reading["sensor_id"], reading["value"], anomaly_score, is_fault,
    )
    return serialize_mongo_doc(doc)


async def list_predictions(
    db: AsyncIOMotorDatabase, sensor_id: str | None = None, limit: int = 100
) -> list[dict]:
    query: dict = {}
    if sensor_id is not None:
        query["sensor_id"] = sensor_id

    cursor = (
        db[PREDICTIONS_COLLECTION]
        .find(query)
        .sort([("timestamp", -1), ("_id", -1)])
        .limit(limit)
    )
    results = []
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        results.append(serialize_mongo_doc(doc))
    return results


async def get_latest_prediction(db: AsyncIOMotorDatabase, sensor_id: str) -> dict:
    from fastapi import HTTPException, status

    await ensure_exists(db, SENSORS_COLLECTION, "sensor_id", sensor_id)
    doc = await db[PREDICTIONS_COLLECTION].find_one(
        {"sensor_id": sensor_id}, sort=[("timestamp", -1), ("_id", -1)]
    )
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No predictions found for sensor_id='{sensor_id}'",
        )
    doc["id"] = str(doc["_id"])
    return serialize_mongo_doc(doc)
