"""
services/sensor_service.py
---------------------------
Business logic for the Sensor entity. Routes call these functions
instead of touching the database directly - keeps routes thin and
logic testable/reusable.
"""

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.sensor import (
    SensorCreate,
    SensorUpdate,
    SensorStatus,
    now_utc,
)
from utils.validators import ensure_unique, ensure_exists, serialize_mongo_doc

logger = logging.getLogger(__name__)

COLLECTION = "sensors"


async def create_sensor(db: AsyncIOMotorDatabase, payload: SensorCreate) -> dict:
    await ensure_unique(db, COLLECTION, "sensor_id", payload.sensor_id)

    now = now_utc()
    doc = payload.model_dump()
    doc.update(
        {
            "status": SensorStatus.HEALTHY.value,
            "trust_score": 100.0,
            "created_at": now,
            "updated_at": now,
        }
    )

    await db[COLLECTION].insert_one(doc)
    logger.info("Sensor created: %s", payload.sensor_id)
    return serialize_mongo_doc(doc)


async def list_sensors(
    db: AsyncIOMotorDatabase,
    sensor_type: str | None = None,
    enabled: bool | None = None,
) -> list[dict]:
    query: dict = {}
    if sensor_type is not None:
        query["sensor_type"] = sensor_type
    if enabled is not None:
        query["enabled"] = enabled

    cursor = db[COLLECTION].find(query).sort("created_at", 1)
    return [serialize_mongo_doc(doc) async for doc in cursor]


async def get_sensor(db: AsyncIOMotorDatabase, sensor_id: str) -> dict:
    doc = await ensure_exists(db, COLLECTION, "sensor_id", sensor_id)
    return serialize_mongo_doc(doc)


async def update_sensor(
    db: AsyncIOMotorDatabase, sensor_id: str, payload: SensorUpdate
) -> dict:
    await ensure_exists(db, COLLECTION, "sensor_id", sensor_id)

    update_data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not update_data:
        # Nothing to update - just return the current document
        doc = await db[COLLECTION].find_one({"sensor_id": sensor_id})
        return serialize_mongo_doc(doc)

    update_data["updated_at"] = now_utc()

    await db[COLLECTION].update_one(
        {"sensor_id": sensor_id}, {"$set": update_data}
    )
    logger.info("Sensor updated: %s -> %s", sensor_id, list(update_data.keys()))

    doc = await db[COLLECTION].find_one({"sensor_id": sensor_id})
    return serialize_mongo_doc(doc)


async def delete_sensor(db: AsyncIOMotorDatabase, sensor_id: str) -> None:
    await ensure_exists(db, COLLECTION, "sensor_id", sensor_id)
    await db[COLLECTION].delete_one({"sensor_id": sensor_id})
    logger.info("Sensor deleted: %s", sensor_id)


async def set_sensor_enabled(
    db: AsyncIOMotorDatabase, sensor_id: str, enabled: bool
) -> dict:
    await ensure_exists(db, COLLECTION, "sensor_id", sensor_id)
    await db[COLLECTION].update_one(
        {"sensor_id": sensor_id},
        {"$set": {"enabled": enabled, "updated_at": now_utc()}},
    )
    logger.info("Sensor %s set enabled=%s", sensor_id, enabled)
    doc = await db[COLLECTION].find_one({"sensor_id": sensor_id})
    return serialize_mongo_doc(doc)
