"""
services/automation_rule_service.py
-------------------------------------
Business logic for the Automation Rule entity. Phase 1 scope: CRUD
only. The actual rule EVALUATION engine (reading trusted sensor
values and driving actuator state) is implemented in Phase 9, after
the self-healing pipeline (Phases 3-6) exists.

We DO validate here that sensor_id and actuator_id actually exist,
since a rule pointing at a non-existent sensor/actuator would be a
silent bug later.
"""

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.automation_rule import (
    AutomationRuleCreate,
    AutomationRuleUpdate,
    now_utc,
)
from utils.validators import ensure_unique, ensure_exists, serialize_mongo_doc

logger = logging.getLogger(__name__)

COLLECTION = "automation_rules"


async def create_rule(db: AsyncIOMotorDatabase, payload: AutomationRuleCreate) -> dict:
    await ensure_unique(db, COLLECTION, "rule_id", payload.rule_id)

    # Fail fast if the rule references a sensor/actuator that doesn't exist.
    await ensure_exists(db, "sensors", "sensor_id", payload.sensor_id)
    await ensure_exists(db, "actuators", "actuator_id", payload.actuator_id)

    now = now_utc()
    doc = payload.model_dump()
    doc.update({"created_at": now, "updated_at": now})

    await db[COLLECTION].insert_one(doc)
    logger.info(
        "Automation rule created: %s (%s -> %s)",
        payload.rule_id, payload.sensor_id, payload.actuator_id,
    )
    return serialize_mongo_doc(doc)


async def list_rules(
    db: AsyncIOMotorDatabase,
    sensor_id: str | None = None,
    actuator_id: str | None = None,
    enabled: bool | None = None,
) -> list[dict]:
    query: dict = {}
    if sensor_id is not None:
        query["sensor_id"] = sensor_id
    if actuator_id is not None:
        query["actuator_id"] = actuator_id
    if enabled is not None:
        query["enabled"] = enabled

    cursor = db[COLLECTION].find(query).sort("created_at", 1)
    return [serialize_mongo_doc(doc) async for doc in cursor]


async def get_rule(db: AsyncIOMotorDatabase, rule_id: str) -> dict:
    doc = await ensure_exists(db, COLLECTION, "rule_id", rule_id)
    return serialize_mongo_doc(doc)


async def update_rule(
    db: AsyncIOMotorDatabase, rule_id: str, payload: AutomationRuleUpdate
) -> dict:
    await ensure_exists(db, COLLECTION, "rule_id", rule_id)

    update_data = {k: v for k, v in payload.model_dump().items() if v is not None}

    # If sensor_id/actuator_id are being changed, validate they exist too.
    if "sensor_id" in update_data:
        await ensure_exists(db, "sensors", "sensor_id", update_data["sensor_id"])
    if "actuator_id" in update_data:
        await ensure_exists(db, "actuators", "actuator_id", update_data["actuator_id"])

    if not update_data:
        doc = await db[COLLECTION].find_one({"rule_id": rule_id})
        return serialize_mongo_doc(doc)

    update_data["updated_at"] = now_utc()

    await db[COLLECTION].update_one({"rule_id": rule_id}, {"$set": update_data})
    logger.info("Automation rule updated: %s -> %s", rule_id, list(update_data.keys()))

    doc = await db[COLLECTION].find_one({"rule_id": rule_id})
    return serialize_mongo_doc(doc)


async def delete_rule(db: AsyncIOMotorDatabase, rule_id: str) -> None:
    await ensure_exists(db, COLLECTION, "rule_id", rule_id)
    await db[COLLECTION].delete_one({"rule_id": rule_id})
    logger.info("Automation rule deleted: %s", rule_id)
