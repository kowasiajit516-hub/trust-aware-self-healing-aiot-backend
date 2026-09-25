"""
services/actuator_service.py
------------------------------
Business logic for the Actuator entity. Routes call these functions
instead of touching the database directly - keeps routes thin and
logic testable/reusable.

Manual ON/OFF/AUTO commands are handled here too (Phase 1 scope:
just record state + mode changes; the actual automation DECISION
engine that drives AUTO mode from trusted sensor values is Phase 9).

Phase 8 adds one additive step to send_command: if the actuator
belongs to a real node (actuator.node_id is set), the command is also
queued for that node to pick up (services/node_command_service.py) and
a best-effort direct push is attempted (esp32/esp32_client.py). This
never changes send_command's Phase 1 response shape or its existing
writes to `actuators`/`actuator_events` - see _try_dispatch_to_node.
"""

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.actuator import (
    ActuatorCreate,
    ActuatorUpdate,
    ActuatorCommand,
    ActuatorMode,
    ActuatorState,
    now_utc,
)
from utils.validators import ensure_unique, ensure_exists, serialize_mongo_doc

logger = logging.getLogger(__name__)

COLLECTION = "actuators"
EVENTS_COLLECTION = "actuator_events"


async def create_actuator(db: AsyncIOMotorDatabase, payload: ActuatorCreate) -> dict:
    await ensure_unique(db, COLLECTION, "actuator_id", payload.actuator_id)

    now = now_utc()
    doc = payload.model_dump()
    doc.update(
        {
            "state": ActuatorState.OFF.value,
            "last_command": None,
            "controlling_sensor": None,
            "created_at": now,
            "updated_at": now,
        }
    )

    await db[COLLECTION].insert_one(doc)
    logger.info("Actuator created: %s", payload.actuator_id)
    return serialize_mongo_doc(doc)


async def list_actuators(
    db: AsyncIOMotorDatabase,
    actuator_type: str | None = None,
    enabled: bool | None = None,
) -> list[dict]:
    query: dict = {}
    if actuator_type is not None:
        query["actuator_type"] = actuator_type
    if enabled is not None:
        query["enabled"] = enabled

    cursor = db[COLLECTION].find(query).sort("created_at", 1)
    return [serialize_mongo_doc(doc) async for doc in cursor]


async def get_actuator(db: AsyncIOMotorDatabase, actuator_id: str) -> dict:
    doc = await ensure_exists(db, COLLECTION, "actuator_id", actuator_id)
    return serialize_mongo_doc(doc)


async def update_actuator(
    db: AsyncIOMotorDatabase, actuator_id: str, payload: ActuatorUpdate
) -> dict:
    await ensure_exists(db, COLLECTION, "actuator_id", actuator_id)

    update_data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not update_data:
        doc = await db[COLLECTION].find_one({"actuator_id": actuator_id})
        return serialize_mongo_doc(doc)

    update_data["updated_at"] = now_utc()

    await db[COLLECTION].update_one(
        {"actuator_id": actuator_id}, {"$set": update_data}
    )
    logger.info("Actuator updated: %s -> %s", actuator_id, list(update_data.keys()))

    doc = await db[COLLECTION].find_one({"actuator_id": actuator_id})
    return serialize_mongo_doc(doc)


async def delete_actuator(db: AsyncIOMotorDatabase, actuator_id: str) -> None:
    await ensure_exists(db, COLLECTION, "actuator_id", actuator_id)
    await db[COLLECTION].delete_one({"actuator_id": actuator_id})
    logger.info("Actuator deleted: %s", actuator_id)


async def send_command(
    db: AsyncIOMotorDatabase, actuator_id: str, command: ActuatorCommand
) -> dict:
    """
    Handle a manual command: ON, OFF, or AUTO.

    - ON / OFF -> switches actuator to MANUAL mode and sets that state.
    - AUTO     -> switches actuator back to AUTO mode; the automation
                  engine (Phase 9) will then resume driving its state
                  from trusted sensor values.

    Every command is recorded in actuator_events for history/audit,
    per the master plan's Actuator Event History requirement.

    Phase 8: if this actuator belongs to a real node, the command is
    additionally dispatched there (see _try_dispatch_to_node) - this
    is a best-effort side effect and cannot change anything below.
    """
    current = await ensure_exists(db, COLLECTION, "actuator_id", actuator_id)
    previous_state = current.get("state", ActuatorState.OFF.value)

    if command == ActuatorCommand.AUTO:
        new_mode = ActuatorMode.AUTO.value
        new_state = previous_state  # automation engine will decide next state later
        reason = "Switched to AUTO mode"
    else:
        new_mode = ActuatorMode.MANUAL.value
        new_state = command.value  # ON or OFF
        reason = f"Manual command: {command.value}"

    now = now_utc()
    await db[COLLECTION].update_one(
        {"actuator_id": actuator_id},
        {
            "$set": {
                "mode": new_mode,
                "state": new_state,
                "last_command": command.value,
                "updated_at": now,
            }
        },
    )

    await db[EVENTS_COLLECTION].insert_one(
        {
            "actuator_id": actuator_id,
            "command": command.value,
            "previous_state": previous_state,
            "new_state": new_state,
            "mode": new_mode,
            "reason": reason,
            "sensor_id": current.get("controlling_sensor"),
            "trusted_value": None,  # populated once trust engine exists (Phase 4+)
            "timestamp": now,
        }
    )

    logger.info(
        "Actuator %s command=%s state %s -> %s (mode=%s)",
        actuator_id, command.value, previous_state, new_state, new_mode,
    )

    await _try_dispatch_to_node(db, current, actuator_id, command.value)

    doc = await db[COLLECTION].find_one({"actuator_id": actuator_id})
    return serialize_mongo_doc(doc)


async def _try_dispatch_to_node(
    db: AsyncIOMotorDatabase, actuator: dict, actuator_id: str, command: str
) -> None:
    """
    Best-effort delivery to the actuator's real ESP32 node (Phase 8).
    Never raises and never touches send_command's return value or its
    writes to `actuators`/`actuator_events` (Phase 1 contract fully
    preserved) - if the actuator has no node_id, the node is offline,
    or the direct push fails for any reason, the command simply waits
    in node_commands for the node's next poll (or forever, if the
    actuator has no node at all - unchanged Phase 1-7 behavior).
    """
    node_id = actuator.get("node_id")
    if not node_id:
        return

    from services import node_command_service
    from esp32 import esp32_client

    try:
        queued = await node_command_service.enqueue_command(
            db, node_id, actuator_id, command
        )
    except Exception:
        logger.exception(
            "Failed to queue node command for actuator %s (node %s); "
            "the actuator command itself was still applied normally.",
            actuator_id, node_id,
        )
        return

    try:
        node = await db["nodes"].find_one({"node_id": node_id})
        if node:
            await esp32_client.push_command_to_node(
                node, actuator_id, command, queued["id"]
            )
    except Exception:
        logger.exception(
            "Best-effort direct push to node %s failed; command %s "
            "remains queued for the node's next poll.",
            node_id, queued.get("id"),
        )


async def list_actuator_events(
    db: AsyncIOMotorDatabase, actuator_id: str | None = None, limit: int = 100
) -> list[dict]:
    """List recorded actuator command/state-change events, most recent first."""
    query: dict = {}
    if actuator_id is not None:
        query["actuator_id"] = actuator_id

    cursor = (
        db[EVENTS_COLLECTION]
        .find(query)
        .sort([("timestamp", -1), ("_id", -1)])
        .limit(limit)
    )
    results = []
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        results.append(serialize_mongo_doc(doc))
    return results


async def get_latest_actuator_event(db: AsyncIOMotorDatabase, actuator_id: str) -> dict:
    """Get the most recent recorded event for a given actuator. 404 if none exist yet."""
    from fastapi import HTTPException, status as http_status

    await ensure_exists(db, COLLECTION, "actuator_id", actuator_id)
    doc = await db[EVENTS_COLLECTION].find_one(
        {"actuator_id": actuator_id}, sort=[("timestamp", -1), ("_id", -1)]
    )
    if doc is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"No actuator events found for actuator_id='{actuator_id}'",
        )
    doc["id"] = str(doc["_id"])
    return serialize_mongo_doc(doc)
