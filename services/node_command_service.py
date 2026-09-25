"""
services/node_command_service.py
----------------------------------
Business logic for the node_commands collection - the queue a real
ESP32 node polls (GET /nodes/{node_id}/commands/pending) to pick up
manual actuator commands issued through POST /actuators/{id}/command,
and acknowledges (POST /nodes/{node_id}/commands/{command_id}/ack)
once actually applied.

Phase 8 addition. Rows are appended by actuator_service.send_command
as a best-effort side effect - a failure here must never break that
endpoint's existing Phase 1 response contract (see the try/except in
actuator_service._try_dispatch_to_node). Rows are never deleted, only
transitioned PENDING -> ACKED, so this also serves as a delivery audit
trail.
"""

import logging

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException, status as http_status
from motor.motor_asyncio import AsyncIOMotorDatabase

from models.node import now_utc
from models.node_command import NodeCommandStatus
from utils.validators import serialize_mongo_doc

logger = logging.getLogger(__name__)

COLLECTION = "node_commands"


async def enqueue_command(
    db: AsyncIOMotorDatabase, node_id: str, actuator_id: str, command: str
) -> dict:
    """Queue one actuator command for a node to pick up."""
    doc = {
        "node_id": node_id,
        "actuator_id": actuator_id,
        "command": command,
        "status": NodeCommandStatus.PENDING.value,
        "created_at": now_utc(),
        "acked_at": None,
    }
    result = await db[COLLECTION].insert_one(doc)
    doc["id"] = str(result.inserted_id)
    logger.info(
        "Queued node command: node=%s actuator=%s command=%s id=%s",
        node_id, actuator_id, command, doc["id"],
    )
    return serialize_mongo_doc(doc)


async def list_pending_commands(db: AsyncIOMotorDatabase, node_id: str) -> list[dict]:
    """
    List PENDING commands for a node, oldest first (FIFO), for the
    ESP32 firmware's poll loop. This does NOT mark them delivered -
    the firmware must call acknowledge_command once it has actually
    applied each one, so a dropped HTTP response on the way back to
    the node never silently loses a command.
    """
    cursor = (
        db[COLLECTION]
        .find({"node_id": node_id, "status": NodeCommandStatus.PENDING.value})
        .sort([("created_at", 1), ("_id", 1)])
    )
    results = []
    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        results.append(serialize_mongo_doc(doc))
    return results


async def acknowledge_command(
    db: AsyncIOMotorDatabase, node_id: str, command_id: str
) -> dict:
    """Mark a queued command as ACKED once the node has applied it."""
    try:
        object_id = ObjectId(command_id)
    except InvalidId:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"No node command found with id='{command_id}'",
        )

    doc = await db[COLLECTION].find_one({"_id": object_id, "node_id": node_id})
    if doc is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"No node command found with id='{command_id}' for node_id='{node_id}'",
        )

    now = now_utc()
    await db[COLLECTION].update_one(
        {"_id": object_id},
        {"$set": {"status": NodeCommandStatus.ACKED.value, "acked_at": now}},
    )
    logger.info("Node command acked: node=%s id=%s", node_id, command_id)

    doc = await db[COLLECTION].find_one({"_id": object_id})
    doc["id"] = str(doc["_id"])
    return serialize_mongo_doc(doc)
