"""
services/node_service.py
--------------------------
Business logic for the Node (ESP32) entity.

Phase 1 scope was CRUD only - status was always UNKNOWN and last_seen
was always None; nothing ever updated them. Phase 8 adds the real
thing:

  - record_heartbeat() is called from POST /nodes/{node_id}/heartbeat
    whenever a real ESP32 node checks in: sets status=ONLINE,
    last_seen=now.
  - Every read path (get_node / list_nodes) lazily downgrades a node
    to OFFLINE if it has gone quiet for longer than
    OFFLINE_AFTER_SECONDS. This needs no background cron job, mirroring
    the lazy/expiry-style pattern already used elsewhere in this
    project (Phase 7's password_reset_tokens Mongo TTL index - though
    here it's a status flip on read, not a delete).
"""

import logging
from datetime import timedelta, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.node import NodeCreate, NodeUpdate, NodeStatus, now_utc
from utils.validators import ensure_unique, ensure_exists, serialize_mongo_doc

logger = logging.getLogger(__name__)

COLLECTION = "nodes"

# A node that hasn't sent a heartbeat in this long is considered OFFLINE.
# Phase 8 keeps this a fixed constant (not per-node configurable) to stay
# in scope; a per-node "expected heartbeat interval" field is a
# reasonable future enhancement, not required by the master plan.
OFFLINE_AFTER_SECONDS = 90


async def create_node(db: AsyncIOMotorDatabase, payload: NodeCreate) -> dict:
    await ensure_unique(db, COLLECTION, "node_id", payload.node_id)

    now = now_utc()
    doc = payload.model_dump()
    doc.update(
        {
            "status": NodeStatus.UNKNOWN.value,
            "last_seen": None,
            "created_at": now,
            "updated_at": now,
        }
    )

    await db[COLLECTION].insert_one(doc)
    logger.info("Node created: %s", payload.node_id)
    return serialize_mongo_doc(doc)


def _ensure_aware(dt):
    """Mongo/motor can hand back naive UTC datetimes depending on client
    config; normalize before doing timezone-aware arithmetic on them."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def _maybe_mark_offline(db: AsyncIOMotorDatabase, doc: dict) -> dict:
    """
    Lazily downgrade a node's status to OFFLINE if its last_seen is
    older than OFFLINE_AFTER_SECONDS. Only writes to the DB (and only
    mutates the returned doc) when a change is actually needed, so a
    normal read of a healthy, already-OFFLINE, or still-UNKNOWN node
    stays a single find with no extra write.
    """
    if doc.get("status") != NodeStatus.ONLINE.value:
        return doc

    last_seen = doc.get("last_seen")
    if last_seen is None:
        return doc

    if now_utc() - _ensure_aware(last_seen) <= timedelta(seconds=OFFLINE_AFTER_SECONDS):
        return doc

    await db[COLLECTION].update_one(
        {"node_id": doc["node_id"]},
        {"$set": {"status": NodeStatus.OFFLINE.value, "updated_at": now_utc()}},
    )
    logger.info(
        "Node %s marked OFFLINE (no heartbeat for >%ss)",
        doc["node_id"], OFFLINE_AFTER_SECONDS,
    )
    doc["status"] = NodeStatus.OFFLINE.value
    return doc


async def list_nodes(
    db: AsyncIOMotorDatabase, enabled: bool | None = None
) -> list[dict]:
    query: dict = {}
    if enabled is not None:
        query["enabled"] = enabled

    cursor = db[COLLECTION].find(query).sort("created_at", 1)
    docs = [doc async for doc in cursor]
    docs = [await _maybe_mark_offline(db, doc) for doc in docs]
    return [serialize_mongo_doc(doc) for doc in docs]


async def get_node(db: AsyncIOMotorDatabase, node_id: str) -> dict:
    doc = await ensure_exists(db, COLLECTION, "node_id", node_id)
    doc = await _maybe_mark_offline(db, doc)
    return serialize_mongo_doc(doc)


async def update_node(
    db: AsyncIOMotorDatabase, node_id: str, payload: NodeUpdate
) -> dict:
    await ensure_exists(db, COLLECTION, "node_id", node_id)

    update_data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not update_data:
        doc = await db[COLLECTION].find_one({"node_id": node_id})
        return serialize_mongo_doc(doc)

    update_data["updated_at"] = now_utc()

    await db[COLLECTION].update_one({"node_id": node_id}, {"$set": update_data})
    logger.info("Node updated: %s -> %s", node_id, list(update_data.keys()))

    doc = await db[COLLECTION].find_one({"node_id": node_id})
    return serialize_mongo_doc(doc)


async def delete_node(db: AsyncIOMotorDatabase, node_id: str) -> None:
    await ensure_exists(db, COLLECTION, "node_id", node_id)
    await db[COLLECTION].delete_one({"node_id": node_id})
    logger.info("Node deleted: %s", node_id)


async def record_heartbeat(
    db: AsyncIOMotorDatabase, node_id: str, ip_address: str | None = None
) -> dict:
    """
    Called from POST /nodes/{node_id}/heartbeat whenever a real ESP32
    node checks in. Sets status=ONLINE and last_seen=now; updates
    ip_address only if the node actually reported one, letting a node
    on DHCP self-correct its recorded IP without a manual PUT.
    """
    await ensure_exists(db, COLLECTION, "node_id", node_id)

    now = now_utc()
    update_data: dict = {
        "status": NodeStatus.ONLINE.value,
        "last_seen": now,
        "updated_at": now,
    }
    if ip_address:
        update_data["ip_address"] = ip_address

    await db[COLLECTION].update_one({"node_id": node_id}, {"$set": update_data})
    logger.info("Heartbeat received from node %s", node_id)

    doc = await db[COLLECTION].find_one({"node_id": node_id})
    return serialize_mongo_doc(doc)
