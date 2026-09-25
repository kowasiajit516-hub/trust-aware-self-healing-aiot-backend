"""
database.py
------------
⚠️ RECONSTRUCTION NOTICE: This file was never uploaded to this chat.
Everything below is inferred from how every other file in this project
imports and uses it (mongo_conn.client / mongo_conn.db in conftest.py,
get_database/connect_to_mongo/close_mongo_connection in main.py and
every route, create_indexes() called in conftest.py, settings.mongodb_uri
/ settings.mongodb_db_name in config.py, and the unique/TTL indexes
HANDOVER_07 explicitly documented for users/password_reset_tokens).

DO NOT blindly overwrite your real database.py with this. Diff the two
first. If your real file has additional indexes, helper functions, or
different internal structure, keep those - the only thing Phase 8
actually needs is the one node_commands index call added inside
create_indexes(), marked below.
------------------------------------------------------------------
MongoDB connection lifecycle + shared Motor client for the FastAPI app.
"""

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config import settings

logger = logging.getLogger(__name__)


class MongoConnection:
    """
    Small mutable holder for the shared Motor client/db, so tests can
    swap it out (see tests/conftest.py: mongo_conn.client = ...;
    mongo_conn.db = ...) without needing dependency-injection overrides
    on every route.
    """

    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None


mongo_conn = MongoConnection()


async def connect_to_mongo() -> None:
    """Called from main.py's lifespan startup."""
    logger.info("Connecting to MongoDB at %s", settings.mongodb_uri)
    mongo_conn.client = AsyncIOMotorClient(settings.mongodb_uri)
    mongo_conn.db = mongo_conn.client[settings.mongodb_db_name]
    await create_indexes()
    logger.info("MongoDB connection established (db=%s)", settings.mongodb_db_name)


async def close_mongo_connection() -> None:
    """Called from main.py's lifespan shutdown."""
    if mongo_conn.client is not None:
        mongo_conn.client.close()
        logger.info("MongoDB connection closed")


def get_database() -> AsyncIOMotorDatabase:
    """
    FastAPI dependency used by every route:
        db: AsyncIOMotorDatabase = Depends(get_database)
    """
    if mongo_conn.db is None:
        raise RuntimeError(
            "Database not initialized - connect_to_mongo() must run first "
            "(normally via the app's lifespan, or test_db in tests)."
        )
    return mongo_conn.db


async def create_indexes() -> None:
    """
    Create all indexes the app relies on. Safe to call repeatedly -
    create_index() is a no-op if an equivalent index already exists.
    Called once on real startup (connect_to_mongo) and once per test
    (tests/conftest.py's test_db fixture, against the test database).
    """
    db = mongo_conn.db

    # ---- Phase 1: core entities -----------------------------------
    await db["sensors"].create_index("sensor_id", unique=True)
    await db["actuators"].create_index("actuator_id", unique=True)
    await db["nodes"].create_index("node_id", unique=True)
    await db["automation_rules"].create_index("rule_id", unique=True)

    # ---- Phase 2+: raw readings -------------------------------------
    await db["sensor_readings"].create_index([("sensor_id", 1), ("timestamp", -1)])

    # ---- Phase 3-5: AI pipeline outputs ------------------------------
    await db["predictions"].create_index([("sensor_id", 1), ("timestamp", -1)])
    await db["sensor_health"].create_index([("sensor_id", 1), ("timestamp", -1)])
    await db["trusted_readings"].create_index([("sensor_id", 1), ("timestamp", -1)])

    # ---- Phase 6: fault events ---------------------------------------
    await db["fault_events"].create_index([("sensor_id", 1), ("timestamp", -1)])

    # ---- Phase 7: actuator events, auth ------------------------------
    await db["actuator_events"].create_index([("actuator_id", 1), ("timestamp", -1)])
    await db["users"].create_index("email", unique=True)
    await db["users"].create_index("google_sub", unique=True, sparse=True)
    await db["password_reset_tokens"].create_index("token_hash", unique=True)
    await db["password_reset_tokens"].create_index("expires_at", expireAfterSeconds=0)

    # ---- Phase 8: node_commands queue --------------------------------
    # NEW - required for this handover. If you already have this file,
    # this is the one line to add inside your real create_indexes().
    await db["node_commands"].create_index([("node_id", 1), ("status", 1), ("created_at", 1)])

    # ---- Google Sign-In addition --------------------------------------
    # sparse=True is required here: only Google-linked accounts have a
    # google_sub field at all, so a plain unique index would otherwise
    # try to enforce uniqueness across many documents that are all
    # missing the field (which some MongoDB versions reject outright).
    logger.info("MongoDB indexes ensured")