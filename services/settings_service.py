"""
services/settings_service.py
-------------------------------
Backing logic for the one system-wide settings document. Real
persistence (MongoDB), real defaults matching what trust_engine.py
used to hardcode - editing a setting here genuinely changes trust
calculation behavior on the next reading, verified in
tests/test_settings.py.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.settings import DEFAULT_SETTINGS, now_utc

COLLECTION = "system_settings"
DOC_ID = "system"


async def get_settings(db: AsyncIOMotorDatabase) -> dict:
    """Return the current settings, creating the default document if none exists yet."""
    doc = await db[COLLECTION].find_one({"id": DOC_ID})
    if doc is None:
        doc = {**DEFAULT_SETTINGS, "updated_at": now_utc()}
        await db[COLLECTION].insert_one(doc)
    return doc


async def update_settings(db: AsyncIOMotorDatabase, updates: dict) -> dict:
    """Apply a partial update (only non-None fields) and return the new full document."""
    await get_settings(db)  # ensure the doc exists before updating it

    changes = {k: v for k, v in updates.items() if v is not None}
    if changes:
        changes["updated_at"] = now_utc()
        await db[COLLECTION].update_one({"id": DOC_ID}, {"$set": changes})

    return await db[COLLECTION].find_one({"id": DOC_ID})
