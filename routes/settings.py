"""
routes/settings.py
---------------------
GET /settings - real current trust-engine configuration
PUT /settings - real partial update; takes effect on the VERY NEXT
                reading scored (no restart, no redeploy needed) since
                services/trust_engine.update_trust() reads this
                document fresh on every call.

See models/settings.py's module docstring for which parameters from
the original spec were deliberately left out because they don't
control anything real yet (anomaly threshold, fan thresholds, sensor
timeout).
"""

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.settings import SettingsResponse, SettingsUpdate
from services import settings_service

router = APIRouter(prefix="/settings", tags=["Settings"])


@router.get("", response_model=SettingsResponse)
async def get_settings(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Return the current, real trust-engine parameters."""
    return await settings_service.get_settings(db)


@router.put("", response_model=SettingsResponse)
async def update_settings(payload: SettingsUpdate, db: AsyncIOMotorDatabase = Depends(get_database)):
    """Update one or more trust-engine parameters. Takes effect immediately."""
    return await settings_service.update_settings(db, payload.model_dump())
