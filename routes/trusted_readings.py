"""
routes/trusted_readings.py
----------------------------
Read-only endpoints exposing Phase 5's TRUSTED VALUE stage.

No POST/PUT - same design principle Phase 3/4 used for predictions/
sensor_health: a trusted_readings document only ever comes from the
virtual sensor reacting to a real, already trust-scored reading (see
services/virtual_sensor.py). There is no manual way to inject a
trusted value that didn't come from the real pipeline.
"""

from fastapi import APIRouter, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.trusted_reading import TrustedReadingResponse
from services import virtual_sensor

router = APIRouter(prefix="/trusted-readings", tags=["Trusted Readings"])


@router.get("", response_model=list[TrustedReadingResponse])
async def list_trusted_readings(
    sensor_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """List trusted values, most recent first."""
    return await virtual_sensor.list_trusted(db, sensor_id, limit)


@router.get("/{sensor_id}/latest", response_model=TrustedReadingResponse)
async def get_latest_trusted_reading(
    sensor_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Get the most recent trusted value for a given sensor."""
    return await virtual_sensor.get_latest_trusted(db, sensor_id)
