"""
routes/sensor_health.py
-------------------------
Read-only endpoints exposing Phase 4's dynamic trust-score state.

No POST/PUT - a sensor_health document only ever comes from the trust
engine reacting to a real fault-detection prediction (see
services/trust_engine.py), the same design principle Phase 3 used for
`predictions` (no manual POST there either - see routes/predictions.py).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.sensor_health import SensorHealthResponse
from services import trust_engine
from utils.validators import ensure_exists

router = APIRouter(prefix="/sensor-health", tags=["Sensor Health"])


@router.get("", response_model=list[SensorHealthResponse])
async def list_sensor_health(
    limit: int = 100, db: AsyncIOMotorDatabase = Depends(get_database)
):
    """List current trust/health snapshots for every sensor that has at least one scored prediction."""
    return await trust_engine.list_health(db, limit=limit)


@router.get("/{sensor_id}", response_model=SensorHealthResponse)
async def get_sensor_health(
    sensor_id: str, db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get the current trust/health snapshot for one sensor."""
    await ensure_exists(db, "sensors", "sensor_id", sensor_id)

    health = await trust_engine.get_health(db, sensor_id)
    if health is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No health data yet for sensor_id='{sensor_id}' "
                "(no predictions scored yet - post a reading first)."
            ),
        )
    return health
