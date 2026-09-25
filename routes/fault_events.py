"""
routes/fault_events.py
-------------------------
Read-only endpoints exposing Phase 6's self-healing engine.

No POST/PUT - same design principle Phase 3/4/5 used for predictions/
sensor_health/trusted_readings: a fault_events document only ever
comes from services/trust_engine.py reacting to a real status
transition it just derived. There is no manual way to inject an event
that didn't come from the real pipeline.
"""

from fastapi import APIRouter, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.fault_event import FaultEventResponse
from services import self_healing

router = APIRouter(prefix="/fault-events", tags=["Fault Events"])


@router.get("", response_model=list[FaultEventResponse])
async def list_fault_events(
    sensor_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """List self-healing status transitions, most recent first."""
    return await self_healing.list_fault_events(db, sensor_id, limit)


@router.get("/{sensor_id}/latest", response_model=FaultEventResponse)
async def get_latest_fault_event(
    sensor_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Get the most recent status transition for a given sensor."""
    return await self_healing.get_latest_fault_event(db, sensor_id)