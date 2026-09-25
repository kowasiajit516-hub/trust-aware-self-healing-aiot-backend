"""
routes/actuator_events.py
----------------------------
Read-only endpoints exposing recorded actuator command/state history.

No POST/PUT - same design principle Phase 3/4/5/6 used for
predictions/sensor_health/trusted_readings/fault_events: an
actuator_events document only ever comes from
services/actuator_service.send_command() reacting to a real command
against a real actuator (see routes/actuators.py's
POST /actuators/{id}/command). There is no manual way to inject an
event that didn't come from the real pipeline.

This closes a gap: the `actuator_events` collection/index has existed
since Phase 1 and has been written to since Phase 1, and the master
plan's section 7 names `GET /actuator-status` / `GET /actuator-events`
explicitly, but no route ever exposed it for reading until now.
"""

from fastapi import APIRouter, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.actuator_event import ActuatorEventResponse
from services import actuator_service

router = APIRouter(prefix="/actuator-events", tags=["Actuator Events"])


@router.get("", response_model=list[ActuatorEventResponse])
async def list_actuator_events(
    actuator_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """List recorded actuator command/state-change events, most recent first."""
    return await actuator_service.list_actuator_events(db, actuator_id, limit)


@router.get("/{actuator_id}/latest", response_model=ActuatorEventResponse)
async def get_latest_actuator_event(
    actuator_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Get the most recent recorded event for a given actuator."""
    return await actuator_service.get_latest_actuator_event(db, actuator_id)
