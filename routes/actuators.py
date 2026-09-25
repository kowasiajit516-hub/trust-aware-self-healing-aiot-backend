"""
routes/actuators.py
--------------------
FastAPI router for Actuator CRUD + manual control endpoints.

    POST   /actuators
    GET    /actuators
    GET    /actuators/{actuator_id}
    PUT    /actuators/{actuator_id}
    DELETE /actuators/{actuator_id}
    PATCH  /actuators/{actuator_id}/enable
    PATCH  /actuators/{actuator_id}/disable
    POST   /actuators/{actuator_id}/command

Routes stay thin: they validate the request via Pydantic, call the
service layer, and return the result. All business logic lives in
services/actuator_service.py.
"""

from fastapi import APIRouter, Depends, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.actuator import (
    ActuatorCreate,
    ActuatorUpdate,
    ActuatorResponse,
    ActuatorType,
    ActuatorCommandRequest,
)
from services import actuator_service

router = APIRouter(prefix="/actuators", tags=["Actuators"])


@router.post("", response_model=ActuatorResponse, status_code=status.HTTP_201_CREATED)
async def create_actuator(
    payload: ActuatorCreate,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Add a new actuator (any type, including CUSTOM)."""
    return await actuator_service.create_actuator(db, payload)


@router.get("", response_model=list[ActuatorResponse])
async def list_actuators(
    actuator_type: ActuatorType | None = Query(default=None),
    enabled: bool | None = Query(default=None),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """List actuators, optionally filtered by type and/or enabled status."""
    actuator_type_value = actuator_type.value if actuator_type else None
    return await actuator_service.list_actuators(db, actuator_type_value, enabled)


@router.get("/{actuator_id}", response_model=ActuatorResponse)
async def get_actuator(
    actuator_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Get a single actuator by its actuator_id."""
    return await actuator_service.get_actuator(db, actuator_id)


@router.put("/{actuator_id}", response_model=ActuatorResponse)
async def update_actuator(
    actuator_id: str,
    payload: ActuatorUpdate,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Edit an existing actuator (partial update)."""
    return await actuator_service.update_actuator(db, actuator_id, payload)


@router.delete("/{actuator_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_actuator(
    actuator_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Delete an actuator."""
    await actuator_service.delete_actuator(db, actuator_id)


@router.patch("/{actuator_id}/enable", response_model=ActuatorResponse)
async def enable_actuator(
    actuator_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Enable an actuator."""
    from services.actuator_service import COLLECTION
    from models.actuator import now_utc

    await db[COLLECTION].update_one(
        {"actuator_id": actuator_id},
        {"$set": {"enabled": True, "updated_at": now_utc()}},
    )
    return await actuator_service.get_actuator(db, actuator_id)


@router.patch("/{actuator_id}/disable", response_model=ActuatorResponse)
async def disable_actuator(
    actuator_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Disable an actuator."""
    from services.actuator_service import COLLECTION
    from models.actuator import now_utc

    await db[COLLECTION].update_one(
        {"actuator_id": actuator_id},
        {"$set": {"enabled": False, "updated_at": now_utc()}},
    )
    return await actuator_service.get_actuator(db, actuator_id)


@router.post("/{actuator_id}/command", response_model=ActuatorResponse)
async def send_command(
    actuator_id: str,
    payload: ActuatorCommandRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Manually control an actuator: ON, OFF, or AUTO.
    ON/OFF switch it to MANUAL mode; AUTO hands control back to the
    automation engine (fully wired in Phase 9).
    """
    return await actuator_service.send_command(db, actuator_id, payload.command)
