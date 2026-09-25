"""
routes/sensors.py
------------------
FastAPI router for Sensor CRUD endpoints.

    POST   /sensors
    GET    /sensors
    GET    /sensors/{sensor_id}
    PUT    /sensors/{sensor_id}
    DELETE /sensors/{sensor_id}
    PATCH  /sensors/{sensor_id}/enable
    PATCH  /sensors/{sensor_id}/disable

Routes stay thin: they validate the request via Pydantic, call the
service layer, and return the result. All business logic lives in
services/sensor_service.py.
"""

from fastapi import APIRouter, Depends, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.sensor import SensorCreate, SensorUpdate, SensorResponse, SensorType
from services import sensor_service

router = APIRouter(prefix="/sensors", tags=["Sensors"])


@router.post("", response_model=SensorResponse, status_code=status.HTTP_201_CREATED)
async def create_sensor(
    payload: SensorCreate,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Add a new sensor (any type, including CUSTOM)."""
    return await sensor_service.create_sensor(db, payload)


@router.get("", response_model=list[SensorResponse])
async def list_sensors(
    sensor_type: SensorType | None = Query(default=None),
    enabled: bool | None = Query(default=None),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """List sensors, optionally filtered by type and/or enabled status."""
    sensor_type_value = sensor_type.value if sensor_type else None
    return await sensor_service.list_sensors(db, sensor_type_value, enabled)


@router.get("/{sensor_id}", response_model=SensorResponse)
async def get_sensor(
    sensor_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Get a single sensor by its sensor_id."""
    return await sensor_service.get_sensor(db, sensor_id)


@router.put("/{sensor_id}", response_model=SensorResponse)
async def update_sensor(
    sensor_id: str,
    payload: SensorUpdate,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Edit an existing sensor (partial update)."""
    return await sensor_service.update_sensor(db, sensor_id, payload)


@router.delete("/{sensor_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sensor(
    sensor_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Delete a sensor."""
    await sensor_service.delete_sensor(db, sensor_id)


@router.patch("/{sensor_id}/enable", response_model=SensorResponse)
async def enable_sensor(
    sensor_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Enable a sensor."""
    return await sensor_service.set_sensor_enabled(db, sensor_id, True)


@router.patch("/{sensor_id}/disable", response_model=SensorResponse)
async def disable_sensor(
    sensor_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Disable a sensor."""
    return await sensor_service.set_sensor_enabled(db, sensor_id, False)
