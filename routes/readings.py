"""
routes/readings.py
-------------------
FastAPI router for raw sensor reading ingestion.

    POST /readings                    - ingest one raw reading
    GET  /readings                    - list readings (filter by sensor_id, limit)
    GET  /readings/{sensor_id}/latest - most recent reading for a sensor

This is the entry point of the core pipeline (see master plan):

    RAW SENSOR -> VALIDATION -> AI FAULT DETECTION -> ...

Phase 2 scope: ingestion + storage only. The simulator (and, later,
real ESP32 nodes) must POST here - direct database writes bypassing
this endpoint are a "simulator-only shortcut" and are not allowed.
Fault detection, trust scoring, and virtual sensing are Phases 3-5.
"""

from fastapi import APIRouter, Depends, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.reading import SensorReadingCreate, SensorReadingResponse
from services import reading_service

router = APIRouter(prefix="/readings", tags=["Readings"])


@router.post("", response_model=SensorReadingResponse, status_code=status.HTTP_201_CREATED)
async def create_reading(
    payload: SensorReadingCreate,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Ingest a single raw sensor reading. sensor_id must already exist."""
    return await reading_service.create_reading(db, payload)


@router.get("", response_model=list[SensorReadingResponse])
async def list_readings(
    sensor_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """List raw readings, most recent first, optionally filtered by sensor_id."""
    return await reading_service.list_readings(db, sensor_id, limit)


@router.get("/{sensor_id}/latest", response_model=SensorReadingResponse)
async def get_latest_reading(
    sensor_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Get the most recent raw reading stored for a given sensor."""
    return await reading_service.get_latest_reading(db, sensor_id)
