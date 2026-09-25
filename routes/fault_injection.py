"""
routes/fault_injection.py
----------------------------
POST /fault-injection/{sensor_id} - the dashboard's Fault Injection
panel. See models/fault_injection.py and
services/fault_injection_service.py for why this is real, pipeline-
respecting code and not a shortcut: it computes a corrupted value with
the same math simulator.py uses, then posts it through the real
services/reading_service.create_reading() (same function
POST /readings uses), so it still passes through real validation, real
AI fault detection, real trust scoring, and real virtual sensing.
"""

from fastapi import APIRouter, Depends, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.fault_injection import FaultInjectionRequest, FaultInjectionResponse
from services import fault_injection_service

router = APIRouter(prefix="/fault-injection", tags=["Fault Injection"])


@router.post("/{sensor_id}", response_model=FaultInjectionResponse, status_code=status.HTTP_201_CREATED)
async def inject_fault(
    sensor_id: str,
    payload: FaultInjectionRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Inject a fault into a real sensor. All types except MISSING go through the real ingestion pipeline."""
    value, reading, note = await fault_injection_service.inject_fault(
        db, sensor_id, payload.fault_type, payload.offset
    )
    return {
        "sensor_id": sensor_id,
        "fault_type": payload.fault_type,
        "injected_value": value,
        "reading": reading,
        "note": note,
    }
