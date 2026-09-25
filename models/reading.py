"""
models/reading.py
------------------
Pydantic schemas for raw sensor readings.

This is the entry point of the core pipeline:

    RAW SENSOR -> VALIDATION -> AI FAULT DETECTION -> ...

Phase 1 only defines the schema (so sensors/actuators/nodes/rules can
reference it consistently). The actual ingestion ENDPOINT
(POST /readings) and the simulator that calls it are built in Phase 2.
Raw readings are never deleted, even if later marked faulty - see
sensor_health / fault_events (Phase 3+) for fault tracking.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SensorReadingCreate(BaseModel):
    """
    Payload for ingesting a single raw sensor reading.
    Used by the simulator (Phase 2) and, later, real ESP32 nodes (Phase 8).
    """

    sensor_id: str = Field(..., min_length=1, max_length=64, examples=["TEMP_01"])
    value: float = Field(..., examples=[27.4])
    timestamp: datetime | None = Field(
        default=None,
        description="If omitted, the server sets it to the current UTC time.",
    )


class SensorReadingResponse(SensorReadingCreate):
    """What the API returns after storing a raw reading."""

    id: str = Field(..., description="MongoDB document id as a string")
    timestamp: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "66d0f1e2b8f1c2a3d4e5f6a7",
                "sensor_id": "TEMP_01",
                "value": 27.4,
                "timestamp": "2026-08-22T10:00:05Z",
            }
        }
    }


def now_utc() -> datetime:
    """Helper used across models/services for consistent UTC timestamps."""
    return datetime.now(timezone.utc)
