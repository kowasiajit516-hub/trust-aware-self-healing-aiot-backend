"""
models/fault_event.py
------------------------
Pydantic schema for the generic `fault_events` collection - written by
the SELF-HEALING ENGINE (services/self_healing.py, Phase 6).

This is the observability layer for the master plan's section 4 demo
loop:

    NORMAL -> SENSOR FAULT -> AI DETECTION -> TRUST DECREASE -> VIRTUAL SENSOR
       -> TRUSTED VALUE -> AUTOMATION CONTINUES -> ACTUATOR CORRECT
       -> SENSOR RECOVERY -> TRUST RECOVERY -> REAL SENSOR RESTORED

Phases 3-5 already compute and persist every value in that loop
(predictions, sensor_health, trusted_readings), but nothing previously
recorded the loop's STATE TRANSITIONS themselves - `sensor_health` is a
live snapshot, not a history, so "when did this sensor become FAULTY"
or "when did it recover" was previously unanswerable without diffing
snapshots over time. One document here = one status transition for one
sensor (e.g. HEALTHY -> RECOVERING), not one per reading.

`fault_events`' collection/index was defined since Phase 1
(database.py: `[("sensor_id", 1), ("timestamp", -1)]`) but unused until
now.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from models.sensor import SensorStatus


class FaultEventResponse(BaseModel):
    """One self-healing loop transition for one sensor."""

    id: str = Field(..., description="MongoDB document id as a string")
    sensor_id: str
    previous_status: SensorStatus
    new_status: SensorStatus
    trust_score: float = Field(..., ge=0.0, le=100.0, description="trust_score at the moment of transition")
    trigger_reading_id: str | None = Field(
        default=None, description="id of the sensor_readings document whose prediction caused this transition"
    )
    timestamp: datetime
    created_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "66d0f1e2b8f1c2a3d4e5f6b1",
                "sensor_id": "TEMP_01",
                "previous_status": "HEALTHY",
                "new_status": "RECOVERING",
                "trust_score": 80.0,
                "trigger_reading_id": "66d0f1e2b8f1c2a3d4e5f6a7",
                "timestamp": "2026-08-22T10:00:05Z",
                "created_at": "2026-08-22T10:00:05Z",
            }
        }
    }


def now_utc() -> datetime:
    """Helper used across models/services for consistent UTC timestamps."""
    return datetime.now(timezone.utc)