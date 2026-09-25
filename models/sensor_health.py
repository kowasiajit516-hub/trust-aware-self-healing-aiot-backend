"""
models/sensor_health.py
-------------------------
Pydantic schema for the generic `sensor_health` collection - written by
the DYNAMIC TRUST SCORE stage (services/trust_engine.py, Phase 4).

One document per sensor (sensor_id is unique, upserted), representing a
live snapshot of that sensor's current trust state - not a history log.
Per-event history remains in `predictions` (Phase 3, already real) and
the future `fault_events` collection.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from models.sensor import SensorStatus


class SensorHealthResponse(BaseModel):
    """Current trust/health snapshot for one sensor."""

    sensor_id: str
    trust_score: float = Field(..., ge=0.0, le=100.0)
    status: SensorStatus
    consecutive_faults: int = Field(..., ge=0)
    consecutive_healthy: int = Field(..., ge=0)
    last_prediction_id: str | None = None
    last_is_fault: bool | None = None
    last_anomaly_score: float | None = None
    last_reading_timestamp: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "sensor_id": "TEMP_01",
                "trust_score": 75.0,
                "status": "RECOVERING",
                "consecutive_faults": 0,
                "consecutive_healthy": 1,
                "last_prediction_id": "66d0f1e2b8f1c2a3d4e5f6a8",
                "last_is_fault": False,
                "last_anomaly_score": 0.042,
                "last_reading_timestamp": "2026-08-22T10:00:05Z",
                "created_at": "2026-08-22T09:55:00Z",
                "updated_at": "2026-08-22T10:00:05Z",
            }
        }
    }
