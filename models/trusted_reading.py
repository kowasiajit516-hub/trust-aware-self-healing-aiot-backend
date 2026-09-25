"""
models/trusted_reading.py
---------------------------
Pydantic schema for the generic `trusted_readings` collection - written
by the VIRTUAL SENSOR stage (services/virtual_sensor.py, Phase 5).

This is the TRUSTED VALUE stage of the core pipeline:

    ... -> VIRTUAL SENSOR -> TRUSTED VALUE -> AUTOMATION ENGINE -> ACTUATOR

One document per trust-scored reading (mirrors `predictions`' one-per-
reading shape, unlike `sensor_health`'s one-per-sensor live snapshot).
Phase 6/9's automation engine is expected to read `trusted_value` from
THIS collection, never raw `sensor_readings` directly - per the master
plan's hardest rule ("actuators act on trusted values only, never raw
sensor values").
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class TrustedReadingResponse(BaseModel):
    """
    One trusted value for one reading - either the raw reading itself
    (sensor was HEALTHY, so it's already trustworthy) or a Random-
    Forest-predicted virtual value (sensor was RECOVERING/FAULTY, so
    the raw value is not trusted and is stood in for).
    """

    id: str = Field(..., description="MongoDB document id as a string")
    sensor_id: str
    reading_id: str = Field(..., description="id of the sensor_readings document this covers")
    raw_value: float = Field(
        ..., description="the original raw value - always retained, never deleted (rule #8)"
    )
    trusted_value: float = Field(
        ..., description="the value automation/actuators should use going forward (Phase 6/9)"
    )
    source: str = Field(
        ..., description="'RAW' if the sensor was HEALTHY at prediction time, 'VIRTUAL' if predicted"
    )
    status_at_prediction: str = Field(
        ..., description="the sensor's SensorStatus at the moment this trusted value was computed"
    )
    timestamp: datetime
    created_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "66d0f1e2b8f1c2a3d4e5f6a9",
                "sensor_id": "TEMP_01",
                "reading_id": "66d0f1e2b8f1c2a3d4e5f6a7",
                "raw_value": 9999.0,
                "trusted_value": 22.3,
                "source": "VIRTUAL",
                "status_at_prediction": "FAULTY",
                "timestamp": "2026-08-22T10:00:05Z",
                "created_at": "2026-08-22T10:00:05Z",
            }
        }
    }


def now_utc() -> datetime:
    """Helper used across models/services for consistent UTC timestamps."""
    return datetime.now(timezone.utc)
