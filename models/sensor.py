"""
models/sensor.py
-----------------
Pydantic schemas for the generic Sensor entity.

Supports ANY sensor type dynamically (temperature, humidity, light, gas,
smoke, pressure, air quality, soil moisture, vibration, water level,
motion, current, voltage, custom, ...). The backend never hardcodes
per-type logic - "sensor_type" is just a string/enum value.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SensorType(str, Enum):
    TEMPERATURE = "TEMPERATURE"
    HUMIDITY = "HUMIDITY"
    LIGHT = "LIGHT"
    GAS = "GAS"
    SMOKE = "SMOKE"
    PRESSURE = "PRESSURE"
    AIR_QUALITY = "AIR_QUALITY"
    SOIL_MOISTURE = "SOIL_MOISTURE"
    VIBRATION = "VIBRATION"
    WATER_LEVEL = "WATER_LEVEL"
    MOTION = "MOTION"
    CURRENT = "CURRENT"
    VOLTAGE = "VOLTAGE"
    CUSTOM = "CUSTOM"


class SensorStatus(str, Enum):
    HEALTHY = "HEALTHY"
    FAULTY = "FAULTY"
    RECOVERING = "RECOVERING"
    OFFLINE = "OFFLINE"


class SensorBase(BaseModel):
    """Fields shared by create and update operations."""

    sensor_id: str = Field(..., min_length=1, max_length=64, examples=["TEMP_01"])
    name: str = Field(..., min_length=1, max_length=128, examples=["Room Temperature"])
    sensor_type: SensorType
    unit: str = Field(..., min_length=1, max_length=16, examples=["°C"])
    location: str = Field(..., min_length=1, max_length=128, examples=["Lab 1"])
    node_id: str | None = Field(default=None, examples=["ESP32_01"])

    normal_min: float = Field(..., examples=[15.0])
    normal_max: float = Field(..., examples=[40.0])

    sampling_interval_seconds: int = Field(default=5, ge=1, le=3600)
    enabled: bool = Field(default=True)

    @field_validator("normal_max")
    @classmethod
    def max_must_exceed_min(cls, v: float, info):
        normal_min = info.data.get("normal_min")
        if normal_min is not None and v <= normal_min:
            raise ValueError("normal_max must be greater than normal_min")
        return v


class SensorCreate(SensorBase):
    """Payload for POST /sensors"""
    pass


class SensorUpdate(BaseModel):
    """Payload for PUT /sensors/{sensor_id}. All fields optional (partial update)."""

    name: str | None = Field(default=None, min_length=1, max_length=128)
    sensor_type: SensorType | None = None
    unit: str | None = Field(default=None, min_length=1, max_length=16)
    location: str | None = Field(default=None, min_length=1, max_length=128)
    node_id: str | None = None
    normal_min: float | None = None
    normal_max: float | None = None
    sampling_interval_seconds: int | None = Field(default=None, ge=1, le=3600)
    enabled: bool | None = None


class SensorResponse(SensorBase):
    """What the API returns for a sensor."""

    status: SensorStatus = SensorStatus.HEALTHY
    trust_score: float = Field(default=100.0, ge=0.0, le=100.0)
    created_at: datetime
    updated_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "sensor_id": "TEMP_01",
                "name": "Room Temperature",
                "sensor_type": "TEMPERATURE",
                "unit": "°C",
                "location": "Lab 1",
                "node_id": "ESP32_01",
                "normal_min": 15.0,
                "normal_max": 40.0,
                "sampling_interval_seconds": 5,
                "enabled": True,
                "status": "HEALTHY",
                "trust_score": 100.0,
                "created_at": "2026-08-22T10:00:00Z",
                "updated_at": "2026-08-22T10:00:00Z",
            }
        }
    }


def now_utc() -> datetime:
    """Helper used across models/services for consistent UTC timestamps."""
    return datetime.now(timezone.utc)
