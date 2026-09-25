"""
models/actuator.py
-------------------
Pydantic schemas for the generic Actuator entity.

Supports ANY actuator type dynamically (fan, LED, buzzer, pump, relay,
motor, light, valve, servo, custom, ...). The backend never hardcodes
per-type logic - "actuator_type" is just a string/enum value.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class ActuatorType(str, Enum):
    FAN = "FAN"
    LED = "LED"
    BUZZER = "BUZZER"
    PUMP = "PUMP"
    RELAY = "RELAY"
    MOTOR = "MOTOR"
    LIGHT = "LIGHT"
    VALVE = "VALVE"
    SERVO = "SERVO"
    CUSTOM = "CUSTOM"


class ActuatorMode(str, Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"


class ActuatorState(str, Enum):
    ON = "ON"
    OFF = "OFF"


class ActuatorCommand(str, Enum):
    """Commands accepted by POST /actuators/{id}/command"""
    ON = "ON"
    OFF = "OFF"
    AUTO = "AUTO"


class ActuatorBase(BaseModel):
    """Fields shared by create and update operations."""

    actuator_id: str = Field(..., min_length=1, max_length=64, examples=["FAN_01"])
    name: str = Field(..., min_length=1, max_length=128, examples=["Cooling Fan"])
    actuator_type: ActuatorType
    location: str = Field(..., min_length=1, max_length=128, examples=["Lab 1"])
    node_id: str | None = Field(default=None, examples=["ESP32_01"])
    mode: ActuatorMode = Field(default=ActuatorMode.AUTO)
    enabled: bool = Field(default=True)


class ActuatorCreate(ActuatorBase):
    """Payload for POST /actuators"""
    pass


class ActuatorUpdate(BaseModel):
    """Payload for PUT /actuators/{actuator_id}. All fields optional (partial update)."""

    name: str | None = Field(default=None, min_length=1, max_length=128)
    actuator_type: ActuatorType | None = None
    location: str | None = Field(default=None, min_length=1, max_length=128)
    node_id: str | None = None
    mode: ActuatorMode | None = None
    enabled: bool | None = None


class ActuatorCommandRequest(BaseModel):
    """Payload for POST /actuators/{actuator_id}/command"""

    command: ActuatorCommand


class ActuatorResponse(ActuatorBase):
    """What the API returns for an actuator."""

    state: ActuatorState = ActuatorState.OFF
    last_command: ActuatorCommand | None = None
    controlling_sensor: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "actuator_id": "FAN_01",
                "name": "Cooling Fan",
                "actuator_type": "FAN",
                "location": "Lab 1",
                "node_id": "ESP32_01",
                "mode": "AUTO",
                "enabled": True,
                "state": "OFF",
                "last_command": None,
                "controlling_sensor": "TEMP_01",
                "created_at": "2026-08-22T10:00:00Z",
                "updated_at": "2026-08-22T10:00:00Z",
            }
        }
    }


def now_utc() -> datetime:
    """Helper used across models/services for consistent UTC timestamps."""
    return datetime.now(timezone.utc)
