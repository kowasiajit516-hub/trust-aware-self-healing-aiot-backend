"""
models/actuator_event.py
---------------------------
Pydantic schema for the generic `actuator_events` collection.

The `actuator_events` collection and its index have existed since
Phase 1 (database.py: [("actuator_id", 1), ("timestamp", -1)]), and
services/actuator_service.send_command() has been writing one document
here per manual ON/OFF/AUTO command since Phase 1 as well - but no
route ever exposed it for reading, even though the master plan's
section 7 (Key API Groups) names `GET /actuator-events` explicitly.

This mirrors the exact read-only pattern Phase 6 used for
fault_events: no POST/PUT here - an actuator_events document only
ever comes from services/actuator_service.send_command() reacting to
a real command against a real actuator. There is no manual way to
inject an event that didn't come from the real pipeline.

`trusted_value` stays null until the automation engine (Phase 9)
starts driving AUTO-mode commands from trusted sensor values - the
field already exists on the written document (see actuator_service.py)
so no schema migration will be needed then.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ActuatorEventResponse(BaseModel):
    """One recorded command/state-change for one actuator."""

    id: str = Field(..., description="MongoDB document id as a string")
    actuator_id: str
    command: str = Field(..., description="ON, OFF, or AUTO")
    previous_state: str
    new_state: str
    mode: str = Field(..., description="Actuator mode (AUTO/MANUAL) after this event")
    reason: str
    sensor_id: str | None = Field(
        default=None, description="controlling_sensor at the time of the command, if any"
    )
    trusted_value: float | None = Field(
        default=None,
        description="trusted value that drove this command; null until Phase 9's automation engine exists",
    )
    timestamp: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "66d0f1e2b8f1c2a3d4e5f6c1",
                "actuator_id": "FAN_01",
                "command": "ON",
                "previous_state": "OFF",
                "new_state": "ON",
                "mode": "MANUAL",
                "reason": "Manual command: ON",
                "sensor_id": "TEMP_01",
                "trusted_value": None,
                "timestamp": "2026-08-22T10:00:05Z",
            }
        }
    }
