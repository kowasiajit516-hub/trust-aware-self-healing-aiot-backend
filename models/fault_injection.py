"""
models/fault_injection.py
----------------------------
Pydantic schemas for the dashboard's Fault Injection panel.

This does NOT bypass the core pipeline for the five types that submit
a reading (SPIKE, SUDDEN_OFFSET, STUCK, FROZEN, DRIFT, COMMUNICATION):
POST /fault-injection computes a deliberately corrupted value and then
calls the real services/reading_service.create_reading() - the same
function POST /readings uses - so the corrupted value still goes
through real validation, real AI fault detection, real trust scoring,
and real virtual sensing. Nothing here writes to sensor_readings/
predictions/sensor_health/trusted_readings/fault_events directly.

MISSING is different and its limitation is documented on the type
itself: it deliberately submits NO reading (simulating a dropout), so
injected_value/reading are null in the response. Because this project
has no background staleness/timeout watcher, a MISSING injection will
NOT by itself change a sensor's status or trust_score - status only
changes in reaction to a NEW reading being scored. Building real
staleness detection (a scheduled job flagging sensors with no recent
reading as OFFLINE) is a separate, larger feature this does not
include.
"""

from enum import Enum

from pydantic import BaseModel, Field


class FaultType(str, Enum):
    SPIKE = "SPIKE"  # sudden large deviation past the normal range edge
    SUDDEN_OFFSET = "SUDDEN_OFFSET"  # smaller constant shift from the expected value (calibration-error style)
    STUCK = "STUCK"  # repeats the sensor's last known value
    FROZEN = "FROZEN"  # same behavior as STUCK - value stops changing (kept as a separate label to match common IoT fault terminology)
    DRIFT = "DRIFT"  # gradual wander away from the normal range
    MISSING = "MISSING"  # submits no reading at all - see module docstring for the real limitation this has
    COMMUNICATION = "COMMUNICATION"  # sends an implausible sentinel value simulating garbled/invalid sensor data


class FaultInjectionRequest(BaseModel):
    """Payload for POST /fault-injection/{sensor_id}"""

    fault_type: FaultType
    offset: float | None = Field(
        default=None,
        description=(
            "Optional explicit offset added to the sensor's current "
            "value/range edge. If omitted, a random offset is generated "
            "the same way simulator.py's --fault-rate does. Ignored for MISSING."
        ),
    )


class FaultInjectionResponse(BaseModel):
    """
    What the API returns after injecting a fault. injected_value/reading
    are null for MISSING (no reading was submitted) - see FaultType.MISSING.
    """

    sensor_id: str
    fault_type: FaultType
    injected_value: float | None = None
    reading: dict | None = Field(
        default=None, description="the SensorReadingResponse produced by the real /readings pipeline, or null for MISSING"
    )
    note: str | None = Field(
        default=None, description="present for MISSING - explains the real limitation, not a fabricated success message"
    )
