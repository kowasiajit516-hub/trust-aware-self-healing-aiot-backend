"""
models/node_command.py
-----------------------
Pydantic schemas for queued actuator commands delivered to a real
ESP32 node (Phase 8).

When a manual actuator command (ON/OFF/AUTO, via
POST /actuators/{id}/command) targets an actuator whose actuator_id
belongs to a node (actuator.node_id is set), actuator_service.send_command
enqueues a NodeCommand here as a best-effort side effect - this is
purely additive; the existing Phase 1 write path to `actuators` and
`actuator_events` is completely unchanged.

The node then either:
  - receives it immediately via a best-effort direct HTTP push
    (esp32/esp32_client.py -> the node's own tiny HTTP server), or
  - picks it up on its next GET /nodes/{node_id}/commands/pending poll
    (the reliable path - always works, even if the direct push fails
    or the node has no known ip_address yet)

...and acknowledges it via POST /nodes/{node_id}/commands/{command_id}/ack.
Commands are never deleted, only marked ACKED, so this collection also
serves as a delivery audit trail (same "never delete, keep raw/derived
records distinguishable" spirit as rule #8, applied to command delivery).
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class NodeCommandStatus(str, Enum):
    PENDING = "PENDING"
    ACKED = "ACKED"


class NodeCommandResponse(BaseModel):
    """What the API returns for a queued/acknowledged node command."""

    id: str
    node_id: str
    actuator_id: str
    command: str  # "ON" | "OFF" | "AUTO" - mirrors ActuatorCommand's value
    status: NodeCommandStatus
    created_at: datetime
    acked_at: datetime | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "66d0f1e2b8f1c2a3d4e5f6a8",
                "node_id": "ESP32_01",
                "actuator_id": "FAN_01",
                "command": "ON",
                "status": "PENDING",
                "created_at": "2026-08-30T10:00:00Z",
                "acked_at": None,
            }
        }
    }
