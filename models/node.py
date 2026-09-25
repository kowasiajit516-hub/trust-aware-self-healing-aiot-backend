"""
models/node.py
---------------
Pydantic schemas for the Node entity (an ESP32 physical device).

A node is the physical hardware unit that hosts one or more sensors
and one or more actuators. Phase 1 only needed the CRUD skeleton.
Phase 8 adds real ESP32 communication (heartbeat -> NodeHeartbeat).
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    UNKNOWN = "UNKNOWN"


class NodeBase(BaseModel):
    """Fields shared by create and update operations."""

    node_id: str = Field(..., min_length=1, max_length=64, examples=["ESP32_01"])
    name: str = Field(..., min_length=1, max_length=128, examples=["Lab 1 Node"])
    location: str = Field(..., min_length=1, max_length=128, examples=["Lab 1"])
    ip_address: str | None = Field(default=None, examples=["192.168.1.50"])
    enabled: bool = Field(default=True)


class NodeCreate(NodeBase):
    """Payload for POST /nodes"""
    pass


class NodeUpdate(BaseModel):
    """Payload for PUT /nodes/{node_id}. All fields optional (partial update)."""

    name: str | None = Field(default=None, min_length=1, max_length=128)
    location: str | None = Field(default=None, min_length=1, max_length=128)
    ip_address: str | None = None
    enabled: bool | None = None


class NodeHeartbeat(BaseModel):
    """
    Payload for POST /nodes/{node_id}/heartbeat (Phase 8). A real ESP32
    node calls this on a fixed interval to report it is alive. All
    fields optional - an empty body ({}) is a valid heartbeat.
    """

    ip_address: str | None = Field(
        default=None,
        description="Lets a node on DHCP self-report a changed IP without a manual PUT.",
        examples=["192.168.1.50"],
    )


class NodeResponse(NodeBase):
    """What the API returns for a node."""

    status: NodeStatus = NodeStatus.UNKNOWN
    last_seen: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "node_id": "ESP32_01",
                "name": "Lab 1 Node",
                "location": "Lab 1",
                "ip_address": "192.168.1.50",
                "enabled": True,
                "status": "UNKNOWN",
                "last_seen": None,
                "created_at": "2026-08-22T10:00:00Z",
                "updated_at": "2026-08-22T10:00:00Z",
            }
        }
    }


def now_utc() -> datetime:
    """Helper used across models/services for consistent UTC timestamps."""
    return datetime.now(timezone.utc)
