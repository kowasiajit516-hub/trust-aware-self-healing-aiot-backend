"""
models/automation_rule.py
--------------------------
Pydantic schemas for the Automation Rule entity.

An automation rule links one sensor to one actuator with a condition
and threshold (e.g. "IF trusted TEMP_01 >= 28 THEN turn FAN_01 ON").

Phase 1 only needs the CRUD skeleton - the actual rule EVALUATION
engine (reading trusted values and driving actuators) is implemented
in Phase 9, after the self-healing pipeline (Phases 3-6) exists.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from models.actuator import ActuatorState


class RuleCondition(str, Enum):
    GREATER_THAN = "GREATER_THAN"
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL"
    LESS_THAN = "LESS_THAN"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"
    EQUAL = "EQUAL"


class RuleAction(str, Enum):
    ON = "ON"
    OFF = "OFF"


class AutomationRuleBase(BaseModel):
    """Fields shared by create and update operations."""

    rule_id: str = Field(..., min_length=1, max_length=64, examples=["RULE_001"])
    name: str = Field(..., min_length=1, max_length=128, examples=["Fan on high temp"])
    sensor_id: str = Field(..., examples=["TEMP_01"])
    actuator_id: str = Field(..., examples=["FAN_01"])

    condition: RuleCondition = Field(..., examples=["GREATER_THAN_OR_EQUAL"])
    threshold: float = Field(..., examples=[28.0])
    action: RuleAction = Field(..., examples=["ON"])

    # Optional hysteresis: the "reverse" threshold that flips the actuator
    # back off (or on). E.g. ON at >=28, OFF at <=26. Left null = no hysteresis.
    reset_threshold: float | None = Field(default=None, examples=[26.0])
    reset_action: RuleAction | None = Field(default=None, examples=["OFF"])

    enabled: bool = Field(default=True)


class AutomationRuleCreate(AutomationRuleBase):
    """Payload for POST /automation-rules"""
    pass


class AutomationRuleUpdate(BaseModel):
    """Payload for PUT /automation-rules/{rule_id}. All fields optional."""

    name: str | None = Field(default=None, min_length=1, max_length=128)
    sensor_id: str | None = None
    actuator_id: str | None = None
    condition: RuleCondition | None = None
    threshold: float | None = None
    action: RuleAction | None = None
    reset_threshold: float | None = None
    reset_action: RuleAction | None = None
    enabled: bool | None = None


class AutomationRuleResponse(AutomationRuleBase):
    """What the API returns for an automation rule."""

    created_at: datetime
    updated_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "rule_id": "RULE_001",
                "name": "Fan on high temp",
                "sensor_id": "TEMP_01",
                "actuator_id": "FAN_01",
                "condition": "GREATER_THAN_OR_EQUAL",
                "threshold": 28.0,
                "action": "ON",
                "reset_threshold": 26.0,
                "reset_action": "OFF",
                "enabled": True,
                "created_at": "2026-08-22T10:00:00Z",
                "updated_at": "2026-08-22T10:00:00Z",
            }
        }
    }


def now_utc() -> datetime:
    """Helper used across models/services for consistent UTC timestamps."""
    return datetime.now(timezone.utc)


# Re-exported for convenience where automation rules reference actuator state
__all__ = [
    "RuleCondition",
    "RuleAction",
    "AutomationRuleBase",
    "AutomationRuleCreate",
    "AutomationRuleUpdate",
    "AutomationRuleResponse",
    "ActuatorState",
    "now_utc",
]
