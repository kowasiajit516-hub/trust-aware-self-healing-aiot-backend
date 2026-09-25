"""
routes/automation_rules.py
----------------------------
FastAPI router for Automation Rule CRUD endpoints.

    POST   /automation-rules
    GET    /automation-rules
    GET    /automation-rules/{rule_id}
    PUT    /automation-rules/{rule_id}
    DELETE /automation-rules/{rule_id}

Phase 1 scope: CRUD skeleton only. The rule EVALUATION engine that
actually reads trusted sensor values and drives actuator state is
implemented in Phase 9.
"""

from fastapi import APIRouter, Depends, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.automation_rule import (
    AutomationRuleCreate,
    AutomationRuleUpdate,
    AutomationRuleResponse,
)
from services import automation_rule_service

router = APIRouter(prefix="/automation-rules", tags=["Automation Rules"])


@router.post("", response_model=AutomationRuleResponse, status_code=status.HTTP_201_CREATED)
async def create_rule(
    payload: AutomationRuleCreate,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Create a new automation rule linking a sensor to an actuator."""
    return await automation_rule_service.create_rule(db, payload)


@router.get("", response_model=list[AutomationRuleResponse])
async def list_rules(
    sensor_id: str | None = Query(default=None),
    actuator_id: str | None = Query(default=None),
    enabled: bool | None = Query(default=None),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """List automation rules, optionally filtered by sensor/actuator/enabled."""
    return await automation_rule_service.list_rules(db, sensor_id, actuator_id, enabled)


@router.get("/{rule_id}", response_model=AutomationRuleResponse)
async def get_rule(
    rule_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Get a single automation rule by its rule_id."""
    return await automation_rule_service.get_rule(db, rule_id)


@router.put("/{rule_id}", response_model=AutomationRuleResponse)
async def update_rule(
    rule_id: str,
    payload: AutomationRuleUpdate,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Edit an existing automation rule (partial update)."""
    return await automation_rule_service.update_rule(db, rule_id, payload)


@router.delete("/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(
    rule_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Delete an automation rule."""
    await automation_rule_service.delete_rule(db, rule_id)
