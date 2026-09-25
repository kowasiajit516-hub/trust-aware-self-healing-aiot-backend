"""
services/automation.py
------------------------
AUTOMATION ENGINE (Phase 9) - the final stage of the core pipeline:

    ... -> VIRTUAL SENSOR -> TRUSTED VALUE -> AUTOMATION ENGINE -> ACTUATOR

Reads automation_rules (CRUD already existed since Phase 1, via
services/automation_rule_service.py) and evaluates them against ONE
freshly-computed trusted_readings document - NEVER a raw
sensor_readings value (master plan rule #2, the hardest rule in this
project). Actuators are only ever driven through
services/actuator.apply_automated_state, which refuses to touch any
actuator that is not currently in AUTO mode - a human's MANUAL
override (via POST /actuators/{id}/command) always wins over
automation, with no exceptions.

Called as a best-effort step from services/reading_service.py, right
after virtual_sensor.compute_trusted_value has already succeeded - the
same "never break POST /readings" contract every downstream pipeline
stage follows since Phase 3
(fault_detection -> trust_engine -> virtual_sensor -> automation).

RULE EVALUATION MODEL
----------------------
Each ENABLED rule for the reading's sensor_id is evaluated independently
against the SAME trusted_value:

  - No reset_threshold/reset_action set (no hysteresis):
        condition true  -> actuator commanded to `action`
        condition false -> actuator commanded to the OPPOSITE of `action`
    i.e. the actuator simply tracks the condition directly, toggling
    immediately at the single threshold.

  - reset_threshold/reset_action set (hysteresis):
        condition true (vs threshold)                     -> `action`
        reverse-condition true (vs reset_threshold)        -> `reset_action`
        neither                                            -> no change
    e.g. "ON at >=28, OFF at <=26" leaves the actuator wherever it
    already was for 26 < trusted_value < 28 (the dead zone), avoiding
    rapid ON/OFF flapping right at a single threshold.

KNOWN LIMITATION - multiple rules on one actuator: if more than one
enabled rule targets the same actuator_id, rules are evaluated in
created_at order and the LAST rule to produce a non-None desired
command wins. The master plan does not define rule-priority/conflict
resolution and the dashboard's rules UI (Phase 7) doesn't expose a
priority field, so Phase 9 does not invent one. Flagged explicitly in
the Phase 9 handover for the next chat to revisit if it becomes a real
requirement.
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

AUTOMATION_RULES_COLLECTION = "automation_rules"

_REVERSE_CONDITION = {
    "GREATER_THAN": "LESS_THAN",
    "GREATER_THAN_OR_EQUAL": "LESS_THAN_OR_EQUAL",
    "LESS_THAN": "GREATER_THAN",
    "LESS_THAN_OR_EQUAL": "GREATER_THAN_OR_EQUAL",
    "EQUAL": "EQUAL",
}

_OPPOSITE_ACTION = {"ON": "OFF", "OFF": "ON"}


def _check(condition: str, value: float, threshold: float) -> bool:
    if condition == "GREATER_THAN":
        return value > threshold
    if condition == "GREATER_THAN_OR_EQUAL":
        return value >= threshold
    if condition == "LESS_THAN":
        return value < threshold
    if condition == "LESS_THAN_OR_EQUAL":
        return value <= threshold
    if condition == "EQUAL":
        return value == threshold
    raise ValueError(f"Unknown rule condition: {condition}")


def _desired_command_for_rule(rule: dict, trusted_value: float) -> str | None:
    """
    Resolve one rule's desired ON/OFF command for this trusted_value,
    or None if this rule wants no change (inside a hysteresis dead
    zone). Pure function - no I/O - so it's directly unit-testable.
    """
    condition = rule["condition"]
    threshold = rule["threshold"]
    action = rule["action"]

    if _check(condition, trusted_value, threshold):
        return action

    reset_threshold = rule.get("reset_threshold")
    reset_action = rule.get("reset_action")

    if reset_threshold is not None and reset_action is not None:
        reverse_condition = _REVERSE_CONDITION[condition]
        if _check(reverse_condition, trusted_value, reset_threshold):
            return reset_action
        return None  # inside the hysteresis dead zone - leave actuator as-is

    # No hysteresis configured: track the condition directly.
    return _OPPOSITE_ACTION[action]


async def evaluate_and_apply(db: AsyncIOMotorDatabase, trusted_reading: dict) -> None:
    """
    Phase 9 entrypoint. `trusted_reading` is the dict returned by
    services.virtual_sensor.compute_trusted_value for ONE reading -
    must contain sensor_id and trusted_value. Evaluates every enabled
    rule for that sensor_id and applies the result via
    services.actuator.apply_automated_state.

    Best-effort: never raises. Caller
    (reading_service._try_evaluate_automation) treats any failure here
    as log-and-continue, same as every other downstream pipeline stage.
    """
    sensor_id = trusted_reading["sensor_id"]
    trusted_value = trusted_reading["trusted_value"]

    cursor = (
        db[AUTOMATION_RULES_COLLECTION]
        .find({"sensor_id": sensor_id, "enabled": True})
        .sort("created_at", 1)
    )
    rules = [rule async for rule in cursor]
    if not rules:
        return

    desired_by_actuator: dict[str, dict] = {}
    for rule in rules:
        try:
            desired = _desired_command_for_rule(rule, trusted_value)
        except ValueError:
            logger.exception(
                "Automation rule %s has an invalid condition; skipping.",
                rule.get("rule_id"),
            )
            continue

        if desired is not None:
            # Last rule targeting this actuator wins - see module docstring.
            desired_by_actuator[rule["actuator_id"]] = {
                "desired": desired,
                "rule_id": rule["rule_id"],
            }

    if not desired_by_actuator:
        return

    from services import actuator as actuator_control

    for actuator_id, info in desired_by_actuator.items():
        try:
            await actuator_control.apply_automated_state(
                db,
                actuator_id=actuator_id,
                desired_state=info["desired"],
                sensor_id=sensor_id,
                trusted_value=trusted_value,
                rule_id=info["rule_id"],
            )
        except Exception:
            logger.exception(
                "Failed to apply automation for actuator %s (rule %s, sensor %s); "
                "trusted value was still stored normally.",
                actuator_id, info["rule_id"], sensor_id,
            )
