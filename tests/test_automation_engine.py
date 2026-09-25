"""
tests/test_automation_engine.py
---------------------------------
Phase 9: automation engine tests.

Covers services/automation.py (rule evaluation) and services/actuator.py
(safe application of automated state changes), plus the additive
reading_service.py hook that wires them into the ingestion pipeline.

These tests call the service layer directly with hand-inserted
sensors/actuators/automation_rules/trusted_readings documents, rather
than going through the full ML pipeline (POST /readings would require
trained Isolation Forest + RandomForestRegressor models, which is out
of scope for this test file - see test_fault_detection.py /
test_virtual_sensor.py for those). This mirrors the master plan's core
pipeline rule directly: automation must act ONLY on trusted_readings,
so these tests construct that document explicitly and check what
automation does with it.
"""

import pytest

from services import actuator as actuator_control
from services import automation
from models.actuator import now_utc as actuator_now_utc

pytestmark = pytest.mark.asyncio


async def _insert_actuator(db, actuator_id, mode="AUTO", enabled=True, state="OFF", node_id=None):
    now = actuator_now_utc()
    doc = {
        "actuator_id": actuator_id,
        "name": f"Test {actuator_id}",
        "actuator_type": "FAN",
        "location": "Test Lab",
        "node_id": node_id,
        "mode": mode,
        "enabled": enabled,
        "state": state,
        "last_command": None,
        "controlling_sensor": None,
        "created_at": now,
        "updated_at": now,
    }
    await db["actuators"].insert_one(doc)
    return doc


async def _insert_rule(
    db,
    rule_id,
    sensor_id,
    actuator_id,
    condition,
    threshold,
    action,
    reset_threshold=None,
    reset_action=None,
    enabled=True,
):
    now = actuator_now_utc()
    doc = {
        "rule_id": rule_id,
        "name": f"Test rule {rule_id}",
        "sensor_id": sensor_id,
        "actuator_id": actuator_id,
        "condition": condition,
        "threshold": threshold,
        "action": action,
        "reset_threshold": reset_threshold,
        "reset_action": reset_action,
        "enabled": enabled,
        "created_at": now,
        "updated_at": now,
    }
    await db["automation_rules"].insert_one(doc)
    return doc


# ---------------------------------------------------------------------
# services/actuator.py - apply_automated_state
# ---------------------------------------------------------------------

async def test_apply_automated_state_turns_on_auto_actuator(test_db):
    await _insert_actuator(test_db, "FAN_A1", mode="AUTO", state="OFF")

    result = await actuator_control.apply_automated_state(
        test_db, actuator_id="FAN_A1", desired_state="ON",
        sensor_id="TEMP_A1", trusted_value=31.5, rule_id="RULE_A1",
    )

    assert result is not None
    assert result["state"] == "ON"
    assert result["mode"] == "AUTO", "automation must never change an actuator's mode"

    event = await test_db["actuator_events"].find_one({"actuator_id": "FAN_A1"})
    assert event is not None
    assert event["trusted_value"] == 31.5
    assert event["sensor_id"] == "TEMP_A1"
    assert event["mode"] == "AUTO"


async def test_apply_automated_state_ignores_manual_actuator(test_db):
    await _insert_actuator(test_db, "FAN_A2", mode="MANUAL", state="OFF")

    result = await actuator_control.apply_automated_state(
        test_db, actuator_id="FAN_A2", desired_state="ON",
        sensor_id="TEMP_A2", trusted_value=31.5, rule_id="RULE_A2",
    )

    assert result is None, "a MANUAL actuator must never be touched by automation"
    doc = await test_db["actuators"].find_one({"actuator_id": "FAN_A2"})
    assert doc["state"] == "OFF"
    assert doc["mode"] == "MANUAL"
    assert await test_db["actuator_events"].count_documents({"actuator_id": "FAN_A2"}) == 0


async def test_apply_automated_state_ignores_disabled_actuator(test_db):
    await _insert_actuator(test_db, "FAN_A3", mode="AUTO", enabled=False, state="OFF")

    result = await actuator_control.apply_automated_state(
        test_db, actuator_id="FAN_A3", desired_state="ON",
        sensor_id="TEMP_A3", trusted_value=31.5, rule_id="RULE_A3",
    )

    assert result is None
    doc = await test_db["actuators"].find_one({"actuator_id": "FAN_A3"})
    assert doc["state"] == "OFF"


async def test_apply_automated_state_is_idempotent(test_db):
    await _insert_actuator(test_db, "FAN_A4", mode="AUTO", state="ON")

    result = await actuator_control.apply_automated_state(
        test_db, actuator_id="FAN_A4", desired_state="ON",  # already ON
        sensor_id="TEMP_A4", trusted_value=31.5, rule_id="RULE_A4",
    )

    assert result is None, "already-in-desired-state must be a no-op"
    assert await test_db["actuator_events"].count_documents({"actuator_id": "FAN_A4"}) == 0


async def test_apply_automated_state_missing_actuator_does_not_raise(test_db):
    # No actuator inserted at all.
    result = await actuator_control.apply_automated_state(
        test_db, actuator_id="DOES_NOT_EXIST", desired_state="ON",
        sensor_id="TEMP_A5", trusted_value=31.5, rule_id="RULE_A5",
    )
    assert result is None


# ---------------------------------------------------------------------
# services/automation.py - evaluate_and_apply
# ---------------------------------------------------------------------

async def test_evaluate_and_apply_no_rules_is_silent_noop(test_db):
    # No automation_rules for this sensor at all - must not raise.
    await automation.evaluate_and_apply(
        test_db, {"sensor_id": "NO_RULES_SENSOR", "trusted_value": 99.0}
    )


async def test_evaluate_and_apply_simple_threshold_no_hysteresis(test_db):
    await _insert_actuator(test_db, "FAN_B1", mode="AUTO", state="OFF")
    await _insert_rule(
        test_db, "RULE_B1", sensor_id="TEMP_B1", actuator_id="FAN_B1",
        condition="GREATER_THAN_OR_EQUAL", threshold=28.0, action="ON",
    )

    await automation.evaluate_and_apply(
        test_db, {"sensor_id": "TEMP_B1", "trusted_value": 30.0}
    )
    doc = await test_db["actuators"].find_one({"actuator_id": "FAN_B1"})
    assert doc["state"] == "ON"

    # Below threshold, no hysteresis configured -> toggles back OFF directly.
    await automation.evaluate_and_apply(
        test_db, {"sensor_id": "TEMP_B1", "trusted_value": 20.0}
    )
    doc = await test_db["actuators"].find_one({"actuator_id": "FAN_B1"})
    assert doc["state"] == "OFF"


async def test_evaluate_and_apply_hysteresis_dead_zone(test_db):
    await _insert_actuator(test_db, "FAN_B2", mode="AUTO", state="OFF")
    await _insert_rule(
        test_db, "RULE_B2", sensor_id="TEMP_B2", actuator_id="FAN_B2",
        condition="GREATER_THAN_OR_EQUAL", threshold=28.0, action="ON",
        reset_threshold=26.0, reset_action="OFF",
    )

    # Cross the ON threshold.
    await automation.evaluate_and_apply(
        test_db, {"sensor_id": "TEMP_B2", "trusted_value": 30.0}
    )
    doc = await test_db["actuators"].find_one({"actuator_id": "FAN_B2"})
    assert doc["state"] == "ON"

    # Sit in the dead zone (26 < value < 28) - must stay ON, not flap.
    await automation.evaluate_and_apply(
        test_db, {"sensor_id": "TEMP_B2", "trusted_value": 27.0}
    )
    doc = await test_db["actuators"].find_one({"actuator_id": "FAN_B2"})
    assert doc["state"] == "ON", "dead zone must not change actuator state"

    # Cross the reset threshold.
    await automation.evaluate_and_apply(
        test_db, {"sensor_id": "TEMP_B2", "trusted_value": 24.0}
    )
    doc = await test_db["actuators"].find_one({"actuator_id": "FAN_B2"})
    assert doc["state"] == "OFF"


async def test_evaluate_and_apply_disabled_rule_is_ignored(test_db):
    await _insert_actuator(test_db, "FAN_B3", mode="AUTO", state="OFF")
    await _insert_rule(
        test_db, "RULE_B3", sensor_id="TEMP_B3", actuator_id="FAN_B3",
        condition="GREATER_THAN_OR_EQUAL", threshold=28.0, action="ON",
        enabled=False,
    )

    await automation.evaluate_and_apply(
        test_db, {"sensor_id": "TEMP_B3", "trusted_value": 40.0}
    )
    doc = await test_db["actuators"].find_one({"actuator_id": "FAN_B3"})
    assert doc["state"] == "OFF", "a disabled rule must never fire"


async def test_evaluate_and_apply_does_not_override_manual_actuator(test_db):
    """
    Regression guard for the exact bug this phase exists to avoid:
    an actuator a human has manually turned OFF must stay OFF even
    while its trusted value satisfies an ON rule, because it is no
    longer in AUTO mode.
    """
    await _insert_actuator(test_db, "FAN_B4", mode="MANUAL", state="OFF")
    await _insert_rule(
        test_db, "RULE_B4", sensor_id="TEMP_B4", actuator_id="FAN_B4",
        condition="GREATER_THAN_OR_EQUAL", threshold=28.0, action="ON",
    )

    await automation.evaluate_and_apply(
        test_db, {"sensor_id": "TEMP_B4", "trusted_value": 99.0}
    )
    doc = await test_db["actuators"].find_one({"actuator_id": "FAN_B4"})
    assert doc["state"] == "OFF"
    assert doc["mode"] == "MANUAL"


async def test_evaluate_and_apply_multiple_rules_last_one_wins(test_db):
    """
    Documents the known limitation: if two enabled rules target the
    same actuator, the last one evaluated (created_at order) wins.
    """
    await _insert_actuator(test_db, "FAN_B5", mode="AUTO", state="OFF")
    await _insert_rule(
        test_db, "RULE_B5A", sensor_id="TEMP_B5", actuator_id="FAN_B5",
        condition="GREATER_THAN_OR_EQUAL", threshold=10.0, action="ON",
    )
    await _insert_rule(
        test_db, "RULE_B5B", sensor_id="TEMP_B5", actuator_id="FAN_B5",
        condition="GREATER_THAN_OR_EQUAL", threshold=10.0, action="OFF",
    )

    await automation.evaluate_and_apply(
        test_db, {"sensor_id": "TEMP_B5", "trusted_value": 50.0}
    )
    doc = await test_db["actuators"].find_one({"actuator_id": "FAN_B5"})
    assert doc["state"] == "OFF", "second (later-created) rule must win"
