"""
services/actuator.py
---------------------
Generic ACTUATOR CONTROL logic - the AUTOMATION ENGINE's write path.

This is deliberately a SEPARATE file from services/actuator_service.py:

  - actuator_service.py owns MANUAL commands
    (POST /actuators/{id}/command, issued by a human via the dashboard
    or curl). ON/OFF always force mode=MANUAL there - a human's choice
    always sticks until they explicitly send AUTO again.

  - This file (services/actuator.py) owns AUTOMATED state changes,
    issued only by services/automation.py (Phase 9's rule-evaluation
    engine), driven only by TRUSTED sensor values (master plan rule
    #2). It NEVER changes an actuator's mode and NEVER acts on an
    actuator that is not already in AUTO mode - a human's MANUAL
    override always wins over automation, with no exceptions.

Nothing here is called from any route directly - it exists to be
called from services/automation.py after a rule evaluation decides an
actuator's state should change.
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from models.actuator import ActuatorMode, ActuatorState, now_utc
from utils.validators import serialize_mongo_doc

logger = logging.getLogger(__name__)

COLLECTION = "actuators"
EVENTS_COLLECTION = "actuator_events"


async def apply_automated_state(
    db: AsyncIOMotorDatabase,
    actuator_id: str,
    desired_state: str,
    sensor_id: str,
    trusted_value: float,
    rule_id: str,
) -> dict | None:
    """
    Apply a state change decided by the automation engine (Phase 9),
    driven ONLY by a TRUSTED sensor value - never a raw one (rule #2).
    The caller (services/automation.py) resolves WHICH state is
    desired; this function is responsible for SAFELY applying it:

      - No-op (returns None) if the actuator no longer exists.
      - No-op (returns None) if the actuator is disabled.
      - No-op (returns None) if the actuator is not currently in AUTO
        mode - a human's MANUAL override always wins; automation must
        never fight it or silently flip it back.
      - No-op (returns None) if the actuator is already in the desired
        state - avoids a redundant actuator_events write (and a
        redundant node dispatch) on every single matching reading.
      - Otherwise: updates `actuators.state` (mode stays AUTO,
        untouched), records a real actuator_events entry with
        sensor_id/trusted_value populated (closing the
        `"trusted_value": None` placeholder left in
        actuator_service.send_command since Phase 1), and best-effort
        dispatches to the actuator's ESP32 node - same as a manual
        command (Phase 8) - so automated control reaches real
        hardware too.

    Never raises. Called from a best-effort pipeline stage
    (services/automation.py, itself best-effort from
    reading_service.py) - a failure here must never break
    POST /readings.
    """
    doc = await db[COLLECTION].find_one({"actuator_id": actuator_id})
    if doc is None:
        logger.warning(
            "Automation rule %s references missing actuator_id='%s'; skipping.",
            rule_id, actuator_id,
        )
        return None

    if not doc.get("enabled", True):
        logger.debug("Actuator %s is disabled; automation skipped.", actuator_id)
        return None

    if doc.get("mode") != ActuatorMode.AUTO.value:
        logger.debug(
            "Actuator %s is in %s mode, not AUTO; automation skipped "
            "(a human's manual command takes precedence).",
            actuator_id, doc.get("mode"),
        )
        return None

    previous_state = doc.get("state", ActuatorState.OFF.value)
    if previous_state == desired_state:
        return None  # already in the desired state - nothing to do

    now = now_utc()
    await db[COLLECTION].update_one(
        {"actuator_id": actuator_id},
        {
            "$set": {
                "state": desired_state,
                "last_command": desired_state,
                "controlling_sensor": sensor_id,
                "updated_at": now,
            }
        },
    )

    await db[EVENTS_COLLECTION].insert_one(
        {
            "actuator_id": actuator_id,
            "command": desired_state,
            "previous_state": previous_state,
            "new_state": desired_state,
            "mode": ActuatorMode.AUTO.value,
            "reason": (
                f"Automation rule {rule_id}: sensor {sensor_id} "
                f"trusted_value={trusted_value}"
            ),
            "sensor_id": sensor_id,
            "trusted_value": trusted_value,
            "timestamp": now,
        }
    )

    logger.info(
        "Automation applied: actuator=%s %s -> %s (rule=%s sensor=%s trusted_value=%.4f)",
        actuator_id, previous_state, desired_state, rule_id, sensor_id, trusted_value,
    )

    await _try_dispatch_to_node(db, doc, actuator_id, desired_state)

    result = await db[COLLECTION].find_one({"actuator_id": actuator_id})
    return serialize_mongo_doc(result)


async def _try_dispatch_to_node(
    db: AsyncIOMotorDatabase, actuator: dict, actuator_id: str, command: str
) -> None:
    """
    Best-effort delivery to the actuator's real ESP32 node - same
    contract as actuator_service._try_dispatch_to_node (Phase 8),
    duplicated here (rather than imported) so this file's dependency
    surface stays self-contained and doesn't reach into another
    service's private helpers. Keeps automated commands reaching real
    hardware exactly the way manual ones do.
    """
    node_id = actuator.get("node_id")
    if not node_id:
        return

    from services import node_command_service
    from esp32 import esp32_client

    try:
        queued = await node_command_service.enqueue_command(
            db, node_id, actuator_id, command
        )
    except Exception:
        logger.exception(
            "Failed to queue automated node command for actuator %s (node %s); "
            "the actuator state itself was still applied normally.",
            actuator_id, node_id,
        )
        return

    try:
        node = await db["nodes"].find_one({"node_id": node_id})
        if node:
            await esp32_client.push_command_to_node(
                node, actuator_id, command, queued["id"]
            )
    except Exception:
        logger.exception(
            "Best-effort direct push to node %s failed; automated command %s "
            "remains queued for the node's next poll.",
            node_id, queued.get("id"),
        )
