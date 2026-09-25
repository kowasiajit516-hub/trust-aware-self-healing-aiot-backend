"""
backend/esp32/esp32_client.py
-------------------------------
Outbound HTTP client for reaching a real ESP32 node directly (Phase 8).

This is a best-effort LATENCY OPTIMIZATION only, not the primary
delivery mechanism. The primary, reliable mechanism is the node
polling GET /nodes/{node_id}/commands/pending itself (see
routes/nodes.py + services/node_command_service.py) - that queue is
what guarantees a command is never lost even if:
  - the node has no ip_address on record yet,
  - the node is temporarily unreachable or behind NAT,
  - this direct push fails or times out for any reason.

firmware/esp32_node/esp32_node.ino runs a tiny local HTTP server on
port 80 exposing POST /command specifically so this client has
somewhere to push to. Applying the same ON/OFF/AUTO command twice
(once via direct push, once again via the poll loop) is intentionally
idempotent and harmless - the firmware always still acks via the poll
loop even if a direct push already landed, so node_commands stays an
accurate audit trail either way.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

PUSH_TIMEOUT_SECONDS = 2.0


async def push_command_to_node(
    node: dict, actuator_id: str, command: str, command_id: str
) -> bool:
    """
    Best-effort direct push of one command to a node's local HTTP
    server. Returns True/False for logging purposes only - callers
    must never let a False result (or any exception - this function
    catches its own) affect the actuator command's HTTP response.
    """
    ip_address = node.get("ip_address")
    if not ip_address:
        logger.debug(
            "No ip_address on record for node %s; skipping direct push, "
            "command %s will be picked up on the node's next poll.",
            node.get("node_id"), command_id,
        )
        return False

    url = f"http://{ip_address}/command"
    payload = {
        "actuator_id": actuator_id,
        "command": command,
        "command_id": command_id,
    }

    try:
        async with httpx.AsyncClient(timeout=PUSH_TIMEOUT_SECONDS) as http_client:
            response = await http_client.post(url, json=payload)
        if response.status_code >= 400:
            logger.warning(
                "Node %s rejected direct push (status=%s); command %s "
                "remains queued for the next poll.",
                node.get("node_id"), response.status_code, command_id,
            )
            return False
        logger.info(
            "Direct push delivered to node %s: %s", node.get("node_id"), payload
        )
        return True
    except httpx.HTTPError as exc:
        logger.info(
            "Direct push to node %s failed (%s); command %s remains queued "
            "for the next poll. This is expected/normal if the node is "
            "offline or its ip_address is stale.",
            node.get("node_id"), exc, command_id,
        )
        return False
