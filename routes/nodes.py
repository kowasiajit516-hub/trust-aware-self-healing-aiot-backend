"""
routes/nodes.py
----------------
FastAPI router for Node (ESP32) CRUD endpoints, plus real ESP32
communication added in Phase 8.

    POST   /nodes
    GET    /nodes
    GET    /nodes/{node_id}
    PUT    /nodes/{node_id}
    DELETE /nodes/{node_id}

    POST   /nodes/{node_id}/heartbeat               (Phase 8)
    GET    /nodes/{node_id}/commands/pending         (Phase 8)
    POST   /nodes/{node_id}/commands/{command_id}/ack (Phase 8)

Phase 1 scope was the CRUD skeleton only. Phase 8 adds the real
heartbeat / online-offline detection and the actuator-command queue
a real ESP32 node polls and acknowledges - see
services/node_service.py and services/node_command_service.py.
"""

from fastapi import APIRouter, Depends, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.node import NodeCreate, NodeUpdate, NodeResponse, NodeHeartbeat
from models.node_command import NodeCommandResponse
from services import node_service, node_command_service

router = APIRouter(prefix="/nodes", tags=["Nodes"])


@router.post("", response_model=NodeResponse, status_code=status.HTTP_201_CREATED)
async def create_node(
    payload: NodeCreate,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Register a new ESP32 node."""
    return await node_service.create_node(db, payload)


@router.get("", response_model=list[NodeResponse])
async def list_nodes(
    enabled: bool | None = Query(default=None),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """List all registered nodes, optionally filtered by enabled status."""
    return await node_service.list_nodes(db, enabled)


@router.get("/{node_id}", response_model=NodeResponse)
async def get_node(
    node_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Get a single node by its node_id."""
    return await node_service.get_node(db, node_id)


@router.put("/{node_id}", response_model=NodeResponse)
async def update_node(
    node_id: str,
    payload: NodeUpdate,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Edit an existing node (partial update)."""
    return await node_service.update_node(db, node_id, payload)


@router.delete("/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_node(
    node_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Delete a node."""
    await node_service.delete_node(db, node_id)


@router.post("/{node_id}/heartbeat", response_model=NodeResponse)
async def node_heartbeat(
    node_id: str,
    payload: NodeHeartbeat | None = None,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Real ESP32 nodes call this on a fixed interval (Phase 8) to report
    that they are alive. Sets status=ONLINE and last_seen=now. An empty
    body ({}) is valid; sending ip_address lets a DHCP node self-report
    a changed address.
    """
    ip_address = payload.ip_address if payload else None
    return await node_service.record_heartbeat(db, node_id, ip_address)


@router.get("/{node_id}/commands/pending", response_model=list[NodeCommandResponse])
async def get_pending_commands(
    node_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Real ESP32 nodes poll this to pick up actuator commands issued via
    POST /actuators/{id}/command for any actuator whose node_id matches
    this node. Commands stay PENDING until acknowledged below.
    """
    await node_service.get_node(db, node_id)  # 404s cleanly if node doesn't exist
    return await node_command_service.list_pending_commands(db, node_id)


@router.post(
    "/{node_id}/commands/{command_id}/ack", response_model=NodeCommandResponse
)
async def acknowledge_command(
    node_id: str,
    command_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Real ESP32 nodes call this once a queued command has actually been applied."""
    return await node_command_service.acknowledge_command(db, node_id, command_id)
