"""
utils/validators.py
--------------------
Small, reusable validation helpers shared across services/routes.

Pydantic already validates request SHAPE (types, required fields, enums).
These helpers validate business-logic rules that need a database lookup
or apply to multiple entities (e.g. "does this sensor_id already exist").
"""

from motor.motor_asyncio import AsyncIOMotorDatabase

from fastapi import HTTPException, status


async def ensure_unique(
    db: AsyncIOMotorDatabase,
    collection_name: str,
    field: str,
    value: str,
) -> None:
    """
    Raise HTTP 409 if a document with this field/value already exists.
    Used on CREATE for sensor_id, actuator_id, node_id, rule_id, etc.
    """
    existing = await db[collection_name].find_one({field: value})
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{collection_name[:-1]} with {field}='{value}' already exists",
        )


async def ensure_exists(
    db: AsyncIOMotorDatabase,
    collection_name: str,
    field: str,
    value: str,
) -> dict:
    """
    Raise HTTP 404 if no document with this field/value exists.
    Returns the found document otherwise (saves a second query).
    """
    doc = await db[collection_name].find_one({field: value})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{collection_name[:-1]} with {field}='{value}' not found",
        )
    return doc


def validate_range(min_value: float, max_value: float, field_label: str = "range") -> None:
    """Raise HTTP 400 if min_value is not strictly less than max_value."""
    if min_value >= max_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {field_label}: minimum ({min_value}) must be less than maximum ({max_value})",
        )


def serialize_mongo_doc(doc: dict) -> dict:
    """
    Convert a raw MongoDB document into a JSON-serializable dict:
    - drops the internal Mongo _id (we use our own *_id string fields)
    """
    doc = dict(doc)
    doc.pop("_id", None)
    return doc
