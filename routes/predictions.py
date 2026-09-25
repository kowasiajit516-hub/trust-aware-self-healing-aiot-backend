"""
routes/predictions.py
------------------------
FastAPI router for AI fault-detection predictions (Phase 3).

    GET /predictions                    - list predictions (filter: sensor_id, limit)
    GET /predictions/{sensor_id}/latest - most recent prediction for a sensor

Predictions are produced automatically as a side effect of
POST /readings (see services/reading_service.py); there is no manual
POST endpoint here by design - a prediction always corresponds to a
real, stored raw reading that went through the real ingestion path.
"""

from fastapi import APIRouter, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.prediction import PredictionResponse
from services import fault_detection

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("", response_model=list[PredictionResponse])
async def list_predictions(
    sensor_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """List fault-detection predictions, most recent first."""
    return await fault_detection.list_predictions(db, sensor_id, limit)


@router.get("/{sensor_id}/latest", response_model=PredictionResponse)
async def get_latest_prediction(
    sensor_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Get the most recent fault-detection prediction for a given sensor."""
    return await fault_detection.get_latest_prediction(db, sensor_id)
