"""
models/prediction.py
----------------------
Pydantic schemas for Isolation Forest fault-detection predictions.

Stored in the generic `predictions` collection (defined in Phase 1's
database.py indexes). One document per scored raw reading. This is
STRICTLY the AI FAULT DETECTION stage of the core pipeline:

    RAW SENSOR -> VALIDATION -> AI FAULT DETECTION -> DYNAMIC TRUST SCORE -> ...

Trust scoring (Phase 4) and virtual sensing (Phase 5) are separate,
later stages and are NOT computed here.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """A single fault-detection score for one raw reading."""

    id: str = Field(..., description="MongoDB document id as a string")
    sensor_id: str
    reading_id: str = Field(..., description="id of the sensor_readings document this scores")
    value: float
    timestamp: datetime
    anomaly_score: float = Field(
        ..., description="Isolation Forest decision_function score. "
                          "Lower (more negative) = more anomalous."
    )
    is_fault: bool = Field(..., description="True if the model classified this reading as anomalous.")
    model_version: str = Field(..., description="Timestamp of the model file used to score this reading.")

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "id": "66d0f1e2b8f1c2a3d4e5f6a8",
                "sensor_id": "TEMP_01",
                "reading_id": "66d0f1e2b8f1c2a3d4e5f6a7",
                "value": 87.9,
                "timestamp": "2026-08-22T10:00:05Z",
                "anomaly_score": -0.081,
                "is_fault": True,
                "model_version": "2026-08-22T09:00:00+00:00",
            }
        }
    }


def now_utc() -> datetime:
    """Helper used across models/services for consistent UTC timestamps."""
    return datetime.now(timezone.utc)
