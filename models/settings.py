"""
models/settings.py
--------------------
Schema for the one system-wide settings document (a singleton, id
fixed as "system"), covering the trust-engine parameters that are
genuinely read at runtime by services/trust_engine.py.

Deliberately does NOT include:
  - "Anomaly Threshold": the Isolation Forest's fault/not-fault
    boundary comes from `contamination`, a parameter baked into the
    trained model file at training time (services/fault_detection.py
    just calls model.predict()) - there's no live threshold to expose
    here without retraining, so adding a setting for it would control
    nothing real.
  - "Fan ON/OFF Threshold" and "Sensor Timeout": neither has any real
    logic behind it yet (automation-rule evaluation is Phase 9; there
    is no background staleness/timeout watcher) - exposing settings
    for features that don't exist would be a non-functional stub.
  - "API URL" / "Polling Interval": frontend-only concerns (env vars
    and component state), not backend runtime configuration.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class SettingsUpdate(BaseModel):
    """Payload for PUT /settings. All fields optional (partial update)."""

    fault_penalty: float | None = Field(
        default=None, ge=0, le=100, description="Trust points subtracted per is_fault=True reading"
    )
    recovery_gain: float | None = Field(
        default=None, ge=0, le=100, description="Trust points added per is_fault=False reading"
    )
    faulty_threshold: float | None = Field(
        default=None, ge=0, le=100, description="trust_score below this -> FAULTY"
    )
    healthy_threshold: float | None = Field(
        default=None, ge=0, le=100, description="trust_score at/above this -> HEALTHY"
    )
    trust_max: float | None = Field(
        default=None, ge=0, le=100, description="Ceiling for trust_score, and the initial value for new sensors"
    )
    trust_min: float | None = Field(default=None, ge=0, le=100, description="Floor for trust_score")


class SettingsResponse(BaseModel):
    """What GET /settings and PUT /settings return."""

    fault_penalty: float
    recovery_gain: float
    faulty_threshold: float
    healthy_threshold: float
    trust_max: float
    trust_min: float
    updated_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "fault_penalty": 20.0,
                "recovery_gain": 5.0,
                "faulty_threshold": 50.0,
                "healthy_threshold": 90.0,
                "trust_max": 100.0,
                "trust_min": 0.0,
                "updated_at": "2026-09-07T10:00:00Z",
            }
        }
    }


# The real defaults - identical to the hardcoded constants
# services/trust_engine.py used before this feature existed, so
# behavior is unchanged until someone actually edits a setting.
DEFAULT_SETTINGS = {
    "id": "system",
    "fault_penalty": 20.0,
    "recovery_gain": 5.0,
    "faulty_threshold": 50.0,
    "healthy_threshold": 90.0,
    "trust_max": 100.0,
    "trust_min": 0.0,
}
