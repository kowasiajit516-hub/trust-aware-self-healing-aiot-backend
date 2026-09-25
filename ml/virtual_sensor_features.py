"""
ml/virtual_sensor_features.py
-------------------------------
Shared, causal feature computation for the VIRTUAL SENSOR stage
(RandomForestRegressor), used by BOTH:
  - ml/train_random_forest.py   (training, offline, batch)
  - services/virtual_sensor.py  (inference, online, one prediction at a time)

Deliberately a SEPARATE module from ml/feature_engineering.py (the
Isolation Forest's feature module) - per HANDOVER_04's explicit
instruction not to modify that file, and because the two models solve
different problems with different leakage constraints:

  - Isolation Forest (fault detection) legitimately uses the CURRENT
    reading's own value - that is the entire point of scoring a single
    point as anomalous or not.
  - This module NEVER uses the current/target reading's own value as
    an input feature. Phase 5 predicts what a sensor's value SHOULD be
    without ever looking at the (possibly faulty) value it may end up
    replacing - master plan rule #5 ("a faulty sensor's own faulty
    value must never be fed back into predicting itself").

Generic (never keyed by sensor_type): features come only from a
sensor's own recent history (normalized against that sensor's own
normal_min/normal_max, exactly like ml/feature_engineering.py does)
and the timestamp being predicted for (time-of-day is known in
advance - using it is not leakage).
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime

# Rolling window size (number of prior values) used for the
# mean/std/lag baseline - mirrors ml/feature_engineering.py's WINDOW.
WINDOW = 10

FEATURE_NAMES = [
    "lag_1", "lag_2", "lag_3",
    "rolling_mean", "rolling_std", "trend",
    "hour_sin", "hour_cos",
]

# Neutral fallback (normalized units) for a sensor with no history yet -
# the midpoint of its own normal range, not 0 in raw units, which would
# fabricate a bias toward the low end of an arbitrary sensor's scale.
NEUTRAL_NORMALIZED = 0.5


def _normalize(value: float, normal_min: float, normal_max: float) -> float:
    span = normal_max - normal_min if normal_max > normal_min else 1.0
    return (value - normal_min) / span


def denormalize(normalized: float, normal_min: float, normal_max: float) -> float:
    """
    Inverse of the normalization used internally by this module and by
    compute_virtual_feature_matrix's targets. Public - services/
    virtual_sensor.py uses this to convert a model's normalized
    prediction back into the sensor's own raw units.
    """
    span = normal_max - normal_min if normal_max > normal_min else 1.0
    return normalized * span + normal_min


def _time_features(timestamp: datetime) -> tuple[float, float]:
    """Cyclic hour-of-day encoding - generic, sensor_type-agnostic."""
    hour_frac = timestamp.hour + timestamp.minute / 60.0
    angle = 2 * math.pi * hour_frac / 24.0
    return math.sin(angle), math.cos(angle)


def compute_virtual_features(
    timestamp: datetime,
    normal_min: float,
    normal_max: float,
    history_values: list[float],
) -> list[float]:
    """
    Compute the feature vector used to predict a sensor's TRUSTED value
    at `timestamp`.

    Args:
        timestamp: the timestamp of the reading being predicted FOR.
            Only used for time-of-day features - never used to look up
            future data.
        normal_min / normal_max: the sensor's own declared normal range.
        history_values: prior TRUSTED values for this SAME sensor, in
            chronological order (oldest first, most recent last), all
            strictly before `timestamp`, NOT including the value being
            predicted/replaced - this is what makes the feature set
            leakage-free. Pass an empty list for a sensor with no
            trusted history yet.

    Returns:
        [lag_1, lag_2, lag_3, rolling_mean, rolling_std, trend,
         hour_sin, hour_cos] - see FEATURE_NAMES.
    """
    normalized_history = [_normalize(v, normal_min, normal_max) for v in history_values]

    if normalized_history:
        recent = normalized_history[-WINDOW:]
        lag_1 = recent[-1]
        lag_2 = recent[-2] if len(recent) >= 2 else lag_1
        lag_3 = recent[-3] if len(recent) >= 3 else lag_2
        rolling_mean = statistics.fmean(recent)
        rolling_std = statistics.pstdev(recent) if len(recent) > 1 else 0.0
        trend = lag_1 - lag_2
    else:
        # No trusted history yet (sensor's very first reading) - neutral
        # midpoint defaults rather than fabricating a trend.
        lag_1 = lag_2 = lag_3 = NEUTRAL_NORMALIZED
        rolling_mean = NEUTRAL_NORMALIZED
        rolling_std = 0.0
        trend = 0.0

    hour_sin, hour_cos = _time_features(timestamp)

    return [lag_1, lag_2, lag_3, rolling_mean, rolling_std, trend, hour_sin, hour_cos]


def compute_virtual_feature_matrix(
    normal_min: float,
    normal_max: float,
    timestamps_chronological: list[datetime],
    values_chronological: list[float],
) -> tuple[list[list[float]], list[float]]:
    """
    Build (X, y) supervised training pairs for an entire
    chronologically-sorted series of values from ONE sensor, in one
    pass. Used by the training script.

    Row i's features use only values_chronological[:i] as history
    (causal - no lookahead) and the label is the normalized value AT
    index i - i.e. this is a one-step-ahead forecaster: "given what
    came before, what should this reading be".
    """
    X: list[list[float]] = []
    y: list[float] = []
    for i, (ts, value) in enumerate(zip(timestamps_chronological, values_chronological)):
        history = values_chronological[:i]
        X.append(compute_virtual_features(ts, normal_min, normal_max, history))
        y.append(_normalize(value, normal_min, normal_max))
    return X, y
