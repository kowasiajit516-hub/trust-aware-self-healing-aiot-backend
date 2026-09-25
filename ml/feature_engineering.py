"""
ml/feature_engineering.py
--------------------------
Shared feature computation for the Isolation Forest fault detector.

Used by BOTH:
  - ml/train_isolation_forest.py (training, offline, batch)
  - services/fault_detection.py  (inference, online, one reading at a time)

so that training and serving can never drift apart (train/serve skew).

Per the master plan's non-negotiable rules:
  - Fully generic: features are derived only from a reading's own value,
    its sensor's own normal_min/normal_max, and that sensor's own recent
    history. sensor_type is never used - no per-type hardcoded logic.
  - No data leakage: features for reading i are computed using only
    readings that occurred strictly BEFORE reading i (causal window).
    A reading's own value contributes to its own deviation/z-score (that
    is the entire point of anomaly detection on a single time series),
    but never "sees into the future" and never uses another reading's
    value as if it were its own.
"""

from __future__ import annotations

import statistics

# Rolling window size (number of prior readings) used for the mean/std
# baseline of "what normal looks like recently" for a given sensor.
WINDOW = 10

# Avoids division by zero when a sensor's recent readings are perfectly
# flat (std == 0) - keeps z-scores finite instead of raising/inf.
EPS = 1e-6

# Two readings closer than this (in raw units) count as "the same value"
# for STUCK detection - protects against float rounding noise from a
# genuinely repeated value.
STUCK_EPS = 1e-9

FEATURE_NAMES = ["deviation", "zscore", "delta", "stuck"]


def compute_feature_vector(
    value: float,
    normal_min: float,
    normal_max: float,
    history_values: list[float],
) -> list[float]:
    """
    Compute the 4-feature vector for a single reading.

    Args:
        value: the raw reading value being scored.
        normal_min / normal_max: the sensor's own declared normal range.
        history_values: prior values for this SAME sensor, in
            chronological order (oldest first, most recent last),
            NOT including `value` itself. Pass an empty list for a
            sensor's very first reading.

    Returns:
        [deviation, zscore, delta, stuck] - see FEATURE_NAMES.
    """
    span = normal_max - normal_min if normal_max > normal_min else 1.0

    # deviation: how far outside the declared normal range, normalized
    # by span. 0.0 if comfortably inside the range.
    deviation = max(0.0, normal_min - value, value - normal_max) / span

    if history_values:
        recent = history_values[-WINDOW:]
        mean = statistics.fmean(recent)
        std = statistics.pstdev(recent) if len(recent) > 1 else 0.0
        zscore = abs(value - mean) / (std + EPS)

        previous = history_values[-1]
        delta = abs(value - previous) / span
        stuck = 1.0 if abs(value - previous) < STUCK_EPS else 0.0
    else:
        # No history yet for this sensor - neutral defaults. deviation
        # alone can still flag an out-of-range first reading.
        zscore = 0.0
        delta = 0.0
        stuck = 0.0

    return [deviation, zscore, delta, stuck]


def compute_feature_matrix(
    normal_min: float,
    normal_max: float,
    values_chronological: list[float],
) -> list[list[float]]:
    """
    Compute feature vectors for an entire chronologically-sorted list of
    values from ONE sensor, in one pass. Used by the training script.

    Each feature vector for index i is computed using only
    values_chronological[:i] as history (causal - no lookahead).
    """
    vectors: list[list[float]] = []
    for i, value in enumerate(values_chronological):
        history = values_chronological[:i]
        vectors.append(
            compute_feature_vector(value, normal_min, normal_max, history)
        )
    return vectors
