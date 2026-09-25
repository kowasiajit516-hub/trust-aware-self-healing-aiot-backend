"""
ml/train_random_forest.py
---------------------------
Trains a REAL sklearn RandomForestRegressor for Phase 5's VIRTUAL
SENSOR stage:

    ... -> DYNAMIC TRUST SCORE -> VIRTUAL SENSOR -> TRUSTED VALUE -> ...

Per the master plan's non-negotiable rules:
  - Real, trained, saved/loaded model - no hardcoded predictions.
  - Generic: features never depend on sensor_type (see
    ml/virtual_sensor_features.py).
  - No data leakage (rule #5): a faulty reading's own value is never
    used as a FEATURE for its own target (the feature module already
    guarantees this structurally). Going further than the strict
    minimum, this trainer also EXCLUDES any reading labeled faulty
    (via labeled_readings.jsonl, when available - see
    ml/generate_training_data.py) from the historical VALUE SERIES
    entirely, so a faulty spike can't even leak into a *neighboring*
    reading's lag/rolling features via training.

Bootstrapping note: `trusted_readings` does not exist yet the first
time this script runs (Phase 5 hasn't produced any yet). This script
therefore trains on raw `sensor_readings`, filtered down to only
readings NOT labeled faulty (see above) - a reasonable stand-in for
"trusted history" for the first training pass. Once the real pipeline
has been running (writing to `trusted_readings`), a future retrain
could read from `trusted_readings` directly instead - not required for
Phase 5's scope, and not built here.

Reads directly from MongoDB with a synchronous pymongo client - same
offline-batch-job pattern as ml/train_isolation_forest.py (this is not
part of the request path, so it does not need Motor/async).

Usage:
    python ml/train_random_forest.py
    python ml/train_random_forest.py --n-estimators 300 --max-depth 12
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from ml.virtual_sensor_features import FEATURE_NAMES, compute_virtual_feature_matrix  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "random_forest.joblib"
META_PATH = MODELS_DIR / "random_forest_meta.json"
LABELED_DATA_PATH = Path(__file__).resolve().parent / "datasets" / "labeled_readings.jsonl"

MIN_READINGS_REQUIRED = 30
TRAIN_FRACTION = 0.8


def _load_readings_by_sensor(db) -> tuple[dict, dict]:
    """Load every reading, grouped by sensor_id, sorted chronologically."""
    sensors = {s["sensor_id"]: s for s in db.sensors.find({})}
    readings_by_sensor: dict[str, list[dict]] = defaultdict(list)

    cursor = db.sensor_readings.find({}).sort([("sensor_id", 1), ("timestamp", 1), ("_id", 1)])
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        readings_by_sensor[doc["sensor_id"]].append(doc)

    return sensors, readings_by_sensor


def _load_fault_labels() -> dict[str, bool]:
    """id -> True if labeled faulty, from generate_training_data.py's output. Empty dict if unavailable."""
    if not LABELED_DATA_PATH.exists():
        return {}
    labels: dict[str, bool] = {}
    with LABELED_DATA_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[row["id"]] = row["injected_fault"] is not None
    return labels


def build_dataset(
    sensors: dict, readings_by_sensor: dict, fault_labels: dict
) -> tuple[list, list, list, list, list]:
    """
    Returns (train_X, train_y, eval_X, eval_y, eval_sensor_ids).

    For each sensor: drop any reading labeled faulty from its value
    series entirely (see module docstring), then chronologically split
    the remaining "clean" series - first 80% -> train, last 20% ->
    eval (held out, never trained on) - the same leakage-avoidance
    shape ml/train_isolation_forest.py uses.
    """
    train_X: list[list[float]] = []
    train_y: list[float] = []
    eval_X: list[list[float]] = []
    eval_y: list[float] = []
    eval_sensor_ids: list[str] = []

    for sensor_id, readings in readings_by_sensor.items():
        sensor = sensors.get(sensor_id)
        if sensor is None:
            continue

        clean = [r for r in readings if not fault_labels.get(r["id"], False)]
        if len(clean) < 2:
            continue

        timestamps = [r["timestamp"] for r in clean]
        values = [r["value"] for r in clean]
        X, y = compute_virtual_feature_matrix(
            sensor["normal_min"], sensor["normal_max"], timestamps, values
        )

        split_idx = max(1, int(len(clean) * TRAIN_FRACTION))
        train_X.extend(X[:split_idx])
        train_y.extend(y[:split_idx])

        for i in range(split_idx, len(clean)):
            eval_X.append(X[i])
            eval_y.append(y[i])
            eval_sensor_ids.append(sensor_id)

    return train_X, train_y, eval_X, eval_y, eval_sensor_ids


def evaluate(model: RandomForestRegressor, eval_X: list[list[float]], eval_y: list[float]) -> dict:
    if not eval_X:
        return {"n_eval": 0}
    preds = model.predict(eval_X)
    return {
        "n_eval": len(eval_X),
        "mae_normalized": round(float(mean_absolute_error(eval_y, preds)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    client = MongoClient(settings.mongodb_uri)
    db = client[settings.mongodb_db_name]

    sensors, readings_by_sensor = _load_readings_by_sensor(db)
    total_readings = sum(len(v) for v in readings_by_sensor.values())

    if total_readings < MIN_READINGS_REQUIRED:
        client.close()
        raise SystemExit(
            f"Only {total_readings} sensor_readings found (need >= {MIN_READINGS_REQUIRED}).\n"
            "Run the simulator first, e.g.:\n"
            "  python ml/generate_training_data.py --in-process --ticks 200 --fault-rate 0.15"
        )

    fault_labels = _load_fault_labels()
    train_X, train_y, eval_X, eval_y, eval_sensor_ids = build_dataset(
        sensors, readings_by_sensor, fault_labels
    )
    client.close()

    if len(train_X) < MIN_READINGS_REQUIRED:
        raise SystemExit(
            f"Only {len(train_X)} clean training rows after filtering + chronological split "
            f"(need >= {MIN_READINGS_REQUIRED}). Generate more data first, or run "
            "ml/generate_training_data.py so faulty readings can be excluded by label."
        )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(train_X, train_y)

    eval_metrics = evaluate(model, eval_X, eval_y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "feature_names": FEATURE_NAMES,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "n_train_samples": len(train_X),
        "n_sensors_trained_on": len(readings_by_sensor),
        "n_faulty_readings_excluded": sum(1 for v in fault_labels.values() if v),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "eval": eval_metrics,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print("Trained RandomForestRegressor virtual sensor.")
    print(f"  train samples: {len(train_X)}  (from {len(readings_by_sensor)} sensors)")
    print(f"  faulty readings excluded from training series: {meta['n_faulty_readings_excluded']}")
    print(f"  eval: {eval_metrics if eval_metrics.get('n_eval') else '(no eval rows)'}")
    print(f"  saved model:  {MODEL_PATH}")
    print(f"  saved meta:   {META_PATH}")


if __name__ == "__main__":
    main()
