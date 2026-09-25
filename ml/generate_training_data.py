"""
ml/generate_training_data.py
------------------------------
Generates a REAL, labeled dataset for evaluating the Isolation Forest
fault detector, by driving the actual Simulator class (Phase 2) against
the actual FastAPI app - never touching MongoDB directly.

Every reading goes through the real POST /readings endpoint (same as
Phase 2's "no simulator-only shortcuts" rule). The only thing this
script adds on top of the simulator is: it also writes down, LOCALLY,
which fault type (if any) it asked the simulator to inject for each
reading, keyed by the reading's real MongoDB-assigned id. That label
is never sent to the API and never stored in sensor_readings - it only
exists in this local file, for offline evaluation of the trained model
(train_isolation_forest.py optionally reads it to compute real
precision/recall instead of only unsupervised anomaly-rate stats).

This mirrors how a real deployment would be evaluated: you can't ask a
real ESP32 sensor "was that reading actually faulty?", so ground truth
for evaluation only ever exists in a controlled, simulated setting like
this one - exactly what Phase 3 needs to evaluate against.

Usage (against a real running server):

    python ml/generate_training_data.py --base-url http://localhost:8000 \\
        --ticks 200 --fault-rate 0.15 --sensors 5

Usage (in-process, no server needed - useful for CI/sandboxes without a
long-running uvicorn process):

    python ml/generate_training_data.py --in-process --ticks 200 \\
        --fault-rate 0.15 --sensors 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.simulator import Simulator  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "datasets" / "labeled_readings.jsonl"

SENSOR_TYPES = [
    "TEMPERATURE", "HUMIDITY", "LIGHT", "GAS", "PRESSURE",
]


def _sensor_payload(i: int) -> dict:
    sensor_type = SENSOR_TYPES[i % len(SENSOR_TYPES)]
    return {
        "sensor_id": f"ML_GEN_{sensor_type}_{i:02d}",
        "name": f"ML Training Sensor {i}",
        "sensor_type": sensor_type,
        "unit": "unit",
        "location": "ML Training Rig",
        "node_id": None,
        "normal_min": 10.0,
        "normal_max": 50.0,
        "sampling_interval_seconds": 5,
        "enabled": True,
    }


async def _ensure_sensors(client: httpx.AsyncClient, n_sensors: int) -> list[str]:
    """Create n_sensors training sensors via the real API (idempotent)."""
    sensor_ids = []
    for i in range(n_sensors):
        payload = _sensor_payload(i)
        sensor_ids.append(payload["sensor_id"])
        resp = await client.post("/sensors", json=payload)
        if resp.status_code not in (201, 409):
            resp.raise_for_status()
    return sensor_ids


async def _generate(client: httpx.AsyncClient, ticks: int, fault_rate: float,
                     n_sensors: int, seed: int | None) -> list[dict]:
    await _ensure_sensors(client, n_sensors)

    rng = random.Random(seed)
    sim = Simulator(client=client, fault_rate=fault_rate, rng=rng)

    labeled_rows: list[dict] = []
    for _ in range(ticks):
        results = await sim.tick()
        for r in results:
            reading = r["reading"]
            labeled_rows.append(
                {
                    "id": reading["id"],
                    "sensor_id": r["sensor_id"],
                    "value": r["value"],
                    "injected_fault": r["injected_fault"],
                }
            )
    return labeled_rows


async def _main_async(args: argparse.Namespace) -> None:
    if args.in_process:
        # Import here so this script has no hard dependency on the app
        # module unless --in-process is actually used.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from main import app  # noqa: E402
        from database import connect_to_mongo, close_mongo_connection  # noqa: E402
        from httpx import ASGITransport

        await connect_to_mongo()
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://in-process") as client:
            rows = await _generate(client, args.ticks, args.fault_rate, args.sensors, args.seed)
        await close_mongo_connection()
    else:
        async with httpx.AsyncClient(base_url=args.base_url, timeout=10.0) as client:
            rows = await _generate(client, args.ticks, args.fault_rate, args.sensors, args.seed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n_faulty = sum(1 for r in rows if r["injected_fault"] is not None)
    print(f"Wrote {len(rows)} labeled readings to {OUTPUT_PATH}")
    print(f"  faulty: {n_faulty}  ({n_faulty / len(rows):.1%})" if rows else "  (no rows generated)")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate labeled training/eval data via the real API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--in-process", action="store_true",
                         help="Run against the FastAPI app in-process (no server needed)")
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--fault-rate", type=float, default=0.15)
    parser.add_argument("--sensors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    asyncio.run(_main_async(_build_arg_parser().parse_args()))
