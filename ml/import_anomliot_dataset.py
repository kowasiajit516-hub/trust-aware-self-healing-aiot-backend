"""
ml/import_anomliot_dataset.py
--------------------------------
Imports the real, publicly-available AnoML-IoT dataset (Kaggle:
hkayan/anomliot) into this project through the REAL API - same
"no shortcuts" rule the Phase 2 simulator follows: every sensor is
created via POST /sensors and every reading via POST /readings.
Nothing is written to MongoDB directly.

Accepts EITHER the .csv or the .xlsx form of the file - Kaggle serves
this dataset both ways depending on download method. Both are parsed
into the same row format before anything else runs.

Expected columns (as published, regardless of file format):

    Time,Temperature,Humidity,Air Quality,Light,Loudness

- Time is a Unix timestamp in seconds.
- One column = one generic sensor. Loudness has no matching enum value
  in models/sensor.py's SensorType, so it's imported as CUSTOM - still
  fully generic, no special-cased backend logic.
- The published CSV has no per-row anomaly label column (the dataset's
  anomalies were physically induced with an air dryer during known time
  windows described in the dataset's paper/readme, not tagged inline).
  So training on this data is genuinely UNSUPERVISED, same as
  train_isolation_forest.py's default path when no labeled dataset is
  present - which is realistic: a real deployment usually does not have
  ground-truth fault labels either.
- normal_min/normal_max per sensor are derived from the data itself
  (5th/95th percentile of that column), NOT hardcoded per sensor type,
  keeping with the master plan's "no per-type hardcoded logic" rule.

Usage (against a real running server):

    python ml/import_anomliot_dataset.py --file /path/to/anomliot.csv \\
        --base-url http://localhost:8000
    python ml/import_anomliot_dataset.py --file /path/to/anomliot.xlsx \\
        --base-url http://localhost:8000

Usage (in-process, no server needed):

    python ml/import_anomliot_dataset.py --file /path/to/anomliot.csv --in-process
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# column -> (sensor_id, sensor_type, unit)
COLUMN_SENSOR_MAP = {
    "Temperature": ("ANOML_TEMPERATURE", "TEMPERATURE", "°C"),
    "Humidity": ("ANOML_HUMIDITY", "HUMIDITY", "%"),
    "Air Quality": ("ANOML_AIR_QUALITY", "AIR_QUALITY", "AQI"),
    "Light": ("ANOML_LIGHT", "LIGHT", "lux"),
    "Loudness": ("ANOML_LOUDNESS", "CUSTOM", "dB"),
}

BATCH_SIZE = 50  # concurrent POST /readings per batch, keeps the API responsive


def _load_rows(path: Path) -> list[dict]:
    """Load rows from either a .csv or .xlsx file into a list of dicts
    keyed by column header - the rest of the script doesn't care which
    format the file was in."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _load_csv(path)
    elif suffix in (".xlsx", ".xls"):
        rows = _load_excel(path)
    else:
        raise SystemExit(f"Unsupported file type '{suffix}' - expected .csv, .xlsx, or .xls")

    if not rows:
        raise SystemExit(f"No rows found in {path}")
    missing = [c for c in COLUMN_SENSOR_MAP if c not in rows[0]]
    if missing:
        raise SystemExit(
            f"File is missing expected column(s): {missing}. "
            f"Found columns: {list(rows[0].keys())}"
        )
    return rows


def _load_csv(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_excel(xlsx_path: Path) -> list[dict]:
    from openpyxl import load_workbook

    wb = load_workbook(filename=xlsx_path, read_only=True, data_only=True)
    ws = wb.active

    rows_iter = ws.iter_rows(values_only=True)
    headers = [str(h).strip() if h is not None else "" for h in next(rows_iter)]

    rows = []
    for values in rows_iter:
        if values is None or all(v is None for v in values):
            continue
        row = dict(zip(headers, values))
        # Normalize every value to a string so downstream code (which
        # expects CSV-style string values, e.g. float("37.94")) behaves
        # identically regardless of whether the file was .csv or .xlsx.
        rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})
    return rows


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(pct / 100 * (len(values) - 1)))))
    return values[idx]


def _build_sensor_payloads(rows: list[dict]) -> dict[str, dict]:
    """One payload per CSV column, with normal_min/max from real percentiles."""
    payloads = {}
    for column, (sensor_id, sensor_type, unit) in COLUMN_SENSOR_MAP.items():
        values = [float(r[column]) for r in rows if r[column] not in (None, "")]
        lo = _percentile(values, 5)
        hi = _percentile(values, 95)
        if hi <= lo:  # degenerate column (e.g. constant) - widen slightly
            hi = lo + 1.0

        payloads[column] = {
            "sensor_id": sensor_id,
            "name": f"AnoML-IoT {column}",
            "sensor_type": sensor_type,
            "unit": unit,
            "location": "AnoML-IoT dataset (imported)",
            "node_id": None,
            "normal_min": round(lo, 3),
            "normal_max": round(hi, 3),
            "sampling_interval_seconds": 10,
            "enabled": True,
        }
    return payloads


async def _ensure_sensors(client: httpx.AsyncClient, payloads: dict[str, dict]) -> None:
    for column, payload in payloads.items():
        resp = await client.post("/sensors", json=payload)
        if resp.status_code not in (201, 409):
            resp.raise_for_status()
        print(
            f"  sensor {payload['sensor_id']:>20}  "
            f"range=[{payload['normal_min']}, {payload['normal_max']}]  "
            f"(status {resp.status_code})"
        )


async def _post_reading(client: httpx.AsyncClient, sensor_id: str, value: float, timestamp: str) -> None:
    resp = await client.post(
        "/readings", json={"sensor_id": sensor_id, "value": value, "timestamp": timestamp}
    )
    resp.raise_for_status()


async def _import_readings(client: httpx.AsyncClient, rows: list[dict], payloads: dict[str, dict], limit: int | None) -> int:
    if limit is not None:
        rows = rows[:limit]

    n_posted = 0
    batch: list = []
    for row in rows:
        ts = datetime.fromtimestamp(int(row["Time"]), tz=timezone.utc).isoformat()
        for column, payload in payloads.items():
            value = float(row[column])
            batch.append(_post_reading(client, payload["sensor_id"], value, ts))
            if len(batch) >= BATCH_SIZE:
                await asyncio.gather(*batch)
                n_posted += len(batch)
                batch = []
                if n_posted % 500 == 0:
                    print(f"  ... {n_posted} readings imported")
    if batch:
        await asyncio.gather(*batch)
        n_posted += len(batch)

    return n_posted


async def _main_async(args: argparse.Namespace) -> None:
    rows = _load_rows(Path(args.file))
    print(f"Loaded {len(rows)} rows from {args.file}")

    payloads = _build_sensor_payloads(rows)

    if args.in_process:
        from main import app  # noqa: E402
        from database import connect_to_mongo, close_mongo_connection  # noqa: E402
        from httpx import ASGITransport

        await connect_to_mongo()
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://in-process", timeout=30.0) as client:
            print("Creating sensors...")
            await _ensure_sensors(client, payloads)
            print("Importing readings (this can take a while for large CSVs)...")
            n_posted = await _import_readings(client, rows, payloads, args.limit)
        await close_mongo_connection()
    else:
        async with httpx.AsyncClient(base_url=args.base_url, timeout=30.0) as client:
            print("Creating sensors...")
            await _ensure_sensors(client, payloads)
            print("Importing readings (this can take a while for large CSVs)...")
            n_posted = await _import_readings(client, rows, payloads, args.limit)

    print(f"Done. Imported {n_posted} readings across {len(payloads)} sensors.")
    print("Next: python ml/train_isolation_forest.py")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import the real AnoML-IoT dataset via the real API")
    parser.add_argument("--file", required=True, help="Path to the AnoML-IoT file (.csv or .xlsx)")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--in-process", action="store_true",
                         help="Run against the FastAPI app in-process (no server needed)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only import the first N rows (useful for a quick trial run)")
    return parser


if __name__ == "__main__":
    asyncio.run(_main_async(_build_arg_parser().parse_args()))
