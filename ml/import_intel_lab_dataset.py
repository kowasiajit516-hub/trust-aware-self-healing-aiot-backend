"""
ml/import_intel_lab_dataset.py
---------------------------------
Imports the real Intel Berkeley Research Lab sensor dataset into this
project through the REAL API - same "no shortcuts" rule the Phase 2
simulator follows: every sensor (and mote/node) is created via
POST /sensors (and POST /nodes) and every reading via POST /readings.
Nothing is written to MongoDB directly.

Expected file format (whitespace-delimited .txt, as published):

    date time epoch moteid temperature humidity light voltage

e.g.:
    2004-02-28 00:59:16.02785 3 1 19.9884 37.0933 45.08 2.69964

Each (moteid, field) pair becomes its own generic sensor - e.g. mote 1's
temperature and mote 2's temperature are two separate sensors, matching
how physically separate hardware would really be modeled. moteid maps
to node_id (a Node is created per mote, matching the master plan's
Node entity).

This real dataset is known to contain genuinely malformed/missing rows
and out-of-range physical values (e.g. negative humidity, >100C
temperature) - real sensor faults, not injected. Malformed rows
(wrong field count, non-numeric values) are skipped and counted, never
crashing the import.

The full published file is very large (millions of rows across ~54
motes over about a month). Default flags below import a small, tractable
subset - override with --mote-ids / --limit-per-mote / --max-motes to
pull in more.

Usage (against a real running server):

    python ml/import_intel_lab_dataset.py --txt /path/to/data.txt \\
        --base-url http://localhost:8000

Usage (in-process, no server needed):

    python ml/import_intel_lab_dataset.py --txt /path/to/data.txt --in-process

Import specific motes / more rows:

    python ml/import_intel_lab_dataset.py --txt /path/to/data.txt --in-process \\
        --mote-ids 1,2,3,4,5,10,15 --limit-per-mote 5000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# field name -> (sensor_type, unit)
FIELD_SENSOR_MAP = {
    "temperature": ("TEMPERATURE", "°C"),
    "humidity": ("HUMIDITY", "%"),
    "light": ("LIGHT", "lux"),
    "voltage": ("VOLTAGE", "V"),
}

BATCH_SIZE = 50
DEFAULT_MAX_MOTES = 5
DEFAULT_LIMIT_PER_MOTE = 2000


def _parse_line(line: str) -> dict | None:
    """
    Parse one whitespace-delimited row. Returns None (and lets the
    caller count it) for any malformed row - this real dataset is
    known to have missing fields and garbage rows, and the importer
    must not crash on them.
    """
    parts = line.split()
    if len(parts) != 8:
        return None
    date, time_, epoch, moteid, temp, hum, light, volt = parts
    try:
        ts = datetime.strptime(f"{date} {time_}", "%Y-%m-%d %H:%M:%S.%f")
        ts = ts.replace(tzinfo=timezone.utc)
        return {
            "timestamp": ts.isoformat(),
            "moteid": int(moteid),
            "temperature": float(temp),
            "humidity": float(hum),
            "light": float(light),
            "voltage": float(volt),
        }
    except (ValueError, TypeError):
        return None


def _load_txt(txt_path: Path, max_motes: int, limit_per_mote: int, mote_ids: set[int] | None) -> tuple[dict[int, list[dict]], int]:
    """
    Stream the file (it can be huge) and keep only rows for the
    selected motes, up to limit_per_mote rows each, in the order
    encountered (the file is already roughly chronological).
    """
    rows_by_mote: dict[int, list[dict]] = defaultdict(list)
    n_malformed = 0
    selected_motes: set[int] = set(mote_ids) if mote_ids else set()

    with txt_path.open(errors="replace") as f:
        for line in f:
            row = _parse_line(line)
            if row is None:
                n_malformed += 1
                continue

            mote = row["moteid"]

            if mote_ids is None:
                # auto-select the first `max_motes` distinct motes seen
                if mote not in selected_motes and len(selected_motes) >= max_motes:
                    continue
                selected_motes.add(mote)
            elif mote not in selected_motes:
                continue

            if len(rows_by_mote[mote]) >= limit_per_mote:
                continue
            rows_by_mote[mote].append(row)

            if mote_ids is None and len(selected_motes) >= max_motes and all(
                len(rows_by_mote[m]) >= limit_per_mote for m in selected_motes
            ):
                break

    return rows_by_mote, n_malformed


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(pct / 100 * (len(values) - 1)))))
    return values[idx]


def _build_payloads(rows_by_mote: dict[int, list[dict]]) -> tuple[dict[int, dict], dict[tuple[int, str], dict]]:
    """Returns (node_payloads keyed by moteid, sensor_payloads keyed by (moteid, field))."""
    node_payloads = {}
    sensor_payloads = {}

    for mote, rows in rows_by_mote.items():
        node_id = f"INTEL_MOTE_{mote:02d}"
        node_payloads[mote] = {
            "node_id": node_id,
            "name": f"Intel Lab Mote {mote}",
            "location": "Intel Berkeley Research Lab (imported)",
        }

        for field, (sensor_type, unit) in FIELD_SENSOR_MAP.items():
            values = [r[field] for r in rows]
            lo = _percentile(values, 5)
            hi = _percentile(values, 95)
            if hi <= lo:
                hi = lo + 1.0

            sensor_payloads[(mote, field)] = {
                "sensor_id": f"INTEL_{field.upper()}_M{mote:02d}",
                "name": f"Intel Lab Mote {mote} {field.capitalize()}",
                "sensor_type": sensor_type,
                "unit": unit,
                "location": "Intel Berkeley Research Lab (imported)",
                "node_id": node_id,
                "normal_min": round(lo, 3),
                "normal_max": round(hi, 3),
                "sampling_interval_seconds": 30,
                "enabled": True,
            }

    return node_payloads, sensor_payloads


async def _ensure_nodes(client: httpx.AsyncClient, node_payloads: dict) -> None:
    for mote, payload in node_payloads.items():
        resp = await client.post("/nodes", json=payload)
        if resp.status_code not in (201, 409):
            resp.raise_for_status()


async def _ensure_sensors(client: httpx.AsyncClient, sensor_payloads: dict) -> None:
    for key, payload in sensor_payloads.items():
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


async def _import_readings(client: httpx.AsyncClient, rows_by_mote: dict, sensor_payloads: dict) -> int:
    n_posted = 0
    batch: list = []
    for mote, rows in rows_by_mote.items():
        for row in rows:
            for field in FIELD_SENSOR_MAP:
                sensor_id = sensor_payloads[(mote, field)]["sensor_id"]
                batch.append(_post_reading(client, sensor_id, row[field], row["timestamp"]))
                if len(batch) >= BATCH_SIZE:
                    await asyncio.gather(*batch)
                    n_posted += len(batch)
                    batch = []
                    if n_posted % 1000 == 0:
                        print(f"  ... {n_posted} readings imported")
    if batch:
        await asyncio.gather(*batch)
        n_posted += len(batch)
    return n_posted


async def _run(args: argparse.Namespace) -> None:
    mote_ids = set(int(x) for x in args.mote_ids.split(",")) if args.mote_ids else None

    print(f"Scanning {args.txt} ...")
    rows_by_mote, n_malformed = _load_txt(
        Path(args.txt), args.max_motes, args.limit_per_mote, mote_ids
    )
    total_rows = sum(len(v) for v in rows_by_mote.values())
    print(f"Loaded {total_rows} valid rows across {len(rows_by_mote)} motes "
          f"(skipped {n_malformed} malformed rows).")
    if not rows_by_mote:
        raise SystemExit("No usable rows found - check the file format/path.")

    node_payloads, sensor_payloads = _build_payloads(rows_by_mote)

    if args.in_process:
        from main import app  # noqa: E402
        from database import connect_to_mongo, close_mongo_connection  # noqa: E402
        from httpx import ASGITransport

        await connect_to_mongo()
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://in-process", timeout=30.0) as client:
            print("Creating nodes + sensors...")
            await _ensure_nodes(client, node_payloads)
            await _ensure_sensors(client, sensor_payloads)
            print("Importing readings (this can take a while)...")
            n_posted = await _import_readings(client, rows_by_mote, sensor_payloads)
        await close_mongo_connection()
    else:
        async with httpx.AsyncClient(base_url=args.base_url, timeout=30.0) as client:
            print("Creating nodes + sensors...")
            await _ensure_nodes(client, node_payloads)
            await _ensure_sensors(client, sensor_payloads)
            print("Importing readings (this can take a while)...")
            n_posted = await _import_readings(client, rows_by_mote, sensor_payloads)

    print(f"Done. Imported {n_posted} readings across {len(sensor_payloads)} sensors "
          f"({len(rows_by_mote)} motes).")
    print("Next: python ml/train_isolation_forest.py")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import the real Intel Berkeley Lab dataset via the real API")
    parser.add_argument("--txt", required=True, help="Path to the Intel Lab data.txt file")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--in-process", action="store_true",
                         help="Run against the FastAPI app in-process (no server needed)")
    parser.add_argument("--mote-ids", default=None,
                         help="Comma-separated mote ids to import, e.g. 1,2,3. "
                              "Default: auto-select the first --max-motes distinct motes seen.")
    parser.add_argument("--max-motes", type=int, default=DEFAULT_MAX_MOTES,
                         help=f"Used only when --mote-ids is omitted (default: {DEFAULT_MAX_MOTES})")
    parser.add_argument("--limit-per-mote", type=int, default=DEFAULT_LIMIT_PER_MOTE,
                         help=f"Max rows imported per mote (default: {DEFAULT_LIMIT_PER_MOTE})")
    return parser


if __name__ == "__main__":
    asyncio.run(_run(_build_arg_parser().parse_args()))
