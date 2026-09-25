"""
check_fault.py
-----------------
One-off sanity check: post a deliberately absurd reading through the
real /readings endpoint and confirm the AI fault detector (Phase 3)
actually flags it via /predictions.

Run from inside backend/, with the server already running:

    python check_fault.py
"""

import httpx

health = httpx.get("http://localhost:8000/health", timeout=10.0)
print("GET /health ->", health.status_code, health.json())

r = httpx.post(
    "http://localhost:8000/readings",
    json={"sensor_id": "INTEL_TEMPERATURE_M01", "value": 250.0},
    timeout=30.0,  # first call after training loads the model from disk - give it room
)
print("POST /readings ->", r.status_code, r.json())

r2 = httpx.get("http://localhost:8000/predictions/INTEL_TEMPERATURE_M01/latest", timeout=30.0)
print("GET /predictions/.../latest ->", r2.status_code, r2.json())
