"""
simulator/simulator.py
-----------------------
Generic sensor simulator.

Rules from the master plan this file must never break:
  - "Simulator must go through the same real backend/API/AI/self-healing
    pipeline as real hardware - no simulator-only shortcuts."
  - Generic, dynamic sensor types only - no per-type hardcoded backend
    logic (fault injection and value generation here read a sensor's
    own normal_min/normal_max, not a hardcoded table of sensor types).

Concretely: this simulator NEVER imports database.py or talks to
MongoDB. It only calls the same POST /readings (and GET /sensors)
endpoints that a real ESP32 node would call over HTTP, via an
httpx.AsyncClient. That client can point at a real running server
(default use) or, in tests, at the FastAPI app in-process via an
ASGI transport - either way, every reading goes through real
Pydantic validation, the real sensor-existence check, and the real
sensor_readings collection.

Fault injection here produces deliberately anomalous RAW values
(spikes, stuck values, drift) so that Phase 3's Isolation Forest has
realistic faulty data to train/evaluate against. Phase 2 itself does
not detect or react to these faults - it only generates and posts
them like a real, occasionally-misbehaving sensor would.

Usage (against a real running server):

    python simulator/simulator.py --base-url http://localhost:8000 \\
        --interval 5 --fault-rate 0.1

Usage (single pass, useful for smoke-testing):

    python simulator/simulator.py --iterations 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger("simulator")


class FaultType(str, Enum):
    """Kinds of raw-sensor misbehavior this simulator can produce."""

    SPIKE = "SPIKE"    # sudden, large out-of-range value
    STUCK = "STUCK"    # sensor stops updating, repeats its last value
    DRIFT = "DRIFT"    # value wanders steadily outside the normal range


@dataclass
class Simulator:
    """
    Generates plausible (and, optionally, faulty) readings for every
    enabled sensor and posts them through the real /readings endpoint.

    `client` must be an httpx.AsyncClient already configured with the
    right base_url (a real server, or an ASGI transport in tests).
    """

    client: httpx.AsyncClient
    fault_rate: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    _last_values: dict[str, float] = field(default_factory=dict, init=False)

    async def fetch_enabled_sensors(self) -> list[dict]:
        """Fetch every enabled sensor via the real GET /sensors endpoint."""
        response = await self.client.get("/sensors", params={"enabled": True})
        response.raise_for_status()
        return response.json()

    def generate_plausible_value(self, sensor: dict) -> float:
        """
        Generate a realistic in-range value for a sensor, using only the
        generic normal_min/normal_max fields every sensor has - never a
        hardcoded per-sensor-type table.
        """
        lo, hi = sensor["normal_min"], sensor["normal_max"]
        span = hi - lo if hi > lo else 1.0
        mid = (lo + hi) / 2

        # Gaussian noise around the midpoint, with a tiny natural
        # over/undershoot allowed so values aren't suspiciously clamped.
        value = self.rng.gauss(mid, span / 6)
        value = max(lo - span * 0.02, min(hi + span * 0.02, value))
        return round(value, 3)

    def inject_fault(self, sensor: dict, base_value: float) -> tuple[float, FaultType]:
        """
        Corrupt a value to simulate one of the fault types above.
        Purely generic - driven by the sensor's own normal_min/normal_max,
        never by sensor_type.
        """
        lo, hi = sensor["normal_min"], sensor["normal_max"]
        span = hi - lo if hi > lo else 1.0
        sensor_id = sensor["sensor_id"]

        fault_type = self.rng.choice(list(FaultType))

        if fault_type is FaultType.SPIKE:
            direction = self.rng.choice([-1, 1])
            edge = hi if direction > 0 else lo
            value = edge + direction * span * self.rng.uniform(0.5, 2.0)

        elif fault_type is FaultType.STUCK:
            value = self._last_values.get(sensor_id, base_value)

        else:  # DRIFT
            direction = self.rng.choice([-1, 1])
            value = base_value + direction * span * self.rng.uniform(0.3, 0.8)

        return round(value, 3), fault_type

    async def tick(self) -> list[dict]:
        """
        Run one simulation cycle: fetch enabled sensors, generate a
        value for each (possibly faulty), and POST each through the
        real /readings endpoint.
        """
        sensors = await self.fetch_enabled_sensors()
        results: list[dict] = []

        for sensor in sensors:
            sensor_id = sensor["sensor_id"]
            base_value = self.generate_plausible_value(sensor)

            injected: FaultType | None = None
            value = base_value
            if self.fault_rate > 0 and self.rng.random() < self.fault_rate:
                value, injected = self.inject_fault(sensor, base_value)

            self._last_values[sensor_id] = value

            response = await self.client.post(
                "/readings", json={"sensor_id": sensor_id, "value": value}
            )
            response.raise_for_status()

            logger.info(
                "sensor=%s value=%s fault=%s",
                sensor_id, value, injected.value if injected else "none",
            )
            results.append(
                {
                    "sensor_id": sensor_id,
                    "value": value,
                    "injected_fault": injected.value if injected else None,
                    "reading": response.json(),
                }
            )

        return results

    async def run(
        self,
        interval_seconds: float = 5.0,
        iterations: int | None = None,
    ) -> None:
        """
        Run the simulation loop. `iterations=None` runs forever
        (Ctrl+C to stop); a positive integer runs that many ticks then
        returns - handy for smoke tests and CI.
        """
        count = 0
        while iterations is None or count < iterations:
            await self.tick()
            count += 1
            if iterations is None or count < iterations:
                await asyncio.sleep(interval_seconds)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic IoT sensor simulator")
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Base URL of the running FastAPI backend (default: %(default)s)",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Seconds between simulation ticks (default: %(default)s)",
    )
    parser.add_argument(
        "--fault-rate", type=float, default=0.0,
        help="Probability (0.0-1.0) that any given reading is faulty (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Number of ticks to run then exit. Omit to run forever.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible simulation runs.",
    )
    return parser


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    args = _build_arg_parser().parse_args()
    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=10.0) as client:
        sim = Simulator(client=client, fault_rate=args.fault_rate, rng=rng)
        logger.info(
            "Starting simulator against %s (interval=%ss, fault_rate=%s)",
            args.base_url, args.interval, args.fault_rate,
        )
        try:
            await sim.run(interval_seconds=args.interval, iterations=args.iterations)
        except KeyboardInterrupt:
            logger.info("Simulator stopped by user")


if __name__ == "__main__":
    asyncio.run(_main())
