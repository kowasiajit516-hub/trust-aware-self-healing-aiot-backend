"""
main.py
-------
FastAPI application entrypoint.

Wires together:
- logging setup
- MongoDB connection lifecycle (connect on startup, close on shutdown)
- CORS
- all routers: sensors, actuators, nodes, automation-rules, readings,
  predictions (Phase 3), sensor-health (Phase 4), trusted-readings
  (Phase 5), fault-events (Phase 6)

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then open:
    http://localhost:8000/docs   (interactive Swagger UI)
    http://localhost:8000/health (basic health check)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import connect_to_mongo, close_mongo_connection
from utils.logger import setup_logging

from routes.sensors import router as sensors_router
from routes.actuators import router as actuators_router
from routes.nodes import router as nodes_router
from routes.automation_rules import router as automation_rules_router
from routes.readings import router as readings_router
from routes.predictions import router as predictions_router
from routes.sensor_health import router as sensor_health_router
from routes.trusted_readings import router as trusted_readings_router
from routes.fault_events import router as fault_events_router
from routes.actuator_events import router as actuator_events_router
from routes.auth import router as auth_router
from routes.fault_injection import router as fault_injection_router
from routes.settings import router as settings_router

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    logger.info("Starting Trust-Aware Self-Healing IoT backend (env=%s)", settings.app_env)
    await connect_to_mongo()
    yield
    # ---- Shutdown ----
    await close_mongo_connection()
    logger.info("Backend shutdown complete")


app = FastAPI(
    title="Trust-Aware Self-Healing for IoT Sensor Networks",
    description=(
        "Using Machine Learning and Virtual Sensing. "
        "Phase 7: dynamic dashboard (frontend/) consuming the real "
        "API built in Phases 1-6, a read-only GET /actuator-events "
        "endpoint exposing the actuator_events collection (written "
        "since Phase 1 but never previously readable), real dashboard "
        "auth (POST /auth/register, /auth/login, /auth/forgot-password, "
        "/auth/reset-password, GET /auth/me - existing Phase 1-6 routes "
        "stay unauthenticated so the simulator keeps working), and a "
        "real POST /fault-injection/{sensor_id} endpoint that drives "
        "the same SPIKE/STUCK/DRIFT logic simulator.py uses through "
        "the real ingestion pipeline (no shortcuts)."
    ),
    version="0.7.0-phase7",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    # Dev/LAN: also allow the frontend served from localhost or a private
    # LAN IP (e.g. http://192.168.0.107:5173 from a phone), any port.
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Routers
# ---------------------------------------------------
app.include_router(sensors_router)
app.include_router(actuators_router)
app.include_router(nodes_router)
app.include_router(automation_rules_router)
app.include_router(readings_router)
app.include_router(predictions_router)
app.include_router(sensor_health_router)
app.include_router(trusted_readings_router)
app.include_router(fault_events_router)
app.include_router(actuator_events_router)
app.include_router(auth_router)
app.include_router(fault_injection_router)
app.include_router(settings_router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint - confirms the API process is up."""
    return {
        "status": "ok",
        "service": "trust-aware-self-healing-iot-backend",
        "phase": "Phase 7",
        "env": settings.app_env,
    }


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Trust-Aware Self-Healing for IoT Sensor Networks - API is running",
        "docs": "/docs",
    }