"""
config.py
---------
Centralized application configuration.
Loads values from environment variables (via a .env file in development).

Works with EITHER:
- Local MongoDB (mongodb://localhost:27017)
- MongoDB Atlas (mongodb+srv://...)

Never hardcode secrets here. Always read from environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------
    # MongoDB
    # ---------------------------------------------------
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "trust_aware_iot"

    # ---------------------------------------------------
    # FastAPI / App
    # ---------------------------------------------------
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Comma separated string from .env, parsed into a list
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ---------------------------------------------------
    # Auth (Phase 7 addition - login/register for the dashboard)
    # ---------------------------------------------------
    jwt_secret_key: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24h

    # ---------------------------------------------------
    # Google Sign-In (optional - leave blank to disable)
    # ---------------------------------------------------
    # Get this from https://console.cloud.google.com/apis/credentials
    # (OAuth 2.0 Client ID, type "Web application"). Until this is set,
    # POST /auth/google returns 503 and the frontend button falls back
    # to its "not configured" message instead of rendering.
    google_client_id: str = ""

    # ---------------------------------------------------
    # Logging
    # ---------------------------------------------------
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a clean list of strings."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


# Single shared settings instance used across the whole backend
settings = Settings()
