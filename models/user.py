"""
models/user.py
---------------
Pydantic schemas for the `users` collection - real dashboard
authentication (Phase 7 addition).

Password hashing is real (bcrypt via passlib, see utils/security.py) -
plaintext passwords are never stored. Tokens are real signed JWTs.
There is no per-dashboard-endpoint auth enforcement here: existing
Phase 1-6 API routes (sensors, actuators, readings, ...) stay
unauthenticated so the simulator and any real ESP32 node can keep
posting readings without needing a login flow - only the frontend
dashboard's own login/register/reset screens are backed by this.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"


class UserRegister(BaseModel):
    """Payload for POST /auth/register"""

    full_name: str = Field(..., min_length=1, max_length=128, examples=["Admin User"])
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128, examples=["StrongPass123"])


class UserLogin(BaseModel):
    """Payload for POST /auth/login"""

    email: EmailStr
    password: str


class ForgotPasswordRequest(BaseModel):
    """Payload for POST /auth/forgot-password"""

    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Payload for POST /auth/reset-password"""

    token: str
    new_password: str = Field(..., min_length=8, max_length=128)


class GoogleLoginRequest(BaseModel):
    """Payload for POST /auth/google - a Google Identity Services ID token."""

    id_token: str = Field(..., description="The credential/ID token returned by Google Identity Services")


class UserResponse(BaseModel):
    """What the API returns for a user (never includes the password hash)."""

    user_id: str = Field(..., description="MongoDB document id as a string")
    full_name: str
    email: EmailStr
    role: UserRole = UserRole.ADMIN
    email_verified: bool = False
    auth_provider: str = Field(
        default="local", description="'local' (email+password) or 'google'"
    )
    created_at: datetime
    updated_at: datetime


class TokenResponse(BaseModel):
    """What the API returns after a successful login/register."""

    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserResponse


def now_utc() -> datetime:
    """Helper used across models/services for consistent UTC timestamps."""
    return datetime.now(timezone.utc)
