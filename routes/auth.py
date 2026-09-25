"""
routes/auth.py
----------------
Real dashboard authentication endpoints (Phase 7 addition):

    POST /auth/register         - create account, returns a JWT
    POST /auth/login            - verify credentials, returns a JWT
    POST /auth/google           - verify a Google Identity Services ID token, returns a JWT
    GET  /auth/me                - resolve the current JWT to a user
    POST /auth/forgot-password  - issue a reset token (see note below)
    POST /auth/reset-password   - consume a reset token, set new password

Only these dashboard-auth endpoints are protected/token-aware. Every
existing Phase 1-6 endpoint (sensors, actuators, readings, ...) stays
unauthenticated on purpose, so the simulator and real ESP32 nodes keep
working exactly as before (rule #11: never break previous phases).

NOTE on forgot-password: this project has no outbound email service.
POST /auth/forgot-password therefore returns the raw reset token
directly in its JSON response (dev-mode behavior, clearly documented),
instead of pretending an email was sent. Wiring a real email provider
(SendGrid/SES/etc.) is a follow-up, not something faked here.
"""

from fastapi import APIRouter, Depends, Header, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from models.user import (
    UserRegister,
    UserLogin,
    UserResponse,
    TokenResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    GoogleLoginRequest,
)
from services import auth_service
from utils.security import decode_access_token
from config import settings

router = APIRouter(prefix="/auth", tags=["Auth"])


async def get_current_user_dependency(
    authorization: str | None = Header(default=None),
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> dict:
    """Resolve `Authorization: Bearer <token>` into the current user. 401 if invalid/missing."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1].strip()
    user_id = decode_access_token(token)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    return await auth_service.get_current_user(db, user_id)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: UserRegister, db: AsyncIOMotorDatabase = Depends(get_database)):
    """Create a new dashboard account and return a signed access token."""
    user, token = await auth_service.register(db, payload)
    return {
        "access_token": token,
        "expires_in_minutes": settings.jwt_expire_minutes,
        "user": user,
    }


@router.post("/login", response_model=TokenResponse)
async def login(payload: UserLogin, db: AsyncIOMotorDatabase = Depends(get_database)):
    """Verify email/password and return a signed access token."""
    user, token = await auth_service.login(db, payload.email, payload.password)
    return {
        "access_token": token,
        "expires_in_minutes": settings.jwt_expire_minutes,
        "user": user,
    }


@router.post("/google", response_model=TokenResponse)
async def google_login(payload: GoogleLoginRequest, db: AsyncIOMotorDatabase = Depends(get_database)):
    """
    Verify a Google Identity Services ID token and return a signed
    access token, same shape as /auth/login and /auth/register. If a
    user with that Google account's email already exists (registered
    via email/password), they're logged into that same account. If
    not, a new account is created automatically (auth_provider="google",
    no password set).
    """
    user, token = await auth_service.google_login(db, payload.id_token)
    return {
        "access_token": token,
        "expires_in_minutes": settings.jwt_expire_minutes,
        "user": user,
    }


@router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user_dependency)):
    """Return the account associated with the current bearer token."""
    return current_user


@router.post("/forgot-password")
async def forgot_password(
    payload: ForgotPasswordRequest, db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Issue a password-reset token. Always returns 200 regardless of
    whether the email is registered (prevents email enumeration) - but
    dev_reset_token is only populated when it is, since there's no
    email service to deliver it otherwise (see module docstring).
    """
    token = await auth_service.forgot_password(db, payload.email)
    return {
        "message": "If that email is registered, a reset token was generated.",
        "dev_reset_token": token,
    }


@router.post("/reset-password")
async def reset_password(
    payload: ResetPasswordRequest, db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Consume a valid reset token and set a new password."""
    await auth_service.reset_password(db, payload.token, payload.new_password)
    return {"message": "Password has been reset. You can now log in."}
