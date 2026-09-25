"""
services/auth_service.py
--------------------------
Business logic for dashboard authentication (Phase 7 addition).

Real bcrypt password hashing, real signed JWTs, real reset-token flow
stored in a generic `password_reset_tokens` collection (reset tokens
are stored hashed, never in plaintext, and expire after 30 minutes).

There is no outbound email service in this project, so
forgot_password() cannot actually deliver a reset link - it returns
the raw token directly in the API response instead of silently
pretending an email was sent. This is flagged clearly in the endpoint
docstring/response so it's never mistaken for production email
delivery; wiring a real email provider is a follow-up, not something
faked here.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from models.user import UserRegister, UserRole, now_utc
from utils.security import hash_password, verify_password, create_access_token
from utils.validators import serialize_mongo_doc

logger = logging.getLogger(__name__)

USERS_COLLECTION = "users"
RESET_TOKENS_COLLECTION = "password_reset_tokens"
RESET_TOKEN_TTL_MINUTES = 30


def _user_to_response(doc: dict) -> dict:
    return {
        "user_id": str(doc["_id"]),
        "full_name": doc["full_name"],
        "email": doc["email"],
        "role": doc.get("role", UserRole.ADMIN.value),
        "email_verified": doc.get("email_verified", False),
        "auth_provider": doc.get("auth_provider", "local"),
        "created_at": doc["created_at"],
        "updated_at": doc["updated_at"],
    }


async def register(db: AsyncIOMotorDatabase, payload: UserRegister) -> tuple[dict, str]:
    """Create a new user with a bcrypt-hashed password. Returns (user, token)."""
    existing = await db[USERS_COLLECTION].find_one({"email": payload.email.lower()})
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An account with email '{payload.email}' already exists",
        )

    now = now_utc()
    doc = {
        "full_name": payload.full_name,
        "email": payload.email.lower(),
        "password_hash": hash_password(payload.password),
        "role": UserRole.ADMIN.value,
        "email_verified": False,
        "auth_provider": "local",
        "created_at": now,
        "updated_at": now,
    }
    result = await db[USERS_COLLECTION].insert_one(doc)
    doc["_id"] = result.inserted_id

    logger.info("User registered: %s", payload.email)
    user = _user_to_response(doc)
    token = create_access_token(subject=user["user_id"])
    return user, token


async def login(db: AsyncIOMotorDatabase, email: str, password: str) -> tuple[dict, str]:
    """Verify credentials and return (user, token). 401 on any mismatch."""
    doc = await db[USERS_COLLECTION].find_one({"email": email.lower()})
    if doc is None or doc.get("password_hash") is None or not verify_password(
        password, doc["password_hash"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    user = _user_to_response(doc)
    token = create_access_token(subject=user["user_id"])
    logger.info("User logged in: %s", email)
    return user, token


async def google_login(db: AsyncIOMotorDatabase, id_token_str: str) -> tuple[dict, str]:
    """
    Verify a Google Identity Services ID token server-side, then find or
    create a matching user and return (user, token) - same JWT shape as
    a normal login, so the frontend treats both identically.

    Real verification via google-auth (checks the token's signature
    against Google's public keys and confirms `aud` matches our
    configured client ID) - never trusts the token's claims blindly.
    """
    from config import settings

    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google sign-in is not configured on this server",
        )

    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token as google_id_token

    try:
        idinfo = google_id_token.verify_oauth2_token(
            id_token_str, google_requests.Request(), settings.google_client_id
        )
    except ValueError as e:
        # Log the REAL reason server-side (wrong audience, expired token,
        # wrong issuer, etc.) - never expose the raw message to the client,
        # but this is exactly what's needed to diagnose a mismatch.
        logger.warning("Google token verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google credential",
        )

    email = idinfo.get("email")
    if not email or not idinfo.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google account has no verified email",
        )

    now = now_utc()
    email = email.lower()
    doc = await db[USERS_COLLECTION].find_one({"email": email})

    if doc is None:
        doc = {
            "full_name": idinfo.get("name") or email.split("@")[0],
            "email": email,
            "password_hash": None,  # Google-only account, no local password
            "role": UserRole.ADMIN.value,
            "email_verified": True,
            "auth_provider": "google",
            "google_sub": idinfo.get("sub"),
            "created_at": now,
            "updated_at": now,
        }
        result = await db[USERS_COLLECTION].insert_one(doc)
        doc["_id"] = result.inserted_id
        logger.info("New user registered via Google: %s", email)
    else:
        logger.info("User logged in via Google: %s", email)

    user = _user_to_response(doc)
    token = create_access_token(subject=user["user_id"])
    return user, token


async def get_current_user(db: AsyncIOMotorDatabase, user_id: str) -> dict:
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        oid = ObjectId(user_id)
    except InvalidId:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    doc = await db[USERS_COLLECTION].find_one({"_id": oid})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return _user_to_response(doc)


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


async def forgot_password(db: AsyncIOMotorDatabase, email: str) -> str | None:
    """
    Generate a password-reset token if the email belongs to a real
    account. Stores only the token's hash (never the raw token) with a
    30-minute expiry. Returns the RAW token so the caller (route layer)
    can hand it back in the response - there is no email service in
    this project to deliver it any other way; see module docstring.

    Returns None if no account matches (the route still responds 200
    either way, so this endpoint can't be used to enumerate which
    emails are registered).
    """
    doc = await db[USERS_COLLECTION].find_one({"email": email.lower()})
    if doc is None:
        return None

    raw_token = secrets.token_urlsafe(32)
    now = now_utc()
    await db[RESET_TOKENS_COLLECTION].insert_one(
        {
            "user_id": doc["_id"],
            "token_hash": _hash_token(raw_token),
            "expires_at": now + timedelta(minutes=RESET_TOKEN_TTL_MINUTES),
            "used": False,
            "created_at": now,
        }
    )
    logger.info("Password reset token issued for %s", email)
    return raw_token


async def reset_password(db: AsyncIOMotorDatabase, raw_token: str, new_password: str) -> None:
    """Consume a valid, unexpired, unused reset token and set a new password hash."""
    token_hash = _hash_token(raw_token)
    record = await db[RESET_TOKENS_COLLECTION].find_one({"token_hash": token_hash})

    now = now_utc()
    if (
        record is None
        or record["used"]
        or record["expires_at"].replace(tzinfo=timezone.utc) < now
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This reset link is invalid or has expired",
        )

    await db[USERS_COLLECTION].update_one(
        {"_id": record["user_id"]},
        {"$set": {"password_hash": hash_password(new_password), "updated_at": now}},
    )
    await db[RESET_TOKENS_COLLECTION].update_one(
        {"_id": record["_id"]}, {"$set": {"used": True}}
    )
    logger.info("Password reset completed for user_id=%s", record["user_id"])
