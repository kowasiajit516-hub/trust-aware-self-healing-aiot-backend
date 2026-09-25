"""
utils/security.py
-------------------
Real password hashing (bcrypt, direct - not via passlib, to avoid a
known passlib<->bcrypt>=4.1 backend-detection incompatibility) and JWT
signing/verification (python-jose). No plaintext passwords are ever
stored or logged.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from config import settings


def hash_password(plain_password: str) -> str:
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
    """Create a signed JWT whose subject ("sub") is the user_id."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes if expires_minutes is not None else settings.jwt_expire_minutes
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> str | None:
    """Return the user_id encoded in a valid, non-expired token, else None."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload.get("sub")
    except JWTError:
        return None
