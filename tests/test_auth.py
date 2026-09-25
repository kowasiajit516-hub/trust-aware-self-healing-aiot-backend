"""
tests/test_auth.py
--------------------
Tests for real dashboard authentication (Phase 7 addition):
register/login/me/forgot-password/reset-password.

Run with:
    pytest tests/test_auth.py -v
"""

import pytest

from tests.conftest import sample_user_payload


@pytest.mark.asyncio
async def test_register_creates_user_and_returns_token(client):
    resp = await client.post("/auth/register", json=sample_user_payload())
    assert resp.status_code == 201
    data = resp.json()
    assert data["access_token"]
    assert data["user"]["email"] == "test.user@example.com"
    assert "password" not in data["user"]
    assert "password_hash" not in data["user"]


@pytest.mark.asyncio
async def test_register_duplicate_email_fails(client):
    payload = sample_user_payload("dup@example.com")
    resp1 = await client.post("/auth/register", json=payload)
    assert resp1.status_code == 201

    resp2 = await client.post("/auth/register", json=payload)
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_login_with_correct_credentials_succeeds(client):
    payload = sample_user_payload("login.ok@example.com")
    await client.post("/auth/register", json=payload)

    resp = await client.post(
        "/auth/login", json={"email": payload["email"], "password": payload["password"]}
    )
    assert resp.status_code == 200
    assert resp.json()["access_token"]


@pytest.mark.asyncio
async def test_login_with_wrong_password_fails(client):
    payload = sample_user_payload("login.bad@example.com")
    await client.post("/auth/register", json=payload)

    resp = await client.post(
        "/auth/login", json={"email": payload["email"], "password": "WrongPassword1"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_email_fails(client):
    resp = await client.post(
        "/auth/login", json={"email": "nobody@example.com", "password": "whatever123"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_returns_user_for_valid_token(client):
    payload = sample_user_payload("me.valid@example.com")
    register_resp = await client.post("/auth/register", json=payload)
    token = register_resp.json()["access_token"]

    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == payload["email"]


@pytest.mark.asyncio
async def test_me_rejects_missing_token(client):
    resp = await client.get("/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_rejects_garbage_token(client):
    resp = await client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_forgot_password_returns_token_for_known_email(client):
    payload = sample_user_payload("forgot.known@example.com")
    await client.post("/auth/register", json=payload)

    resp = await client.post("/auth/forgot-password", json={"email": payload["email"]})
    assert resp.status_code == 200
    assert resp.json()["dev_reset_token"] is not None


@pytest.mark.asyncio
async def test_forgot_password_does_not_leak_unknown_email(client):
    resp = await client.post("/auth/forgot-password", json={"email": "ghost@example.com"})
    assert resp.status_code == 200
    assert resp.json()["dev_reset_token"] is None


@pytest.mark.asyncio
async def test_reset_password_with_valid_token_then_login_with_new_password(client):
    payload = sample_user_payload("reset.flow@example.com")
    await client.post("/auth/register", json=payload)

    forgot_resp = await client.post("/auth/forgot-password", json={"email": payload["email"]})
    token = forgot_resp.json()["dev_reset_token"]

    reset_resp = await client.post(
        "/auth/reset-password", json={"token": token, "new_password": "BrandNewPass1"}
    )
    assert reset_resp.status_code == 200

    old_login = await client.post(
        "/auth/login", json={"email": payload["email"], "password": payload["password"]}
    )
    assert old_login.status_code == 401

    new_login = await client.post(
        "/auth/login", json={"email": payload["email"], "password": "BrandNewPass1"}
    )
    assert new_login.status_code == 200


@pytest.mark.asyncio
async def test_reset_password_rejects_invalid_token(client):
    resp = await client.post(
        "/auth/reset-password", json={"token": "garbage", "new_password": "AnythingHere1"}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_reset_password_token_is_single_use(client):
    payload = sample_user_payload("reset.reuse@example.com")
    await client.post("/auth/register", json=payload)
    forgot_resp = await client.post("/auth/forgot-password", json={"email": payload["email"]})
    token = forgot_resp.json()["dev_reset_token"]

    first = await client.post(
        "/auth/reset-password", json={"token": token, "new_password": "FirstNewPass1"}
    )
    assert first.status_code == 200

    second = await client.post(
        "/auth/reset-password", json={"token": token, "new_password": "SecondNewPass1"}
    )
    assert second.status_code == 400


@pytest.mark.asyncio
async def test_existing_sensor_routes_stay_unauthenticated(client):
    """Rule #11: Phase 1-6 routes must not start requiring auth."""
    resp = await client.get("/sensors")
    assert resp.status_code == 200


# ---------------------------------------------------------------------
# Google Sign-In
#
# We can't call real Google servers from a test suite, so the one thing
# mocked here is the cryptographic verification step itself
# (google_id_token.verify_oauth2_token) - everything downstream (our
# account creation, account linking, JWT issuance) runs for real
# against the real service/route code, exactly like every other test
# in this file.
# ---------------------------------------------------------------------

GOOGLE_CLAIMS = {
    "sub": "109876543210987654321",
    "email": "google.user@example.com",
    "email_verified": True,
    "name": "Google User",
}


@pytest.mark.asyncio
async def test_google_login_disabled_returns_503_when_unconfigured(client, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "google_client_id", "")
    resp = await client.post("/auth/google", json={"id_token": "whatever"})
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_google_login_creates_new_account(client, monkeypatch):
    from config import settings
    from services import auth_service

    monkeypatch.setattr(settings, "google_client_id", "test-client-id")
    monkeypatch.setattr(
        auth_service.google_id_token, "verify_oauth2_token", lambda *a, **k: GOOGLE_CLAIMS
    )

    resp = await client.post("/auth/google", json={"id_token": "fake-but-verified-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["access_token"]
    assert data["user"]["email"] == "google.user@example.com"
    assert data["user"]["auth_provider"] == "google"
    assert data["user"]["email_verified"] is True


@pytest.mark.asyncio
async def test_google_login_is_idempotent_for_same_user(client, monkeypatch):
    """Signing in twice with the same Google account must not create two users."""
    from config import settings
    from services import auth_service

    monkeypatch.setattr(settings, "google_client_id", "test-client-id")
    monkeypatch.setattr(
        auth_service.google_id_token, "verify_oauth2_token", lambda *a, **k: GOOGLE_CLAIMS
    )

    first = await client.post("/auth/google", json={"id_token": "token-1"})
    second = await client.post("/auth/google", json={"id_token": "token-2"})
    assert first.json()["user"]["user_id"] == second.json()["user"]["user_id"]


@pytest.mark.asyncio
async def test_google_login_links_to_existing_password_account(client, monkeypatch):
    """
    Someone who registered with email+password, then later clicks
    "Continue with Google" using the SAME email, should log into their
    existing account - not get a second, disconnected one.
    """
    from config import settings
    from services import auth_service

    payload = sample_user_payload("linked.user@example.com")
    register_resp = await client.post("/auth/register", json=payload)
    original_user_id = register_resp.json()["user"]["user_id"]

    monkeypatch.setattr(settings, "google_client_id", "test-client-id")
    linked_claims = {**GOOGLE_CLAIMS, "email": "linked.user@example.com"}
    monkeypatch.setattr(
        auth_service.google_id_token, "verify_oauth2_token", lambda *a, **k: linked_claims
    )

    resp = await client.post("/auth/google", json={"id_token": "token"})
    assert resp.status_code == 200
    assert resp.json()["user"]["user_id"] == original_user_id


@pytest.mark.asyncio
async def test_google_login_rejects_unverified_email(client, monkeypatch):
    from config import settings
    from services import auth_service

    monkeypatch.setattr(settings, "google_client_id", "test-client-id")
    unverified_claims = {**GOOGLE_CLAIMS, "email_verified": False}
    monkeypatch.setattr(
        auth_service.google_id_token, "verify_oauth2_token", lambda *a, **k: unverified_claims
    )

    resp = await client.post("/auth/google", json={"id_token": "token"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_google_login_rejects_invalid_token(client, monkeypatch):
    from config import settings
    from services import auth_service

    def _raise(*args, **kwargs):
        raise ValueError("Token used too late")

    monkeypatch.setattr(settings, "google_client_id", "test-client-id")
    monkeypatch.setattr(auth_service.google_id_token, "verify_oauth2_token", _raise)

    resp = await client.post("/auth/google", json={"id_token": "expired-token"})
    assert resp.status_code == 401
