"""
tests/test_google_auth.py
----------------------------
Tests for POST /auth/google (Phase 7 addition).

Google's real token verification requires a live network call to
Google's cert endpoint and a real signed token from a real Google
account, neither of which is available in an automated test run. So
these tests monkeypatch `google.oauth2.id_token.verify_oauth2_token`
(the exact function services.auth_service.google_login calls) to
simulate what a real verification would return - same technique
tests/test_self_healing.py already uses for other best-effort
external calls. The route/service logic being tested (503 when
unconfigured, 401 on bad/unverified tokens, find-or-create by email,
JWT issuance) is 100% real; only Google's own signature check is
stubbed out, since we have no way to produce a token Google's real
servers would consider valid.

Run with:
    pytest tests/test_google_auth.py -v
"""

import pytest
from google.oauth2 import id_token as google_id_token

from config import settings


@pytest.fixture
def configured_google(monkeypatch):
    """Simulate GOOGLE_CLIENT_ID being set in the environment."""
    monkeypatch.setattr(settings, "google_client_id", "test-client-id.apps.googleusercontent.com")
    yield


def _fake_verify(claims):
    def _verify(token, request, audience):
        assert audience == "test-client-id.apps.googleusercontent.com"
        return claims
    return _verify


@pytest.mark.asyncio
async def test_google_login_returns_503_when_not_configured(client):
    resp = await client.post("/auth/google", json={"id_token": "whatever"})
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_google_login_rejects_invalid_token(client, configured_google, monkeypatch):
    def _raise(token, request, audience):
        raise ValueError("bad token")

    monkeypatch.setattr(google_id_token, "verify_oauth2_token", _raise)

    resp = await client.post("/auth/google", json={"id_token": "garbage"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_google_login_rejects_unverified_email(client, configured_google, monkeypatch):
    monkeypatch.setattr(
        google_id_token,
        "verify_oauth2_token",
        _fake_verify({"email": "unverified@example.com", "email_verified": False, "sub": "123"}),
    )

    resp = await client.post("/auth/google", json={"id_token": "tok"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_google_login_creates_new_user(client, configured_google, monkeypatch):
    monkeypatch.setattr(
        google_id_token,
        "verify_oauth2_token",
        _fake_verify(
            {
                "email": "newgoogleuser@example.com",
                "email_verified": True,
                "sub": "google-sub-123",
                "name": "New Google User",
            }
        ),
    )

    resp = await client.post("/auth/google", json={"id_token": "tok"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["access_token"]
    assert data["user"]["email"] == "newgoogleuser@example.com"
    assert data["user"]["auth_provider"] == "google"
    assert data["user"]["email_verified"] is True

    # the returned token really works against /auth/me
    me = await client.get("/auth/me", headers={"Authorization": f"Bearer {data['access_token']}"})
    assert me.status_code == 200
    assert me.json()["email"] == "newgoogleuser@example.com"


@pytest.mark.asyncio
async def test_google_login_logs_into_existing_local_account_by_email(
    client, configured_google, monkeypatch
):
    # Register a normal email/password account first.
    register_resp = await client.post(
        "/auth/register",
        json={"full_name": "Existing User", "email": "existing@example.com", "password": "StrongPass123"},
    )
    assert register_resp.status_code == 201
    local_user_id = register_resp.json()["user"]["user_id"]

    monkeypatch.setattr(
        google_id_token,
        "verify_oauth2_token",
        _fake_verify(
            {"email": "existing@example.com", "email_verified": True, "sub": "google-sub-999"}
        ),
    )

    resp = await client.post("/auth/google", json={"id_token": "tok"})
    assert resp.status_code == 200
    data = resp.json()
    # Same account, not a duplicate - same user_id, still auth_provider "local"
    assert data["user"]["user_id"] == local_user_id
    assert data["user"]["auth_provider"] == "local"


@pytest.mark.asyncio
async def test_google_only_account_cannot_login_with_password(client, configured_google, monkeypatch):
    monkeypatch.setattr(
        google_id_token,
        "verify_oauth2_token",
        _fake_verify(
            {"email": "googleonly@example.com", "email_verified": True, "sub": "google-sub-1"}
        ),
    )
    await client.post("/auth/google", json={"id_token": "tok"})

    # No password was ever set for this account - password login must fail cleanly, not crash.
    resp = await client.post(
        "/auth/login", json={"email": "googleonly@example.com", "password": "anything123"}
    )
    assert resp.status_code == 401
requests==2.32.3     
