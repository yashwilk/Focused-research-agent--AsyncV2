"""Tests for the auth module: registration, login, and token verification."""

import pytest
from httpx import ASGITransport, AsyncClient

from focused_research_agent.api.app import create_app, lifespan


@pytest.fixture
async def client():
    app = create_app()
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_register_returns_access_token(client):
    response = await client.post(
        "/api/v1/auth/register", json={"email": "new@example.com", "password": "strongpass1"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


async def test_register_duplicate_email_rejected(client):
    payload = {"email": "dupe@example.com", "password": "strongpass1"}
    first = await client.post("/api/v1/auth/register", json=payload)
    assert first.status_code == 201

    second = await client.post("/api/v1/auth/register", json=payload)
    assert second.status_code == 400


async def test_login_with_correct_credentials_succeeds(client):
    payload = {"email": "login@example.com", "password": "strongpass1"}
    await client.post("/api/v1/auth/register", json=payload)

    response = await client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 200
    assert response.json()["access_token"]


async def test_login_with_wrong_password_rejected(client):
    payload = {"email": "wrongpw@example.com", "password": "strongpass1"}
    await client.post("/api/v1/auth/register", json=payload)

    response = await client.post(
        "/api/v1/auth/login", json={"email": payload["email"], "password": "wrong"}
    )
    assert response.status_code == 401


async def test_login_with_unknown_email_rejected(client):
    response = await client.post(
        "/api/v1/auth/login", json={"email": "nobody@example.com", "password": "anything"}
    )
    assert response.status_code == 401


async def test_protected_endpoint_rejects_missing_token(client):
    response = await client.get("/api/v1/conversations")
    assert response.status_code in (401, 403)


async def test_protected_endpoint_rejects_garbage_token(client):
    response = await client.get(
        "/api/v1/conversations", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


async def test_protected_endpoint_accepts_valid_token(client):
    payload = {"email": "valid@example.com", "password": "strongpass1"}
    register_response = await client.post("/api/v1/auth/register", json=payload)
    token = register_response.json()["access_token"]

    response = await client.get(
        "/api/v1/conversations", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json() == []
