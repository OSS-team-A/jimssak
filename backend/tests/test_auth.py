import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

AUTH = "/api/v1/auth"
ACCOUNT = {"email": "Mover@Example.com", "password": "a long passphrase"}


@pytest.fixture
def settings(tmp_path):
    return Settings(
        cors_origins=["http://localhost:5173"],
        auth_db_path=tmp_path / "auth.sqlite3",
        auth_session_ttl_seconds=600,
        auth_refresh_ttl_seconds=3600,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


def test_account_and_sessions_survive_restart(settings):
    with TestClient(create_app(settings)) as client:
        signup = client.post(f"{AUTH}/signup", json=ACCOUNT)
        assert signup.status_code == 201
        user = signup.json()
        assert user == {"id": user["id"], "email": "mover@example.com"}
        login = client.post(f"{AUTH}/login", json={**ACCOUNT, "email": "mover@example.com"})
        assert login.status_code == 200
        assert login.headers["Cache-Control"] == "no-store"
        session = login.json()
        assert session["token_type"] == "bearer"
        assert session["expires_in"] == 600
        assert session["refresh_expires_in"] == 3600
        headers = {"Authorization": f"Bearer {session['access_token']}"}

    with TestClient(create_app(settings)) as client:
        assert client.get(f"{AUTH}/me", headers=headers).json() == user
        other_login = client.post(f"{AUTH}/login", json=ACCOUNT).json()
        other_headers = {"Authorization": f"Bearer {other_login['access_token']}"}
        logout = client.post(f"{AUTH}/logout", headers=headers)
        assert logout.status_code == 204
        assert logout.content == b""
        assert client.get(f"{AUTH}/me", headers=headers).status_code == 401
        assert client.post(
            f"{AUTH}/refresh", json={"refresh_token": session["refresh_token"]}
        ).status_code == 401
        assert client.get(f"{AUTH}/me", headers=other_headers).json() == user


def test_refresh_after_access_expiry_and_restart(settings):
    with TestClient(create_app(settings)) as client:
        user = client.post(f"{AUTH}/signup", json=ACCOUNT).json()
        tokens = client.post(f"{AUTH}/login", json=ACCOUNT).json()
        with closing(sqlite3.connect(settings.auth_db_path)) as db, db:
            db.execute("UPDATE sessions SET expires_at = 0")
            db.execute("UPDATE refresh_tokens SET expires_at = expires_at - 3300")
            digest, deadline = db.execute(
                "SELECT token_digest, expires_at FROM refresh_tokens"
            ).fetchone()
            assert digest == hashlib.sha256(tokens["refresh_token"].encode()).digest()
        # A separate login cleans up expired sessions.
        client.post(f"{AUTH}/login", json=ACCOUNT).raise_for_status()

    with TestClient(create_app(settings)) as client:
        old_refresh = {"refresh_token": tokens["refresh_token"]}
        response = client.post(f"{AUTH}/refresh", json=old_refresh)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        new = response.json()
        assert new["access_token"] != tokens["access_token"]
        assert new["refresh_token"] != tokens["refresh_token"]
        assert 0 < new["expires_in"] == new["refresh_expires_in"] <= 300
        assert client.post(f"{AUTH}/refresh", json=old_refresh).status_code == 401
        headers = {"Authorization": f"Bearer {new['access_token']}"}
        assert client.get(f"{AUTH}/me", headers=headers).json() == user
        with closing(sqlite3.connect(settings.auth_db_path)) as db:
            new_deadline, = db.execute(
                "SELECT expires_at FROM refresh_tokens WHERE token_digest = ?",
                (hashlib.sha256(new["refresh_token"].encode()).digest(),),
            ).fetchone()
        assert new_deadline == deadline


def test_same_refresh_token_can_only_be_used_once(client):
    client.post(f"{AUTH}/signup", json=ACCOUNT).raise_for_status()
    tokens = client.post(f"{AUTH}/login", json=ACCOUNT).json()
    payload = {"refresh_token": tokens["refresh_token"]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        requests = [pool.submit(client.post, f"{AUTH}/refresh", json=payload) for _ in range(2)]
        responses = [request.result() for request in requests]
    assert sorted(response.status_code for response in responses) == [200, 401]
    old_headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    assert client.get(f"{AUTH}/me", headers=old_headers).status_code == 401
    renewed = next(response.json() for response in responses if response.is_success)
    assert client.post(
        f"{AUTH}/refresh", json={"refresh_token": renewed["refresh_token"]}
    ).status_code == 200


def test_expired_and_wrong_token_types_rejected(client, settings):
    client.post(f"{AUTH}/signup", json=ACCOUNT).raise_for_status()
    tokens = client.post(f"{AUTH}/login", json=ACCOUNT).json()
    refresh_headers = {"Authorization": f"Bearer {tokens['refresh_token']}"}
    assert client.get(f"{AUTH}/me", headers=refresh_headers).status_code == 401
    assert client.post(
        f"{AUTH}/refresh", json={"refresh_token": tokens["access_token"]}
    ).status_code == 401
    with closing(sqlite3.connect(settings.auth_db_path)) as db, db:
        db.execute("UPDATE refresh_tokens SET expires_at = 0")
    assert client.post(
        f"{AUTH}/refresh", json={"refresh_token": tokens["refresh_token"]}
    ).status_code == 401


def test_duplicate_and_failed_logins(client):
    assert client.post(f"{AUTH}/signup", json=ACCOUNT).status_code == 201
    duplicate = {**ACCOUNT, "email": "mover@example.COM", "password": "wrong password"}
    assert client.post(f"{AUTH}/signup", json=duplicate).status_code == 409
    wrong_password = client.post(f"{AUTH}/login", json=duplicate)
    unknown = client.post(f"{AUTH}/login", json={**ACCOUNT, "email": "unknown@example.com"})
    assert wrong_password.status_code == unknown.status_code == 401
    assert wrong_password.json() == unknown.json()
    assert client.post(f"{AUTH}/login", json=ACCOUNT).status_code == 200


@pytest.mark.parametrize("invalid", [
    {"email": "not-an-email"}, {"password": "tiny!42"}, {"password": "a" * 129},
])
def test_invalid_signup(client, invalid):
    payload = {**ACCOUNT, **invalid}
    response = client.post(f"{AUTH}/signup", json=payload)
    assert response.status_code == 422
    assert payload["password"] not in response.text


@pytest.mark.parametrize("endpoint", ["signup", "login"])
def test_missing_email_error_does_not_echo_credentials(client, endpoint):
    response = client.post(f"{AUTH}/{endpoint}", json={"password": ACCOUNT["password"]})
    assert response.status_code == 422
    assert ACCOUNT["password"] not in response.text


def test_non_auth_validation_keeps_default_error_details(client):
    @client.app.get("/items/{item_id}")
    def item(item_id: int):
        return item_id

    response = client.get("/items/invalid")
    assert response.status_code == 422
    assert response.json()["detail"][0]["input"] == "invalid"


def test_expired_session_and_credential_storage(client, settings):
    client.post(f"{AUTH}/signup", json=ACCOUNT).raise_for_status()
    token = client.post(f"{AUTH}/login", json=ACCOUNT).json()["access_token"]
    with closing(sqlite3.connect(settings.auth_db_path)) as db, db:
        password_hash, = db.execute("SELECT password_hash FROM users").fetchone()
        assert password_hash.startswith("$argon2id$")
        digest, = db.execute("SELECT token_digest FROM sessions").fetchone()
        assert digest == hashlib.sha256(token.encode()).digest()
        db.execute("UPDATE sessions SET expires_at = 0")
    for headers in ({}, {"Authorization": "Bearer invalid"}, {"Authorization": f"Bearer {token}"}):
        response = client.get(f"{AUTH}/me", headers=headers)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"


def test_bearer_header_allowed_from_frontend(client):
    response = client.options(f"{AUTH}/me", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })
    assert response.status_code == 200
    assert "authorization" in response.headers["Access-Control-Allow-Headers"].lower()
