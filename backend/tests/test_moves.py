import sqlite3
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

AUTH = "/api/v1/auth"
MOVES = "/api/v1/moves"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        cors_origins=["http://localhost:5173"],
        auth_db_path=tmp_path / "app.sqlite3",
        guest_move_ttl_seconds=600,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


def member_headers(client: TestClient, email: str) -> dict[str, str]:
    account = {"email": email, "password": "a long passphrase"}
    client.post(f"{AUTH}/signup", json=account).raise_for_status()
    token = client.post(f"{AUTH}/login", json=account).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_member_move_belongs_to_account(client):
    headers = member_headers(client, "mover@example.com")
    created = client.post(MOVES, headers=headers)
    assert created.status_code == 201
    body = created.json()
    assert body["owner"] == "member"
    assert body["guest_token"] is None and body["expires_at"] is None

    snapshot = client.get(f"{MOVES}/{body['move_id']}", headers=headers)
    assert snapshot.status_code == 200
    assert snapshot.json()["owner"] == "member"
    assert snapshot.json()["expires_at"] is None

    other = member_headers(client, "other@example.com")
    assert client.get(f"{MOVES}/{body['move_id']}", headers=other).status_code == 404


def test_guest_move_needs_its_token(client):
    created = client.post(MOVES)
    assert created.status_code == 201
    body = created.json()
    assert body["owner"] == "guest"
    assert body["guest_token"] and body["expires_at"]
    url = f"{MOVES}/{body['move_id']}"

    snapshot = client.get(url, headers={"X-Guest-Token": body["guest_token"]})
    assert snapshot.status_code == 200
    data = snapshot.json()
    assert data["owner"] == "guest"
    assert data["data"]["room_video"] == {"status": "empty", "items": []}
    assert data["data"]["analysis"] == {"status": "not_started"}
    assert data["missing"] == ["room_video", "origin", "destination", "schedule"]
    assert data["next_step"] == "room_video"

    assert client.get(url).status_code == 401
    other_guest = client.post(MOVES).json()["guest_token"]
    assert client.get(url, headers={"X-Guest-Token": other_guest}).status_code == 404
    # A member cannot open a guest move until it is claimed.
    member = member_headers(client, "mover@example.com")
    assert client.get(url, headers=member).status_code == 404


def test_guest_move_expires_after_inactivity(client, settings):
    body = client.post(MOVES).json()
    url = f"{MOVES}/{body['move_id']}"
    headers = {"X-Guest-Token": body["guest_token"]}

    with closing(sqlite3.connect(settings.auth_db_path)) as db, db:
        db.execute("UPDATE moves SET last_used_at = last_used_at - 590")
    # Using the move pushes the deadline back.
    assert client.get(url, headers=headers).status_code == 200
    with closing(sqlite3.connect(settings.auth_db_path)) as db, db:
        db.execute("UPDATE moves SET last_used_at = last_used_at - 590")
    assert client.get(url, headers=headers).status_code == 200

    with closing(sqlite3.connect(settings.auth_db_path)) as db, db:
        db.execute("UPDATE moves SET last_used_at = last_used_at - 600")
    assert client.get(url, headers=headers).status_code == 404


def test_invalid_bearer_is_not_treated_as_guest(client):
    response = client.post(MOVES, headers={"Authorization": "Bearer expired"})
    assert response.status_code == 401


def test_moves_survive_restart(settings):
    with TestClient(create_app(settings)) as client:
        body = client.post(MOVES).json()
    with TestClient(create_app(settings)) as client:
        response = client.get(
            f"{MOVES}/{body['move_id']}", headers={"X-Guest-Token": body["guest_token"]}
        )
        assert response.status_code == 200


def test_cors_allows_guest_header(client):
    response = client.options(MOVES, headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-Guest-Token",
    })
    assert response.status_code == 200
