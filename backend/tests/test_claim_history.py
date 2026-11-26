from datetime import datetime, timedelta

import psycopg
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.schemas.moves import KST

AUTH = "/api/v1/auth"
MOVES = "/api/v1/moves"
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1000
MOVE_DAY = (datetime.now(KST).date() + timedelta(days=30)).isoformat()


@pytest.fixture
def settings(tmp_path):
    return Settings(media_dir=tmp_path / "media",
                    guest_move_ttl_seconds=600)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


def login(client, email):
    account = {"email": email, "password": "a long passphrase"}
    client.post(f"{AUTH}/signup", json=account)
    token = client.post(f"{AUTH}/login", json=account).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def guest_move_with_details(client):
    """A guest gets through SC-03 to SC-06 before being asked to log in."""
    body = client.post(MOVES).json()
    url, guest = f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}
    video = client.post(f"{url}/media", headers=guest,
                        files={"file": ("room.mp4", MP4, "video/mp4")}).json()
    client.patch(url, headers=guest, json={
        "origin": {"road_address": "서울 중구 퇴계로36길 2", "detail_address": "302호", "floor": 3},
        "destination": {"road_address": "서울 성북구 아리랑로4가길 21-7", "floor": 5},
        "schedule": {"move_date": MOVE_DAY, "start_time": "09:00"},
    })
    return body["move_id"], guest, video["media_id"]


def claim(client, move_id, member, guest):
    return client.post(f"{MOVES}/{move_id}/claim", headers={**member, **guest})


def test_claim_keeps_everything_and_switches_owner(client):
    move_id, guest, media_id = guest_move_with_details(client)
    member = login(client, "mover@example.com")

    response = claim(client, move_id, member, guest)
    assert response.status_code == 200
    snapshot = response.json()
    assert snapshot["owner"] == "member" and snapshot["expires_at"] is None
    data = snapshot["data"]
    assert data["room_video"]["video"]["media_id"] == media_id
    assert data["origin"]["value"]["detail_address"] == "302호"
    assert data["destination"]["value"]["floor"] == 5
    assert data["schedule"]["value"]["move_date"] == MOVE_DAY
    assert snapshot["missing"] == []

    url = f"{MOVES}/{move_id}"
    assert client.get(url, headers=member).json()["owner"] == "member"
    # The guest token no longer opens it.
    assert client.get(url, headers=guest).status_code == 404


def test_claiming_twice_is_harmless(client):
    move_id, guest, _ = guest_move_with_details(client)
    member = login(client, "mover@example.com")
    assert claim(client, move_id, member, guest).status_code == 200
    assert claim(client, move_id, member, guest).status_code == 200


def test_another_member_cannot_take_it(client):
    move_id, guest, _ = guest_move_with_details(client)
    assert claim(client, move_id, login(client, "mover@example.com"), guest).status_code == 200
    assert claim(client, move_id, login(client, "other@example.com"), guest).status_code == 404


def test_member_token_alone_is_not_enough(client):
    move_id, _, _ = guest_move_with_details(client)
    member = login(client, "mover@example.com")
    other_guest = {"X-Guest-Token": client.post(MOVES).json()["guest_token"]}
    assert claim(client, move_id, member, other_guest).status_code == 404
    assert claim(client, move_id, member, {}).status_code == 401
    assert claim(client, move_id, {}, other_guest).status_code == 401
    assert claim(client, move_id, {"Authorization": "Bearer expired"}, other_guest).status_code \
        == 401


def test_expired_guest_move_cannot_be_claimed(client, settings):
    move_id, guest, _ = guest_move_with_details(client)
    with psycopg.connect(settings.database_url, autocommit=True) as db:
        db.execute("UPDATE moves SET last_used_at = last_used_at - 601")
    assert claim(client, move_id, login(client, "mover@example.com"), guest).status_code == 404


def test_history_lists_own_moves_newest_first(client, settings):
    member = login(client, "mover@example.com")
    first = client.post(MOVES, headers=member).json()["move_id"]
    claimed, guest, _ = guest_move_with_details(client)
    claim(client, claimed, member, guest)
    with psycopg.connect(settings.database_url, autocommit=True) as db:
        db.execute("UPDATE moves SET created_at = created_at - 100 WHERE id = %s", (first,))
    client.post(MOVES, headers=login(client, "other@example.com"))  # not theirs

    body = client.get(MOVES, headers=member).json()
    assert body["total"] == 2 and (body["limit"], body["offset"]) == (20, 0)
    newest, oldest = body["moves"]
    assert newest["move_id"] == claimed and oldest["move_id"] == first
    assert newest["origin_address"] == "서울 중구 퇴계로36길 2"
    assert newest["destination_address"] == "서울 성북구 아리랑로4가길 21-7"
    assert (newest["move_date"], newest["start_time"]) == (MOVE_DAY, "09:00:00")
    assert newest["stage"] == "preparing"
    assert oldest["move_date"] is None and oldest["origin_address"] is None


def test_history_pages(client):
    member = login(client, "mover@example.com")
    for _ in range(3):
        client.post(MOVES, headers=member)
    page = client.get(MOVES, headers=member, params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 3 and len(page["moves"]) == 1
    assert client.get(MOVES, headers=member, params={"limit": 0}).status_code == 422


def test_history_needs_login(client):
    guest = {"X-Guest-Token": client.post(MOVES).json()["guest_token"]}
    assert client.get(MOVES).status_code == 401
    assert client.get(MOVES, headers=guest).status_code == 401
