from uuid import UUID

import psycopg
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
        guest_move_ttl_seconds=600,
        media_dir=tmp_path / "media",
        media_max_bytes=4096,
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
    assert data["data"]["room_video"] == {"status": "empty", "video": None}
    assert data["data"]["analysis"]["status"] == "not_started"
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

    with psycopg.connect(settings.database_url, autocommit=True) as db:
        db.execute("UPDATE moves SET last_used_at = last_used_at - 590")
    # Using the move pushes the deadline back.
    assert client.get(url, headers=headers).status_code == 200
    with psycopg.connect(settings.database_url, autocommit=True) as db:
        db.execute("UPDATE moves SET last_used_at = last_used_at - 590")
    assert client.get(url, headers=headers).status_code == 200

    with psycopg.connect(settings.database_url, autocommit=True) as db:
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


# The start of a real MP4: a box size followed by the "ftyp" box type.
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1000


def guest_move(client: TestClient) -> tuple[str, dict[str, str]]:
    body = client.post(MOVES).json()
    return f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}


def stored_files(settings) -> list[str]:
    return sorted(p.name for p in settings.media_dir.rglob("*") if p.is_file())


def test_upload_video_and_see_it_in_snapshot(client, settings):
    url, headers = guest_move(client)
    upload = client.post(
        f"{url}/media", headers=headers,
        files={"file": (r"C:\fakepath\room.MP4", MP4, "video/mp4")},
    )
    assert upload.status_code == 201
    media = upload.json()
    assert media["filename"] == "room.MP4"
    assert media["content_type"] == "video/mp4"
    assert media["size_bytes"] == len(MP4)
    assert media["replaced"] is False
    assert "space" not in media

    snapshot = client.get(url, headers=headers).json()
    video = snapshot["data"]["room_video"]
    assert video["status"] == "uploaded"
    assert video["video"]["media_id"] == media["media_id"]
    assert snapshot["missing"] == ["origin", "destination", "schedule"]
    assert snapshot["next_step"] == "origin"
    assert stored_files(settings) == [f"{media['media_id']}.mp4"]


def test_reupload_replaces_the_only_video(client, settings):
    url, headers = guest_move(client)
    first = client.post(
        f"{url}/media", headers=headers, files={"file": ("room.mp4", MP4, "video/mp4")}
    ).json()
    # Re-shoot, as .mov from a browser that does not know the type.
    second = client.post(
        f"{url}/media", headers=headers,
        files={"file": ("retake.mov", MP4, "application/octet-stream")},
    )
    assert second.status_code == 201
    assert second.json()["replaced"] is True
    assert second.json()["content_type"] == "video/quicktime"

    video = client.get(url, headers=headers).json()["data"]["room_video"]["video"]
    assert video["media_id"] == second.json()["media_id"] != first["media_id"]
    assert video["filename"] == "retake.mov"
    # The replaced file is removed from disk.
    assert stored_files(settings) == [f"{second.json()['media_id']}.mov"]


def test_after_move_upload_needs_a_confirmed_list(client, settings):
    url, headers = guest_move(client)
    before = client.post(
        f"{url}/media", headers=headers, files={"file": ("before.mp4", MP4, "video/mp4")}
    ).json()
    response = client.post(
        f"{url}/after-media", headers=headers,
        files={"file": ("after.mp4", MP4, "video/mp4")},
    )
    assert response.status_code == 409
    # Refused before anything is stored.
    assert stored_files(settings) == [f"{before['media_id']}.mp4"]


def test_failed_reupload_keeps_previous_video(client, settings):
    url, headers = guest_move(client)
    first = client.post(
        f"{url}/media", headers=headers, files={"file": ("room.mp4", MP4, "video/mp4")}
    ).json()
    bad = client.post(
        f"{url}/media", headers=headers, files={"file": ("notes.txt", MP4, "text/plain")}
    )
    assert bad.status_code == 415
    video = client.get(url, headers=headers).json()["data"]["room_video"]["video"]
    assert video["media_id"] == first["media_id"]
    assert stored_files(settings) == [f"{first['media_id']}.mp4"]


def test_member_can_upload_to_own_move_only(client):
    owner = member_headers(client, "mover@example.com")
    move_id = client.post(MOVES, headers=owner).json()["move_id"]
    video = {"file": ("room.mp4", MP4, "video/mp4")}
    assert client.post(f"{MOVES}/{move_id}/media", headers=owner, files=video).status_code == 201
    other = member_headers(client, "other@example.com")
    assert client.post(f"{MOVES}/{move_id}/media", headers=other, files=video).status_code == 404


@pytest.mark.parametrize(("name", "content", "content_type"), [
    ("notes.txt", MP4, "text/plain"),
    ("room.avi", MP4, "video/x-msvideo"),
    ("room.mp4", MP4, "image/png"),
    ("room.mp4", b"not a video at all", "video/mp4"),
])
def test_unsupported_files_are_refused(client, settings, name, content, content_type):
    url, headers = guest_move(client)
    response = client.post(
        f"{url}/media", headers=headers, files={"file": (name, content, content_type)}
    )
    assert response.status_code == 415
    assert response.json()["detail"]
    assert stored_files(settings) == []


def test_oversized_video_leaves_no_file(client, settings):
    url, headers = guest_move(client)
    # Over the 4096-byte limit but under the Content-Length pre-check,
    # so the size is caught while writing.
    response = client.post(
        f"{url}/media", headers=headers,
        files={"file": ("room.mp4", MP4 + b"\x00" * 4096, "video/mp4")},
    )
    assert response.status_code == 413
    assert "MB" in response.json()["detail"]
    assert stored_files(settings) == []
    assert client.get(url, headers=headers).json()["data"]["room_video"] == {
        "status": "empty", "video": None
    }


def test_huge_request_refused_before_reading(client, settings):
    url, headers = guest_move(client)
    response = client.post(
        f"{url}/media", headers=headers,
        files={"file": ("room.mp4", MP4 + b"\x00" * 200_000, "video/mp4")},
    )
    assert response.status_code == 413
    assert stored_files(settings) == []


def test_upload_needs_access(client, settings):
    url, headers = guest_move(client)
    video = {"file": ("room.mp4", MP4, "video/mp4")}
    assert client.post(f"{url}/media", files=video).status_code == 401
    _, other = guest_move(client)
    assert client.post(f"{url}/media", headers=other, files=video).status_code == 404
    assert stored_files(settings) == []


def test_failed_database_write_removes_new_file_only(client, settings, monkeypatch):
    url, headers = guest_move(client)
    first = client.post(
        f"{url}/media", headers=headers, files={"file": ("room.mp4", MP4, "video/mp4")}
    ).json()

    def broken(media):
        raise psycopg.OperationalError("disk full")

    monkeypatch.setattr(client.app.state.move_store, "replace_video", broken)
    with pytest.raises(psycopg.OperationalError):
        client.post(f"{url}/media", headers=headers,
                    files={"file": ("retake.mp4", MP4, "video/mp4")})
    # The new file is gone and the previous video is untouched.
    assert stored_files(settings) == [f"{first['media_id']}.mp4"]


class BodyCounter:
    """ASGI wrapper that counts request body bytes the app actually reads."""

    def __init__(self, app):
        self.app = app
        self.read = 0

    async def __call__(self, scope, receive, send):
        async def counting_receive():
            message = await receive()
            if message["type"] == "http.request":
                self.read += len(message.get("body", b""))
            return message

        await self.app(scope, counting_receive, send)


@pytest.mark.parametrize("who", ["nobody", "other_guest", "bad_bearer"])
def test_refused_upload_is_not_read(settings, who):
    counter = BodyCounter(create_app(settings))
    with TestClient(counter) as client:
        url, _ = guest_move(client)
        headers = {
            "nobody": {},
            "other_guest": guest_move(client)[1],
            "bad_bearer": {"Authorization": "Bearer expired"},
        }[who]
        counter.read = 0
        response = client.post(
            f"{url}/media", headers=headers, files={"file": ("room.mp4", MP4, "video/mp4")}
        )
        assert response.status_code == (404 if who == "other_guest" else 401)
        assert counter.read == 0
    assert stored_files(settings) == []


def test_accepted_upload_is_read(settings):
    counter = BodyCounter(create_app(settings))
    with TestClient(counter) as client:
        url, headers = guest_move(client)
        counter.read = 0
        response = client.post(
            f"{url}/media", headers=headers, files={"file": ("room.mp4", MP4, "video/mp4")}
        )
        assert response.status_code == 201
        assert counter.read > len(MP4)


def test_ids_are_uuids(client):
    body = client.post(MOVES).json()
    assert str(UUID(body["move_id"])) == body["move_id"]
    headers = {"X-Guest-Token": body["guest_token"]}
    upload = client.post(
        f"{MOVES}/{body['move_id']}/media", headers=headers,
        files={"file": ("room.mp4", MP4, "video/mp4")},
    ).json()
    assert str(UUID(upload["media_id"])) == upload["media_id"]
    # Any spelling of the same UUID finds the move.
    upper = body["move_id"].upper()
    assert client.get(f"{MOVES}/{upper}", headers=headers).status_code == 200
    assert client.post(
        f"{MOVES}/{upper}/media", headers=headers,
        files={"file": ("room.mp4", MP4, "video/mp4")},
    ).status_code == 201


def test_malformed_move_id_is_rejected(settings):
    counter = BodyCounter(create_app(settings))
    with TestClient(counter) as client:
        _, headers = guest_move(client)
        assert client.get(f"{MOVES}/not-a-uuid", headers=headers).status_code == 422
        counter.read = 0
        response = client.post(
            f"{MOVES}/not-a-uuid/media", headers=headers,
            files={"file": ("room.mp4", MP4, "video/mp4")},
        )
        assert response.status_code == 422
        assert counter.read == 0
