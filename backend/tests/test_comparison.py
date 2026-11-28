from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

MOVES = "/api/v1/moves"
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1000


class Clock:
    """One clock for both stores, so demo jobs advance together."""

    def __init__(self, client):
        self.now = 1_800_000_000.0
        client.app.state.analysis_store.clock = lambda: self.now
        client.app.state.comparison_store.clock = lambda: self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def client(tmp_path):
    settings = Settings(media_dir=tmp_path / "media", analysis_demo=True,
                        analysis_demo_seconds=6)
    with TestClient(create_app(settings)) as client:
        yield client


def upload(client, url, headers, path="media", name="room.mp4"):
    response = client.post(f"{url}/{path}", headers=headers,
                           files={"file": (name, MP4, "video/mp4")})
    assert response.status_code == 201, response.text
    return response.json()


def confirmed_move(client, clock):
    """A guest move whose demo list (desk, chair) plus a box is confirmed."""
    body = client.post(MOVES).json()
    url, headers = f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}
    upload(client, url, headers)
    clock.advance(6)
    client.post(f"{url}/inventory/items", headers=headers, json={"label_ko": "상자"})
    assert client.post(f"{url}/inventory/confirm", headers=headers).status_code == 204
    return url, headers


def comparison(client, url, headers):
    response = client.get(f"{url}/comparison", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_demo_comparison_flags_a_changed_and_a_missing_item(client):
    clock = Clock(client)
    url, headers = confirmed_move(client, clock)
    assert comparison(client, url, headers)["status"] == "not_started"

    upload(client, url, headers, "after-media")
    assert comparison(client, url, headers)["status"] == "queued"
    clock.advance(3)
    running = comparison(client, url, headers)
    assert running["status"] == "running" and running["items"] == []
    clock.advance(3)
    done = comparison(client, url, headers)
    assert done["status"] == "completed" and done["demo"] is True
    assert done["comparison_id"].startswith("cmp_")

    desk, chair, box = done["items"]
    assert (desk["label_ko"], desk["result"], desk["needs_check"]) == ("책상", "changed", True)
    assert desk["reasons"] and desk["before_photos"] and desk["after_photos"]
    assert (chair["result"], chair["after_photos"]) == ("missing", [])
    assert (box["label_ko"], box["result"], box["needs_check"]) == ("상자", "ok", False)
    assert done["summary"] == {"ok": 1, "changed": 1, "missing": 1, "unchecked": 2}
    assert desk["user_check"] is None and done["reviewed_at"] is None

    photo = client.get(desk["after_photos"][0]["image_url"], headers=headers)
    assert photo.status_code == 200 and photo.headers["content-type"] == "image/jpeg"


def test_after_video_needs_a_confirmed_list(client):
    clock = Clock(client)
    body = client.post(MOVES).json()
    url, headers = f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}
    upload(client, url, headers)
    clock.advance(6)
    response = client.post(f"{url}/after-media", headers=headers,
                           files={"file": ("after.mp4", MP4, "video/mp4")})
    assert response.status_code == 409
    assert comparison(client, url, headers)["status"] == "not_started"


def test_after_video_keeps_the_first_video_and_a_retake_starts_over(client):
    clock = Clock(client)
    url, headers = confirmed_move(client, clock)
    before = client.get(url, headers=headers).json()["data"]["room_video"]["video"]
    first_upload = upload(client, url, headers, "after-media")
    assert first_upload["replaced"] is False
    clock.advance(6)
    first = comparison(client, url, headers)

    retake = upload(client, url, headers, "after-media", name="after-again.mp4")
    assert retake["replaced"] is True
    again = comparison(client, url, headers)
    assert again["status"] == "queued" and again["comparison_id"] != first["comparison_id"]
    after = client.get(url, headers=headers).json()["data"]["room_video"]["video"]
    assert after["media_id"] == before["media_id"]


def test_uploads_at_once_leave_jobs_pointing_at_the_kept_video(client):
    clock = Clock(client)
    url, headers = confirmed_move(client, clock)
    move_id = url.rsplit("/", 1)[1]
    state = client.app.state

    def send(path, n):
        return client.post(f"{url}/{path}", headers=headers,
                           files={"file": (f"take-{n}.mp4", MP4, "video/mp4")}).status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(lambda n: send("after-media", n), range(6)))
    assert codes == [201] * 6
    kept = state.move_store.get_after_video(move_id)
    analysis_id = state.analysis_store.current(move_id).analysis_id
    task = state.comparison_store.current(move_id, analysis_id)
    assert task.after_media_id == kept.id
    assert state.media_storage.path(kept).is_file()

    # The same for the first video and its analysis.
    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(lambda n: send("media", n), range(6)))
    assert codes == [201] * 6
    assert state.analysis_store.current(move_id).media_id == state.move_store.get_video(move_id).id


def test_failed_demo_comparison(client):
    clock = Clock(client)
    url, headers = confirmed_move(client, clock)
    upload(client, url, headers, "after-media", name="fail-dark.mp4")
    clock.advance(6)
    failed = comparison(client, url, headers)
    assert failed["status"] == "failed" and failed["error"]["retryable"] is True
    assert failed["items"] == []


def test_user_answers_and_finishes_the_review(client):
    clock = Clock(client)
    url, headers = confirmed_move(client, clock)
    assert client.post(f"{url}/comparison/complete", headers=headers).status_code == 409
    upload(client, url, headers, "after-media")
    clock.advance(6)
    desk, chair, box = comparison(client, url, headers)["items"]

    def check(item, answer):
        return client.patch(f"{url}/comparison/items/{item['item_id']}", headers=headers,
                            json={"user_check": answer})

    assert check(box, "confirmed").status_code == 409          # nothing to check
    assert check({"item_id": "nope"}, "confirmed").status_code == 404
    assert check(desk, "maybe").status_code == 422
    answered = check(desk, "not_found").json()
    assert answered["items"][0]["user_check"] == "not_found"
    assert answered["summary"]["unchecked"] == 1
    # Not every item answered yet.
    assert client.post(f"{url}/comparison/complete", headers=headers).status_code == 409
    # Answers can change until the review is finished.
    assert check(desk, "confirmed").json()["items"][0]["user_check"] == "confirmed"
    check(chair, "not_found")

    done = client.post(f"{url}/comparison/complete", headers=headers)
    assert done.status_code == 200 and done.json()["reviewed_at"] is not None
    assert done.json()["summary"]["unchecked"] == 0
    again = client.post(f"{url}/comparison/complete", headers=headers)
    assert again.status_code == 200
    assert again.json()["reviewed_at"] == done.json()["reviewed_at"]
    assert check(chair, "confirmed").status_code == 409


def test_history_stage_follows_the_after_move_review(client):
    clock = Clock(client)
    client.post("/api/v1/auth/signup",
                json={"email": "mover@example.com", "password": "a long passphrase"})
    token = client.post("/api/v1/auth/login", json={
        "email": "mover@example.com", "password": "a long passphrase"}).json()["access_token"]
    member = {"Authorization": f"Bearer {token}"}

    def stage():
        return client.get(MOVES, headers=member).json()["moves"][0]["stage"]

    url = f"{MOVES}/{client.post(MOVES, headers=member).json()['move_id']}"
    assert stage() == "preparing"
    upload(client, url, member)
    clock.advance(6)
    client.get(url, headers=member)  # the demo analysis finishes when read
    assert stage() == "preparing"
    client.post(f"{url}/inventory/confirm", headers=member)
    assert stage() == "after_move"
    upload(client, url, member, "after-media")
    assert stage() == "comparing"
    clock.advance(6)
    items = comparison(client, url, member)["items"]
    for item in items:
        if item["needs_check"]:
            client.patch(f"{url}/comparison/items/{item['item_id']}", headers=member,
                         json={"user_check": "confirmed"})
    client.post(f"{url}/comparison/complete", headers=member)
    assert stage() == "done"
    # Retaking the first video starts the move over.
    upload(client, url, member)
    assert stage() == "preparing"


def test_comparison_is_private(client):
    clock = Clock(client)
    url, headers = confirmed_move(client, clock)
    other = {"X-Guest-Token": client.post(MOVES).json()["guest_token"]}
    assert client.get(f"{url}/comparison", headers=other).status_code == 404
    assert client.get(f"{url}/comparison").status_code == 401
