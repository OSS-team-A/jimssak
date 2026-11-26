import copy

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

MOVES = "/api/v1/moves"
WORKER = "/api/v1/internal/analyses"
KEY = {"X-Worker-Key": "test-worker-key"}
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1000


def make_client(tmp_path, **overrides):
    values = {
        "media_dir": tmp_path / "media",
        "analysis_worker_key": "test-worker-key", "analysis_timeout_seconds": 600,
        "analysis_demo_seconds": 6, **overrides,
    }
    settings = Settings(**values)
    return TestClient(create_app(settings))


class Clock:
    def __init__(self, client):
        self.now = 1_800_000_000.0
        client.app.state.analysis_store.clock = lambda: self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as client:
        yield client


@pytest.fixture
def demo_client(tmp_path):
    with make_client(tmp_path, analysis_demo=True) as client:
        yield client


def new_move(client):
    body = client.post(MOVES).json()
    return f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}


def upload(client, url, headers, name="room.mp4"):
    response = client.post(f"{url}/media", headers=headers,
                           files={"file": (name, MP4, "video/mp4")})
    assert response.status_code == 201
    return response.json()


def analysis(client, url, headers):
    return client.get(url, headers=headers).json()["data"]["analysis"]


# --- status without an analyzer ---

def test_no_video_means_not_started(client):
    url, headers = new_move(client)
    assert analysis(client, url, headers)["status"] == "not_started"


def test_upload_queues_an_analysis_that_waits_for_the_analyzer(client):
    clock = Clock(client)
    url, headers = new_move(client)
    upload(client, url, headers)
    queued = analysis(client, url, headers)
    assert queued["status"] == "queued" and queued["analysis_id"].startswith("an_")
    assert queued["demo"] is False
    clock.advance(60)
    assert analysis(client, url, headers)["status"] == "queued"
    result = client.get(f"{url}/analysis", headers=headers).json()
    assert result["status"] == "queued" and result["items"] == []


def test_stuck_job_times_out(client):
    clock = Clock(client)
    url, headers = new_move(client)
    upload(client, url, headers)
    clock.advance(600)
    failed = analysis(client, url, headers)
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "timeout" and failed["error"]["retryable"] is True
    assert client.post(f"{WORKER}/claim", headers=KEY).status_code == 204


# --- demo analyzer ---

def test_demo_analyzer_finishes_with_sample_items(demo_client):
    clock = Clock(demo_client)
    url, headers = new_move(demo_client)
    upload(demo_client, url, headers)
    assert analysis(demo_client, url, headers)["status"] == "queued"
    clock.advance(3)
    running = analysis(demo_client, url, headers)
    assert running["status"] == "running" and 0 < running["progress"] < 1
    clock.advance(3)
    done = analysis(demo_client, url, headers)
    assert done["status"] == "completed" and done["demo"] is True and done["item_count"] == 2

    # The agreed v2.0 example, with this job's IDs: a desk and a chair.
    result = demo_client.get(f"{url}/analysis", headers=headers).json()
    assert list(result) == ["schema_version", "analysis_id", "status", "progress", "video",
                            "items", "warnings", "error", "demo"]
    assert result["schema_version"] == "2.0" and result["analysis_id"] == done["analysis_id"]
    assert result["video"]["duration_s"] == 42.5
    desk, chair = result["items"]
    assert (desk["label"], desk["label_ko"], chair["label"], chair["label_ko"]) == (
        "desk", "책상", "chair", "의자")
    assert {desk["source"], chair["source"]} == {"demo"}
    assert chair["needs_review"] and chair["dimensions_cm"]["depth"] is None
    assert [s["time_ms"] for s in desk["snapshots"]] == [4200]
    assert desk["snapshots"][0]["image_url"].endswith("/desk.jpg")
    assert chair["snapshots"][0]["image_url"].endswith("/chair.jpg")

    photo = demo_client.get(chair["snapshots"][0]["image_url"], headers=headers)
    assert photo.status_code == 200 and photo.headers["content-type"] == "image/jpeg"
    assert photo.content[:2] == b"\xff\xd8"  # a JPEG


def test_analysis_result_has_the_same_shape_before_completion(client):
    url, headers = new_move(client)
    assert client.get(f"{url}/analysis", headers=headers).json()["status"] == "not_started"
    media = upload(client, url, headers)
    queued = client.get(f"{url}/analysis", headers=headers).json()
    assert queued["schema_version"] == "2.0" and queued["status"] == "queued"
    assert queued["video"]["video_id"] == media["media_id"] and queued["items"] == []


def test_demo_inventory_edits_and_confirmation(demo_client):
    clock = Clock(demo_client)
    url, headers = new_move(demo_client)
    inventory = f"{url}/inventory"
    _, other = new_move(demo_client)
    assert demo_client.get(inventory, headers=other).status_code == 404
    upload(demo_client, url, headers)
    assert demo_client.get(inventory, headers=headers).json()["status"] == "empty"
    clock.advance(6)
    original = demo_client.get(f"{url}/analysis", headers=headers).json()["items"]
    chair = original[1]
    listed = demo_client.get(inventory, headers=headers).json()["items"][1]
    for field in ("needs_review", "review_reasons", "dimensions_status", "confidence"):
        assert listed[field] == chair[field]
    item_url = f"{inventory}/items/{chair['item_id']}"
    update = {"quantity": 2, "dimensions_cm": {"depth": 48}, "notes": "포장",
              "disassembly_required": True}
    edited = demo_client.patch(item_url, headers=headers, json=update).json()
    assert edited["needs_review"] and edited["review_reasons"] == chair["review_reasons"]
    assert edited["dimensions_cm"] == {**chair["dimensions_cm"], "depth": 48}
    assert edited["notes"] == "포장" and edited["disassembly_required"]
    assert demo_client.patch(item_url, headers=headers, json={"quantity": 0}).status_code == 422

    extra = demo_client.post(f"{inventory}/items", headers=headers, json={"label_ko": "박스"})
    assert extra.status_code == 201 and extra.json()["source"] == "user"
    added_url = f"{inventory}/items/{extra.json()['item_id']}"
    assert demo_client.delete(added_url, headers=headers).status_code == 204
    summary = demo_client.get(url, headers=headers).json()["data"]["inventory"]
    assert summary == {"status": "draft", "item_count": 2, "total_quantity": 3}
    assert demo_client.post(f"{inventory}/confirm", headers=headers).status_code == 204
    confirmed = demo_client.get(inventory, headers=headers).json()
    assert confirmed["status"] == "confirmed" and confirmed["items"][1]["quantity"] == 2
    assert demo_client.delete(item_url, headers=headers).status_code == 409
    assert demo_client.delete(f"{inventory}/confirm", headers=headers).status_code == 204
    assert demo_client.get(inventory, headers=headers).json()["status"] == "draft"
    for length, expected in ((501, 422), (500, 200)):
        response = demo_client.patch(item_url, headers=headers, json={"notes": "x" * length})
        assert response.status_code == expected
    assert demo_client.post(f"{inventory}/confirm", headers=headers).status_code == 204
    assert demo_client.get(f"{url}/analysis", headers=headers).json()["items"] == original

    upload(demo_client, url, headers, name="retake.mp4")
    assert demo_client.get(inventory, headers=headers).json()["items"] == []
    clock.advance(6)
    fresh = demo_client.get(inventory, headers=headers).json()
    assert fresh["status"] == "draft" and fresh["items"][1]["quantity"] == 1


def test_demo_photos_are_private(demo_client):
    url, headers = new_move(demo_client)
    _, other = new_move(demo_client)
    assert demo_client.get(f"{url}/analysis/photos/chair.jpg", headers=other).status_code == 404
    assert demo_client.get(f"{url}/analysis/photos/chair.jpg").status_code == 401
    assert demo_client.get(f"{url}/analysis/photos/secret.jpg", headers=headers).status_code \
        == 404


def test_demo_failure_for_fail_files(demo_client):
    clock = Clock(demo_client)
    url, headers = new_move(demo_client)
    upload(demo_client, url, headers, name="fail-dark-room.mp4")
    clock.advance(6)
    failed = analysis(demo_client, url, headers)
    assert failed["status"] == "failed" and failed["error"]["retryable"] is True


def test_retake_starts_a_new_analysis(demo_client):
    clock = Clock(demo_client)
    url, headers = new_move(demo_client)
    upload(demo_client, url, headers, name="fail-room.mp4")
    clock.advance(6)
    first = analysis(demo_client, url, headers)
    assert first["status"] == "failed"
    upload(demo_client, url, headers, name="retake.mp4")
    second = analysis(demo_client, url, headers)
    assert second["status"] == "queued" and second["analysis_id"] != first["analysis_id"]


def test_demo_jobs_are_not_handed_to_the_analyzer(demo_client):
    url, headers = new_move(demo_client)
    upload(demo_client, url, headers)
    assert demo_client.post(f"{WORKER}/claim", headers=KEY).status_code == 204


# --- the AI analyzer's internal API ---

def v2_result(job, **changes):
    result = {
        "schema_version": "2.0", "analysis_id": job["analysis_id"], "status": "completed",
        "progress": 1.0, "video": {"video_id": job["video_id"], "duration_s": 42.5},
        "items": [{
            "item_id": "item-1", "source": "ai", "label": "desk", "label_ko": "책상",
            "quantity": 1, "confidence": 0.91,
            "dimensions_cm": {"width": 120.3, "depth": 60.1, "height": 73.5},
            "dimensions_status": "estimated", "disassembly_required": None,
            "needs_review": False, "review_reasons": [], "snapshots": [],
        }],
        "warnings": [], "error": None,
    }
    result.update(changes)
    return result


def test_analyzer_takes_job_and_reports_result(client):
    url, headers = new_move(client)
    media = upload(client, url, headers)
    job = client.post(f"{WORKER}/claim", headers=KEY).json()
    assert job["video_id"] == media["media_id"] and job["content_type"] == "video/mp4"
    assert analysis(client, url, headers)["status"] == "running"

    video = client.get(job["video_url"], headers=KEY)
    assert video.status_code == 200 and video.content == MP4

    assert client.post(f"{WORKER}/{job['analysis_id']}/progress", headers=KEY,
                       json={"progress": 0.4}).json()["progress"] == 0.4
    assert analysis(client, url, headers)["progress"] == 0.4

    done = client.post(f"{WORKER}/{job['analysis_id']}/result", headers=KEY, json=v2_result(job))
    assert done.status_code == 200 and done.json()["status"] == "completed"
    result = client.get(f"{url}/analysis", headers=headers).json()
    assert result["demo"] is False and result["items"][0]["label_ko"] == "책상"
    # A finished job cannot be reported twice.
    assert client.post(f"{WORKER}/{job['analysis_id']}/result", headers=KEY,
                       json=v2_result(job)).status_code == 409


def test_analyzer_reports_failure(client):
    url, headers = new_move(client)
    upload(client, url, headers)
    job = client.post(f"{WORKER}/claim", headers=KEY).json()
    failure = v2_result(job, status="failed", items=[], error={
        "code": "video_too_dark", "message": "영상이 너무 어두워요", "retryable": True})
    assert client.post(f"{WORKER}/{job['analysis_id']}/result", headers=KEY,
                       json=failure).status_code == 200
    failed = analysis(client, url, headers)
    assert failed["status"] == "failed" and failed["error"]["code"] == "video_too_dark"


def test_late_result_for_replaced_video_is_refused(client):
    url, headers = new_move(client)
    upload(client, url, headers)
    old = client.post(f"{WORKER}/claim", headers=KEY).json()
    upload(client, url, headers, name="retake.mp4")
    late = client.post(f"{WORKER}/{old['analysis_id']}/result", headers=KEY, json=v2_result(old))
    assert late.status_code == 409
    assert analysis(client, url, headers)["status"] == "queued"


@pytest.mark.parametrize("change", [
    {"schema_version": "1.0"},
    {"analysis_id": "an_someone_else"},
    {"video": {"video_id": "room-01"}},
    {"status": "failed"},                         # failed without error
    {"items": [{"item_id": "x"}]},                # item missing fields
])
def test_bad_results_are_rejected(client, change):
    url, headers = new_move(client)
    upload(client, url, headers)
    job = client.post(f"{WORKER}/claim", headers=KEY).json()
    body = copy.deepcopy(v2_result(job))
    body.update(change)
    response = client.post(f"{WORKER}/{job['analysis_id']}/result", headers=KEY, json=body)
    assert response.status_code == 422
    assert analysis(client, url, headers)["status"] == "running"


def test_internal_api_needs_the_key(client, tmp_path):
    assert client.post(f"{WORKER}/claim").status_code == 401
    assert client.post(f"{WORKER}/claim", headers={"X-Worker-Key": "wrong"}).status_code == 401
    with make_client(tmp_path / "off", analysis_worker_key=None) as off:
        assert off.post(f"{WORKER}/claim", headers=KEY).status_code == 503
