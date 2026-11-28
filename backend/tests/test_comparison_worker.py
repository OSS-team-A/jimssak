import copy

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.tests.test_snapshots import jpeg_size

MOVES = "/api/v1/moves"
WORKER = "/api/v1/internal"
KEY = {"X-Worker-Key": "test-worker-key"}


@pytest.fixture
def client(tmp_path):
    settings = Settings(media_dir=tmp_path / "media", analysis_worker_key="test-worker-key")
    with TestClient(create_app(settings)) as client:
        yield client


def analysis_result(job):
    return {
        "schema_version": "2.0", "analysis_id": job["analysis_id"], "status": "completed",
        "progress": 1.0, "video": {"video_id": job["video_id"]}, "warnings": [], "error": None,
        "items": [{
            "item_id": item_id, "source": "ai", "label": label, "label_ko": label_ko,
            "quantity": 1, "confidence": 0.9, "dimensions_cm": {"width": 100},
            "dimensions_status": "partial", "needs_review": False,
            "snapshots": [{"time_ms": 400}],
        } for item_id, label, label_ko in (("desk-1", "desk", "책상"), ("tv-1", "tv", "TV"))],
    }


def comparison_job(client, videos):
    """Analyze and confirm a list through the worker, then upload the after-move video."""
    body = client.post(MOVES).json()
    url, headers = f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}
    room = ("room.mp4", videos["room"], "video/mp4")
    client.post(f"{url}/media", headers=headers, files={"file": room})
    job = client.post(f"{WORKER}/analyses/claim", headers=KEY).json()
    assert job["kind"] == "analysis"
    sent = client.post(f"{WORKER}/analyses/{job['analysis_id']}/result", headers=KEY,
                       json=analysis_result(job))
    assert sent.status_code == 200, sent.text
    assert client.post(f"{url}/inventory/confirm", headers=headers).status_code == 204
    client.post(f"{url}/after-media", headers=headers,
                files={"file": ("after.mp4", videos["upright"], "video/mp4")})
    # The first-analysis queue never hands out comparisons: an analysis-only worker keeps
    # working.
    assert client.post(f"{WORKER}/analyses/claim", headers=KEY).status_code == 204
    task = client.post(f"{WORKER}/comparisons/claim", headers=KEY).json()
    assert task["kind"] == "comparison"
    return url, headers, task


def result(task, **changes):
    body = {
        "schema_version": "1.0", "comparison_id": task["comparison_id"], "status": "completed",
        "progress": 1.0, "video": {"video_id": task["after_video_id"]}, "warnings": [],
        "error": None, "items": [
            {"item_id": "desk-1", "result": "changed", "reasons": ["다리에 흠집"],
             "before_snapshots": [{"time_ms": 400}],
             "after_snapshots": [{"time_ms": 900, "bbox": [0, 0, 120, 160]}]},
            {"item_id": "tv-1", "result": "missing", "reasons": ["찾지 못함"],
             "before_snapshots": [{"time_ms": 1200}], "after_snapshots": []},
        ],
    }
    body.update(changes)
    return body


def test_analyzer_compares_both_videos_and_photos_are_captured(client, videos):
    url, headers, task = comparison_job(client, videos)
    assert [item["item_id"] for item in task["items"]] == ["desk-1", "tv-1"]
    assert client.get(url + "/comparison", headers=headers).json()["status"] == "running"

    before = client.get(task["before_video_url"], headers=KEY)
    after = client.get(task["after_video_url"], headers=KEY)
    assert (before.content, after.content) == (videos["room"], videos["upright"])
    progress = client.post(f"{WORKER}/comparisons/{task['comparison_id']}/progress",
                           headers=KEY, json={"progress": 0.5})
    assert progress.json()["progress"] == 0.5

    sent = client.post(f"{WORKER}/comparisons/{task['comparison_id']}/result", headers=KEY,
                       json=result(task))
    assert sent.status_code == 200, sent.text
    done = client.get(url + "/comparison", headers=headers).json()
    assert done["status"] == "completed" and done["demo"] is False
    desk, tv = done["items"]
    assert (desk["label_ko"], desk["result"], tv["result"]) == ("책상", "changed", "missing")
    # Before photos come from the first video, after photos from the upright after video.
    before_photo = client.get(desk["before_photos"][0]["image_url"], headers=headers)
    after_photo = client.get(desk["after_photos"][0]["image_url"], headers=headers)
    assert jpeg_size(before_photo.content) == (320, 240)
    assert jpeg_size(after_photo.content) == (120, 160)
    assert desk["after_photos"][0]["image_url"].split("/")[-1].startswith("cmp_")

    late = client.post(f"{WORKER}/comparisons/{task['comparison_id']}/result", headers=KEY,
                       json=result(task))
    assert late.status_code == 409


@pytest.mark.parametrize("change", [
    {"comparison_id": "cmp_0000000000000000"},
    {"video": {"video_id": "someone-else"}},
    {"schema_version": "2.0"},
    {"status": "failed"},                                   # failed without error
])
def test_bad_comparison_results_are_rejected(client, videos, change):
    url, headers, task = comparison_job(client, videos)
    body = copy.deepcopy(result(task))
    body.update(change)
    response = client.post(f"{WORKER}/comparisons/{task['comparison_id']}/result",
                           headers=KEY, json=body)
    assert response.status_code == 422
    assert client.get(url + "/comparison", headers=headers).json()["status"] == "running"


def test_every_listed_item_must_be_reported_once(client, videos):
    _, _, task = comparison_job(client, videos)
    full = result(task)["items"]
    for items in ([full[0]], [*full, full[0]], [*full, {**full[0], "item_id": "ghost"}]):
        response = client.post(f"{WORKER}/comparisons/{task['comparison_id']}/result",
                               headers=KEY, json=result(task, items=items))
        assert response.status_code == 422


def test_impossible_photo_rejects_the_comparison(client, videos):
    url, headers, task = comparison_job(client, videos)
    items = result(task)["items"]
    items[1]["after_snapshots"] = [{"time_ms": 9000}]
    response = client.post(f"{WORKER}/comparisons/{task['comparison_id']}/result",
                           headers=KEY, json=result(task, items=items))
    assert response.status_code == 422 and "tv-1" in response.json()["detail"]
    root = client.app.state.media_storage.root
    assert not list(root.rglob("cmp_*/*.jpg"))


def test_comparison_failure_and_replaced_after_video(client, videos):
    url, headers, task = comparison_job(client, videos)
    failure = result(task, status="failed", items=[], error={
        "code": "video_too_dark", "message": "영상이 너무 어두워요", "retryable": True})
    assert client.post(f"{WORKER}/comparisons/{task['comparison_id']}/result", headers=KEY,
                       json=failure).status_code == 200
    failed = client.get(url + "/comparison", headers=headers).json()
    assert failed["status"] == "failed" and failed["error"]["code"] == "video_too_dark"

    # A new after-move video: the old job's videos are gone and its late result is refused.
    client.post(f"{url}/after-media", headers=headers,
                files={"file": ("again.mp4", videos["room"], "video/mp4")})
    old = f"{WORKER}/comparisons/{task['comparison_id']}"
    assert client.get(f"{old}/after-video", headers=KEY).status_code == 404
    assert client.post(f"{old}/result", headers=KEY, json=result(task)).status_code == 409


def test_comparison_routes_need_the_worker_key(client, videos):
    _, _, task = comparison_job(client, videos)
    base = f"{WORKER}/comparisons/{task['comparison_id']}"
    assert client.get(f"{base}/before-video").status_code == 401
    assert client.post(f"{base}/result", json=result(task)).status_code == 401
    assert client.post(f"{WORKER}/comparisons/claim").status_code == 401
