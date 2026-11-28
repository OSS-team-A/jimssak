import struct
import subprocess

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.snapshots import SnapshotCapture

MOVES = "/api/v1/moves"
WORKER = "/api/v1/internal/analyses"
KEY = {"X-Worker-Key": "test-worker-key"}


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    """A 2-second 320x240 test video, and the same video marked as filmed upright."""
    folder = tmp_path_factory.mktemp("videos")
    ffmpeg = SnapshotCapture().ffmpeg

    def run(*args):
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)

    run("-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(folder / "room.mp4"))
    run("-display_rotation:v:0", "90", "-i", str(folder / "room.mp4"), "-c", "copy",
        str(folder / "upright.mp4"))
    return {name: (folder / f"{name}.mp4").read_bytes() for name in ("room", "upright")}


@pytest.fixture
def client(tmp_path):
    settings = Settings(media_dir=tmp_path / "media", analysis_worker_key="test-worker-key")
    with TestClient(create_app(settings)) as client:
        yield client


def jpeg_size(data: bytes) -> tuple[int, int]:
    assert data[:2] == b"\xff\xd8"
    i = 2
    while i < len(data):
        marker, length = data[i + 1], struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + length
    raise AssertionError("no JPEG size")


def start(client, video):
    body = client.post(MOVES).json()
    url, headers = f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}
    client.post(f"{url}/media", headers=headers, files={"file": ("room.mp4", video, "video/mp4")})
    job = client.post(f"{WORKER}/claim", headers=KEY).json()
    return url, headers, job


def result(job, *snapshot_lists):
    items = [{
        "item_id": f"item-{n}", "source": "ai", "label": "desk", "label_ko": "책상",
        "quantity": 1, "confidence": 0.9, "dimensions_cm": {"width": 100},
        "dimensions_status": "partial", "needs_review": False, "snapshots": snapshots,
    } for n, snapshots in enumerate(snapshot_lists, start=1)]
    return {
        "schema_version": "2.0", "analysis_id": job["analysis_id"], "status": "completed",
        "progress": 1.0, "video": {"video_id": job["video_id"]}, "items": items,
        "warnings": [], "error": None,
    }


def photo_files(client):
    return sorted(p.name for p in client.app.state.media_storage.root.rglob("*.jpg"))


def test_photos_are_captured_at_the_reported_times(client, videos):
    url, headers, job = start(client, videos["room"])
    body = result(job, [{"time_ms": 300}, {"time_ms": 1200, "image_url": None}],
                  [{"time_ms": 900, "bbox": [10, 20, 110, 70]}])
    response = client.post(f"{WORKER}/{job['analysis_id']}/result", headers=KEY, json=body)
    assert response.status_code == 200, response.text

    items = client.get(f"{url}/analysis", headers=headers).json()["items"]
    urls = [shot["image_url"] for item in items for shot in item["snapshots"]]
    assert urls == [f"{url}/analysis/photos/{job['analysis_id']}_{n}.jpg" for n in (1, 2, 3)]
    full = client.get(urls[0], headers=headers)
    assert full.status_code == 200 and full.headers["content-type"] == "image/jpeg"
    assert jpeg_size(full.content) == (320, 240)
    # Only the box is kept.
    assert jpeg_size(client.get(urls[2], headers=headers).content) == (100, 50)
    # The editable list carries the same photos.
    listed = client.get(f"{url}/inventory", headers=headers).json()["items"]
    assert listed[0]["snapshots"][0]["image_url"] == urls[0]

    other = {"X-Guest-Token": client.post(MOVES).json()["guest_token"]}
    assert client.get(urls[0], headers=other).status_code in (401, 404)
    assert client.get(urls[0]).status_code == 401
    for name in (f"{job['analysis_id']}_9.jpg", "../../etc.jpg", "an_x_1.jpg"):
        assert client.get(f"{url}/analysis/photos/{name}", headers=headers).status_code == 404


def test_upright_phone_video_gives_upright_photos(client, videos):
    url, headers, job = start(client, videos["upright"])
    client.post(f"{WORKER}/{job['analysis_id']}/result", headers=KEY,
                json=result(job, [{"time_ms": 500}]))
    photo = client.get(f"{url}/analysis/photos/{job['analysis_id']}_1.jpg", headers=headers)
    assert jpeg_size(photo.content) == (240, 320)


@pytest.mark.parametrize("snapshots", [
    [{"time_ms": 5000}],                           # past the 2-second video
    [{"time_ms": 500, "bbox": [0, 0, 500, 100]}],  # wider than the frame
])
def test_impossible_photo_rejects_the_result(client, videos, snapshots):
    url, headers, job = start(client, videos["room"])
    response = client.post(f"{WORKER}/{job['analysis_id']}/result", headers=KEY,
                           json=result(job, [{"time_ms": 100}], snapshots))
    assert response.status_code == 422
    assert "item-2" in response.json()["detail"]
    assert client.get(url, headers=headers).json()["data"]["analysis"]["status"] == "running"
    assert photo_files(client) == []  # the photo already taken is removed too


@pytest.mark.parametrize("snapshots", [
    [{"time_ms": 100}] * 4,                        # more than 3 for one item
    [{"time_ms": 100, "bbox": [50, 10, 40, 20]}],  # x2 before x1
    [{"time_ms": 100, "bbox": [0, 0, 10]}],
])
def test_bad_photo_requests_are_rejected_before_capture(client, videos, snapshots):
    _, _, job = start(client, videos["room"])
    response = client.post(f"{WORKER}/{job['analysis_id']}/result", headers=KEY,
                           json=result(job, snapshots))
    assert response.status_code == 422
    assert photo_files(client) == []


def test_retake_removes_the_previous_photos(client, videos):
    url, headers, job = start(client, videos["room"])
    client.post(f"{WORKER}/{job['analysis_id']}/result", headers=KEY,
                json=result(job, [{"time_ms": 100}]))
    assert photo_files(client) == ["1.jpg"]
    client.post(f"{url}/media", headers=headers,
                files={"file": ("retake.mp4", videos["room"], "video/mp4")})
    assert photo_files(client) == []
    old = client.get(f"{url}/analysis/photos/{job['analysis_id']}_1.jpg", headers=headers)
    assert old.status_code == 404


def test_result_for_a_replaced_video_leaves_no_photos(client, videos):
    url, headers, job = start(client, videos["room"])
    client.post(f"{url}/media", headers=headers,
                files={"file": ("retake.mp4", videos["room"], "video/mp4")})
    late = client.post(f"{WORKER}/{job['analysis_id']}/result", headers=KEY,
                       json=result(job, [{"time_ms": 100}]))
    assert late.status_code == 409
    assert photo_files(client) == []
