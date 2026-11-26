import pytest
from fastapi.testclient import TestClient

from ai.measurement.calibration import factor_for
from ai.measurement.moving_contract import MeasurementJob, VideoResult
from ai.service import result_v2
from ai.service.pipeline import NoFurniture, Recognition, analyze, load_calibration
from ai.service.worker import Backend, Worker
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.schemas.analysis import AnalysisResultV2

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1000
KEY = "test-worker-key"


def dim(cm, method="registered_multiview", views=5):
    return {"cm": cm, "method": method, "scale_source": "pretrained_depth", "views": views}


def track(track_id, label, dims, status="estimated", frames=(3, 6, 9, 12, 15), qualifiers=()):
    return {"track_id": track_id, "label": label, "status": status,
            "dimensions": {k: (dim(v) if v else None) for k, v in dims.items()},
            "frames": list(frames), "qualifiers": list(qualifiers)}


def video_result(*tracks, family="depth-anything-v2"):
    return VideoResult.model_validate({
        "video_id": "x", "device": "cpu", "depth_model_family": family,
        "pose_source": "rgb_metric_depth_pnp", "intrinsics_source": "approximate_phone_fov",
        "scale": {"status": "model_metric_scale"}, "objects": list(tracks), "poses": [],
        "depth_timing": {"frames": 5, "total_s": 1.0, "median_s": 0.2},
    })


def recognition():
    job = MeasurementJob.model_validate({"videos": [{
        "video_id": "dev", "path": "capture.mp4", "annotation_size_px": [640, 480],
        "views": [{"frame_index": 3, "track_id": "t1", "label": "desk",
                   "bbox": {"x1": 10, "y1": 10, "x2": 200, "y2": 200}}],
    }]})
    return Recognition(video=job.videos[0], confidences={"t1": 0.91})


def test_completed_result_matches_backend_schema():
    result = video_result(
        track("t1", "desk", {"width": 120.34, "depth": 60.1, "height": 73.5}),
        track("t2", "closet", {"width": 141.0, "depth": None, "height": 185.4},
              status="partial"),
    )
    body = result_v2.completed(result, analysis_id="an_1", video_id="m1", fps=30,
                               duration_s=42.5, confidences={"t1": 0.91})
    AnalysisResultV2.model_validate(body)
    desk, wardrobe = body["items"]
    assert desk["label_ko"] == "책상" and desk["dimensions_cm"]["width"] == 120.3
    assert desk["confidence"] == 0.91 and not desk["needs_review"]
    assert [s["time_ms"] for s in desk["snapshots"]] == [100, 300, 500]
    assert wardrobe["label"] == "wardrobe" and wardrobe["dimensions_status"] == "partial"
    assert wardrobe["needs_review"] and "깊이" in wardrobe["review_reasons"][0]
    assert wardrobe["confidence_source"] == "not_provided"
    assert body["warnings"] == ["camera_intrinsics_estimated"]


def test_failed_result_matches_backend_schema():
    body = result_v2.failed(analysis_id="an_1", video_id="m1", code="no_furniture_found",
                            message="가구를 찾지 못했어요", retryable=True)
    assert AnalysisResultV2.model_validate(body).error.retryable


def test_analyze_calibrates_and_snaps_beds(tmp_path):
    raw = video_result(
        track("t1", "desk", {"width": 185.0, "depth": 91.0, "height": 87.5}),
        track("t2", "bed", {"width": 330.0, "depth": 230.0, "height": 60.0}),
    )
    body = analyze(tmp_path / "v.mp4", recognition(), lambda clip: raw,
                   analysis_id="an_1", video_id="m1")
    factor = factor_for(load_calibration("depth-anything-v2"), "rgb", "width",
                        "registered_multiview")
    desk, bed = body["items"]
    assert desk["dimensions_cm"]["width"] == pytest.approx(185.0 * factor, abs=0.1)
    assert bed["standard_size"]["code"] in {"queen", "king", "double", "large_king"}
    assert bed["dimensions_cm"]["width"] == 208.0
    assert "size_not_calibrated" not in body["warnings"]


def test_analyze_without_calibration_warns(tmp_path):
    raw = video_result(track("t1", "desk", {"width": 120, "depth": 60, "height": 70}),
                       family="unknown-model")
    body = analyze(tmp_path / "v.mp4", recognition(), lambda clip: raw,
                   analysis_id="an_1", video_id="m1")
    assert body["items"][0]["dimensions_cm"]["width"] == 120
    assert "size_not_calibrated" in body["warnings"]


@pytest.fixture
def backend(tmp_path):
    settings = Settings(auth_db_path=tmp_path / "app.sqlite3", media_dir=tmp_path / "media",
                        analysis_worker_key=KEY, analysis_timeout_seconds=600)
    with TestClient(create_app(settings)) as client:
        yield client


def upload(client):
    move = client.post("/api/v1/moves").json()
    url, headers = f"/api/v1/moves/{move['move_id']}", {"X-Guest-Token": move["guest_token"]}
    client.post(f"{url}/media", headers=headers, files={"file": ("room.mp4", MP4, "video/mp4")})
    return url, headers


def test_worker_end_to_end_with_backend(backend, tmp_path):
    url, headers = upload(backend)
    seen = {}

    def measure(clip):
        seen["path"] = clip.path
        assert clip.path.read_bytes() == MP4
        return video_result(track("t1", "desk", {"width": 150, "depth": 75, "height": 80}))

    worker = Worker(Backend(backend, KEY), lambda path: recognition(), measure,
                    work_dir=tmp_path)
    sent = worker.run_once()
    assert sent["status"] == "completed"
    stored = backend.get(f"{url}/analysis", headers=headers).json()
    assert stored["status"] == "completed" and stored["demo"] is False
    assert stored["items"][0]["label_ko"] == "책상"
    assert stored["video"]["video_id"] == sent["video"]["video_id"]
    assert not seen["path"].exists()  # temporary download is removed
    assert worker.run_once() is None  # no job left


@pytest.mark.parametrize("recognizer, code", [
    (lambda path: (_ for _ in ()).throw(NoFurniture()), "no_furniture_found"),
    (lambda path: (_ for _ in ()).throw(RuntimeError("model crashed")), "recognition_failed"),
])
def test_worker_reports_failures(backend, tmp_path, recognizer, code):
    url, headers = upload(backend)
    worker = Worker(Backend(backend, KEY), recognizer, lambda clip: None, work_dir=tmp_path)
    assert worker.run_once()["error"]["code"] == code
    stored = backend.get(f"{url}/analysis", headers=headers).json()
    assert stored["status"] == "failed" and stored["error"]["retryable"] is True


def test_worker_reports_measurement_failure(backend, tmp_path):
    upload(backend)

    def measure(clip):
        raise ValueError("tracking lost")

    worker = Worker(Backend(backend, KEY), lambda path: recognition(), measure,
                    work_dir=tmp_path)
    assert worker.run_once()["error"]["code"] == "measurement_failed"
