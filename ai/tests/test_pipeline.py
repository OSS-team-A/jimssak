import pytest
from pydantic import ValidationError

from ai.contracts import AnalysisResult, Camera
from ai.measurement.estimator import estimate
from ai.pipeline import run_demo


def test_demo_repeatable():
    assert run_demo() == run_demo()


def test_pinhole_with_known_scale():
    result = run_demo()
    result.frame.camera = Camera(fx_px=1000, fy_px=1000)
    item = result.detections[0].model_copy(update={"depth_m": 2.0})
    measured = estimate(result.frame, item)
    assert measured.dimensions.width_cm == pytest.approx(102.4)
    assert measured.dimensions.height_cm == pytest.approx(100.8)
    assert measured.dimensions.depth_cm is None
    assert item.dimensions is None


def test_rejects_out_of_frame_bbox():
    payload = run_demo().model_dump()
    payload["detections"][0]["bbox"]["x2"] = 9999
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(payload)


def test_rejects_duplicate_tracks():
    payload = run_demo().model_dump()
    payload["detections"] *= 2
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(payload)


def test_missing_calibration_clears_old_dimensions():
    result = run_demo()
    result.frame.camera = Camera(fx_px=1000, fy_px=1000)
    item = result.detections[0].model_copy(update={"depth_m": 2.0})
    measured = estimate(result.frame, item)
    result.frame.camera = None
    assert estimate(result.frame, measured).dimensions is None
