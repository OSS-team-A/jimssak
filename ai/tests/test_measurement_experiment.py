import pytest
from pydantic import ValidationError

from ai.contracts import Camera
from ai.measurement.experiment import GroundTruth, evaluate, read_truth, run_experiment
from ai.pipeline import run_demo


def truth():
    return [GroundTruth(track_id="demo-room:demo:1", width_cm=100, height_cm=100)]


def test_known_errors_and_input_unchanged():
    result = run_demo()
    result.frame.camera = Camera(fx_px=1000, fy_px=1000)
    result.detections[0].depth_m = 2
    report = run_experiment(result, truth())
    assert report["evaluation"]["measured_objects"] == 1
    assert report["evaluation"]["mae_cm"] == pytest.approx(1.6)
    assert report["evaluation"]["mape_percent"] == pytest.approx(1.6)
    assert result.detections[0].dimensions is None


def test_no_calibration_reports_unavailable_not_zero_error():
    report = run_experiment(run_demo(), truth())
    assert report["evaluation"]["mae_cm"] is None
    assert report["evaluation"]["measured_objects"] == 0
    assert report["evaluation"]["unavailable_track_ids"] == ["demo-room:demo:1"]


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_invalid_ground_truth(width):
    with pytest.raises(ValidationError):
        GroundTruth(track_id="one", width_cm=width, height_cm=100)


def test_invalid_matching_is_not_silently_skipped():
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(run_demo(), truth() * 2)
    with pytest.raises(ValueError, match="missing"):
        evaluate(run_demo(), [GroundTruth(track_id="unknown", width_cm=1, height_cm=1)])


def test_read_real_csv(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text("track_id,width_cm,height_cm\ndemo-room:demo:1,100,100\n")
    assert read_truth(path) == truth()
