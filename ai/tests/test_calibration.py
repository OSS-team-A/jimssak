import json
import math

import pytest

from ai.measurement.calibration import (
    QUALIFIER,
    apply,
    cross_validate,
    factor_for,
    fit,
    main,
    room_of,
    samples,
)
from ai.measurement.moving_contract import MeasurementResponse


def clip(video_id, dims, pose_source="rgb_metric_depth_pnp", method="registered_multiview",
         scale="model_metric_scale"):
    return {
        "video_id": video_id, "device": "cpu", "pose_source": pose_source,
        "intrinsics_source": "provided", "scale": {"status": scale},
        "objects": [{
            "track_id": "t", "label": "table", "status": "estimated",
            "dimensions": {k: ({"cm": v, "method": method, "scale_source": "pretrained_depth",
                                "views": 4} if v else None) for k, v in dims.items()},
            "frames": [1, 2, 3], "qualifiers": [],
        }],
        "poses": [], "depth_timing": {"frames": 3, "total_s": 1.0, "median_s": 0.3},
    }


def response(*clips):
    return MeasurementResponse.model_validate({"videos": list(clips)})


def truth(*rows):
    return {"tracks": [{"video_id": v, "track_id": "t", "dimensions_cm": d} for v, d in rows]}


def test_fit_learns_median_ratio_with_fallbacks():
    rows = [{"video_id": f"4000000{i % 10}-rgb", "kind": "rgb", "axis": "width",
             "method": "registered_multiview", "estimate": 150.0, "actual": 100.0}
            for i in range(12)]
    rows += [{"video_id": "40000000-rgb", "kind": "rgb", "axis": "height",
              "method": "registered_multiview", "estimate": 120.0, "actual": 100.0}]
    model = fit(rows, min_samples=10)
    assert math.isclose(factor_for(model, "rgb", "width", "registered_multiview"), 1 / 1.5,
                        rel_tol=1e-3)
    # height has too few samples on its own: falls back to the whole rgb group
    assert factor_for(model, "rgb", "height", "registered_multiview") == model["factors"][
        "rgb"]["factor"]
    assert factor_for(model, "arkit", "width", "registered_multiview") is None


def test_apply_scales_and_marks_tracks_but_skips_reference_scaled_videos():
    calibration = {"factors": {"rgb/width": {"factor": 0.5, "samples": 20},
                               "arkit": {"factor": 1.0, "samples": 20}}}
    result = apply(response(
        clip("a-rgb", {"width": 100.0, "depth": 40.0, "height": None}),
        clip("b-rgb", {"width": 100.0, "depth": 40.0, "height": 70.0},
             scale="reference_scaled"),
    ), calibration)
    track = result.videos[0].objects[0]
    assert track.dimensions["width"].cm == 50.0
    assert track.dimensions["depth"].cm == 40.0  # no factor for rgb/depth
    assert track.dimensions["height"] is None
    assert QUALIFIER in track.qualifiers
    assert result.videos[1].objects[0].dimensions["width"].cm == 100.0


def test_samples_skip_reference_scaled_and_unmatched_tracks():
    rows = samples(response(
        clip("40000001-rgb", {"width": 150.0, "depth": 60.0, "height": None}),
        clip("40000002", {"width": 90.0, "depth": 50.0, "height": 80.0},
             pose_source="capture_poses"),
        clip("40000003-rgb", {"width": 90.0, "depth": 50.0, "height": 80.0},
             scale="reference_scaled"),
    ), truth(("40000001-rgb", {"width": 100, "depth": 40, "height": 70}),
             ("40000002", {"width": 100, "depth": 40, "height": 70})))
    assert [(r["kind"], r["axis"]) for r in rows] == [
        ("rgb", "width"), ("rgb", "depth"),
        ("arkit", "width"), ("arkit", "depth"), ("arkit", "height")]


def test_cross_validation_holds_out_whole_rooms():
    rooms = {f"4000000{i}": f"room{i // 2}" for i in range(10)}
    assert room_of("40000003-abc123-rgb", rooms) == "room1"
    rows = [{"video_id": f"4000000{i}-x-rgb", "kind": "rgb", "axis": axis,
             "method": "registered_multiview", "estimate": 140.0 + i, "actual": 100.0}
            for i in range(10) for axis in ("width", "depth", "height")]
    report = cross_validate(rows, rooms, min_samples=5)
    rgb = report["kinds"]["rgb"]
    assert report["rooms"] == 5 and rgb["dimensions"] == 30
    assert rgb["calibrated"] < rgb["raw"] / 5


def test_cli_fit_then_apply(tmp_path):
    clips, tracks = [], []
    for i in range(12):
        video = f"4{i:07d}-x-rgb"
        clips.append(clip(video, {"width": 150.0, "depth": 60.0, "height": 84.0}))
        tracks.append((video, {"width": 100, "depth": 40, "height": 70}))
    (tmp_path / "result.json").write_text(response(*clips).model_dump_json())
    (tmp_path / "truth.json").write_text(json.dumps(truth(*tracks)))
    assert main(["fit", str(tmp_path / "truth.json"), str(tmp_path / "result.json"),
                 "--output", str(tmp_path / "calibration.json")]) == 0
    model = json.loads((tmp_path / "calibration.json").read_text())
    assert model["trained_on"]["videos"] == 12
    assert main(["apply", str(tmp_path / "result.json"), "--calibration",
                 str(tmp_path / "calibration.json"), "--output", str(tmp_path / "out.json")]) == 0
    out = MeasurementResponse.model_validate_json((tmp_path / "out.json").read_text())
    assert out.videos[0].objects[0].dimensions["width"].cm == pytest.approx(100.0, abs=0.01)
    assert out.videos[0].objects[0].dimensions["height"].cm == pytest.approx(70.0, abs=0.01)
