import json

from ai.measurement.moving_contract import MeasurementResponse
from ai.measurement.standard_sizes import NONSTANDARD, QUALIFIER, main, match, snap


def response(label, dims):
    return MeasurementResponse.model_validate({"videos": [{
        "video_id": "v", "device": "cpu", "pose_source": "rgb_metric_depth_pnp",
        "intrinsics_source": "provided", "scale": {"status": "model_metric_scale"},
        "objects": [{
            "track_id": "t", "label": label, "status": "estimated",
            "dimensions": {k: ({"cm": v, "method": "registered_multiview",
                                "scale_source": "pretrained_depth", "views": 4} if v else None)
                           for k, v in dims.items()},
            "frames": [1, 2, 3], "qualifiers": [],
        }],
        "poses": [], "depth_timing": {"frames": 3, "total_s": 1.0, "median_s": 0.3},
    }]})


def dims(result):
    track = result.videos[0].objects[0]
    return {k: (v.cm if v else None) for k, v in track.dimensions.items()}, track.qualifiers


def test_bed_snaps_to_nearest_korean_standard():
    size, qualifiers = dims(snap(response("bed", {"width": 231, "depth": 141, "height": 52})))
    assert size == {"width": 208, "depth": 140, "height": 52}
    assert QUALIFIER in qualifiers and f"{QUALIFIER}:double" in qualifiers


def test_label_variants_and_us_region():
    size, qualifiers = dims(snap(response("Bed frame", {"width": 215, "depth": 150,
                                                        "height": 40}), region="us"))
    assert size["depth"] == 152 and size["width"] == 211
    assert f"{QUALIFIER}:queen" in qualifiers


def test_other_furniture_unchanged():
    original = response("sofa", {"width": 210, "depth": 150, "height": 80})
    assert snap(original) == original


def test_far_from_every_standard_keeps_measurement():
    size, qualifiers = dims(snap(response("bed", {"width": 420, "depth": 150, "height": 50})))
    assert size["width"] == 420
    assert NONSTANDARD in qualifiers and QUALIFIER not in qualifiers


def test_length_only_bed():
    size, _ = dims(snap(response("bed", {"width": 190, "depth": None, "height": 45})))
    assert size == {"width": 208, "depth": None, "height": 45}
    assert match({"depth": 112})[0] == "super_single"


def test_cli(tmp_path):
    source = tmp_path / "in.json"
    source.write_text(response("bed", {"width": 200, "depth": 165, "height": 50})
                      .model_dump_json())
    out = tmp_path / "out.json"
    assert main(["apply", str(source), "--output", str(out)]) == 0
    track = json.loads(out.read_text())["videos"][0]["objects"][0]
    assert track["dimensions"]["depth"]["cm"] == 160
