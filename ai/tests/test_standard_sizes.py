import json

from ai.measurement.moving_contract import MeasurementResponse
from ai.measurement.standard_sizes import (
    APPLIANCE_QUALIFIER,
    FURNITURE_MM,
    FURNITURE_QUALIFIER,
    NONSTANDARD,
    QUALIFIER,
    catalog_labels,
    main,
    match,
    snap,
)


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


def test_washer_is_pulled_toward_catalog_sizes():
    # A compact drum washer (about 60 x 55 x 85 cm) measured 25% too wide and deep.
    size, qualifiers = dims(snap(response("washing machine", {"width": 75, "depth": 69,
                                                              "height": 88})))
    assert "appliance_catalog_prior" in qualifiers
    assert 59 <= size["width"] <= 70 and 55 <= size["depth"] <= 66
    assert 82 <= size["height"] <= 100


def test_fridge_height_stays_near_measurement_between_models():
    size, _ = dims(snap(response("refrigerator", {"width": 64, "depth": 57, "height": 158})))
    assert 150 <= size["height"] <= 170


def test_catalog_uses_measurement_axes():
    from ai.measurement.standard_sizes import catalog_cm
    for label in ("washer", "refrigerator"):
        assert all(s["width"] >= s["depth"] for s in catalog_cm(label))


def test_pull_excludes_a_model():
    from ai.measurement.standard_sizes import pull_to_catalog
    exact = {"width": 63.7, "depth": 55.0, "height": 82.0}
    assert pull_to_catalog(exact, "washer", exclude="lg_fr9wpb") != pull_to_catalog(exact,
                                                                                   "washer")


def test_furniture_pulled_toward_korean_catalog():
    size, qualifiers = dims(snap(response("bookshelf", {"width": 87, "depth": 34,
                                                        "height": 124})))
    assert 27 < size["depth"] < 32 and size["width"] < 87
    assert FURNITURE_QUALIFIER in qualifiers and APPLIANCE_QUALIFIER not in qualifiers


def test_furniture_catalog_is_korea_only():
    original = response("wardrobe", {"width": 85, "depth": 70, "height": 190})
    assert snap(original, region="us") == original
    assert "desk" in catalog_labels("kr") and "desk" not in catalog_labels("us")
    assert "washer" in catalog_labels("us")


def test_every_furniture_model_is_plausible():
    for label, models in FURNITURE_MM.items():
        assert len(models) >= 15, label
        for w, d, h in models.values():
            assert 300 <= max(w, d) <= 2100 and 250 <= min(w, d) <= 800 and 400 <= h <= 2200
