import pytest

from ai.measurement.moving_contract import MeasurementResponse
from ai.measurement.refine import LABEL_KO, canonical_label, derive_size_ranges, fuse, review_track


def estimate(cm, views=3, method="registered_multiview"):
    return {"cm": cm, "method": method, "scale_source": "capture_poses", "views": views}


def response(family, dims, pose_source="capture_poses", label="table", qualifiers=()):
    return MeasurementResponse.model_validate({"videos": [{
        "video_id": "v1", "device": "cpu", "depth_model_family": family,
        "pose_source": pose_source, "intrinsics_source": "provided",
        "scale": {"status": "capture_metric_scale"},
        "objects": [{
            "track_id": "t1", "label": label, "status": "estimated",
            "dimensions": {k: (estimate(v) if v else None) for k, v in dims.items()},
            "frames": [1, 2, 3], "qualifiers": list(qualifiers),
        }],
        "poses": [], "depth_timing": {"frames": 3, "total_s": 1.0, "median_s": 0.3},
        "source_sha256": "abc",
    }]})


def test_fuse_takes_width_from_depth_pro_with_capture_poses():
    fused = fuse({
        "depth-anything-v2": response("depth-anything-v2",
                                      {"width": 84, "depth": 47, "height": 39}),
        "depth-pro": response("depth-pro", {"width": 78, "depth": 45, "height": 55}),
    })
    track = fused.videos[0].objects[0]
    assert track.dimensions["width"].cm == 78
    assert track.dimensions["width"].depth_model_family == "depth-pro"
    assert track.dimensions["depth"].cm == 47
    assert track.dimensions["height"].depth_model_family == "depth-anything-v2"
    assert "depth_models_disagree_height" in track.qualifiers
    assert "depth_models_disagree_width" not in track.qualifiers


def test_fuse_falls_back_and_keeps_rgb_videos_on_depth_pro():
    fused = fuse({
        "depth-anything-v2": response("depth-anything-v2", {"width": 84, "depth": 47,
                                                            "height": 39}),
        "depth-pro": response("depth-pro", {"width": None, "depth": 45, "height": 41}),
    })
    width = fused.videos[0].objects[0].dimensions["width"]
    assert (width.cm, width.depth_model_family) == (84, "depth-anything-v2")
    rgb = fuse({
        "depth-anything-v2": response("depth-anything-v2", {"width": 84, "depth": 47,
                                                            "height": 39}, "rgb_metric_depth_pnp"),
        "depth-pro": response("depth-pro", {"width": 80, "depth": 45, "height": 41},
                              "rgb_metric_depth_pnp"),
    }).videos[0].objects[0]
    assert {a: rgb.dimensions[a].depth_model_family for a in ("width", "depth", "height")} == {
        "width": "depth-pro", "depth": "depth-pro", "height": "depth-pro"}


def test_fuse_rejects_different_videos():
    other = response("depth-pro", {"width": 1, "depth": 1, "height": 1})
    other.videos[0].source_sha256 = "different"
    with pytest.raises(ValueError):
        fuse({"depth-anything-v2": response("a", {"width": 1, "depth": 1, "height": 1}),
              "depth-pro": other})


def test_review_flags_missing_out_of_range_and_few_views():
    track = response("x", {"width": 141, "depth": None, "height": 310},
                     label="wardrobe").videos[0].objects[0]
    track.dimensions["width"].views = 1
    result = review_track(track)
    assert result["needs_review"]
    assert result["review_codes"] == [
        "missing_depth", "few_views", "out_of_range_height", "possible_built_in",
    ]
    assert "옆면" in result["review_reasons"][0]
    assert "옷장 높이 310cm" in result["review_reasons"][2]


def test_review_flags_ceiling_height_storage_as_possible_built_in():
    tall = response("x", {"width": 180, "depth": 60, "height": 232},
                    label="cabinet").videos[0].objects[0]
    result = review_track(tall)
    assert result["review_codes"] == ["possible_built_in"]
    assert "붙박이 수납장" in result["review_reasons"][0]
    low = response("x", {"width": 180, "depth": 60, "height": 200},
                   label="wardrobe").videos[0].objects[0]
    assert review_track(low)["review_codes"] == []
    fridge = response("x", {"width": 90, "depth": 75, "height": 230},
                      label="refrigerator").videos[0].objects[0]
    assert "possible_built_in" not in review_track(fridge)["review_codes"]


def test_review_passes_plausible_multiview_estimate():
    track = response("x", {"width": 74, "depth": 49, "height": 43},
                     label="folding-table").videos[0].objects[0]
    assert review_track(track) == {
        "track_id": "t1", "label": "folding-table", "needs_review": False,
        "review_codes": [], "review_reasons": [],
    }


def test_canonical_label_handles_dataset_and_compound_names():
    assert canonical_label("dining table") == "table"
    assert canonical_label("bedside-cabinet") == "cabinet"
    assert canonical_label("wood-nightstand") == "cabinet"
    assert canonical_label("bookshelf") == "shelf"
    assert canonical_label("tv_monitor") == "tv_monitor"
    assert canonical_label("furniture") == "furniture"


def test_canonical_label_maps_ai1_category_codes():
    assert canonical_label("dishwasher") == "dishwasher"
    assert canonical_label("washing_machine") == "washer"
    assert canonical_label("television") == "tv_monitor"
    assert canonical_label("monitor") == "tv_monitor"
    assert canonical_label("drawer_chest") == "drawer"
    assert canonical_label("bookshelf") == "shelf"
    assert canonical_label("air_conditioner") == "air_conditioner"
    ai1_codes = (
        "sofa bed mattress desk table chair stool ottoman cabinet drawer_chest wardrobe "
        "bookshelf television monitor refrigerator washing_machine dishwasher microwave oven "
        "air_conditioner mirror lamp fan suitcase box bicycle piano"
    ).split()
    assert all(canonical_label(code) in LABEL_KO for code in ai1_codes)


def test_derive_size_ranges_uses_percentiles_and_skips_small_classes():
    rows = [{"label": "table", "width_cm": str(50 + i), "depth_cm": "40", "height_cm": "70"}
            for i in range(21)]
    rows.append({"label": "tv_monitor", "width_cm": "100", "depth_cm": "10", "height_cm": "60"})
    ranges = derive_size_ranges(rows)
    assert list(ranges) == ["table"]
    assert ranges["table"]["width"] == (40, 84)
    assert ranges["table"]["height"] == (56, 84)
