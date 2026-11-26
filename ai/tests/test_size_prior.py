import math

import numpy as np
import pytest

from ai.measurement import size_prior
from ai.measurement.moving_contract import MeasurementResponse
from ai.service.pipeline import load_size_prior

CHAIRS = {"mean": [math.log(50), math.log(50), math.log(85)],
          "cov": [[0.02, 0.01, 0.0], [0.01, 0.02, 0.0], [0.0, 0.0, 0.01]]}
ERRORS = {"tau2": 0.02, "sigma2": {"width": 0.06, "depth": 0.1, "height": 0.03}}
PRIOR = {"sizes": {"chair": CHAIRS}, "errors": {"rgb": ERRORS}}


def response(label, dims, scale="model_metric_scale"):
    estimate = {"method": "registered_multiview", "scale_source": "pretrained_depth", "views": 4}
    return MeasurementResponse.model_validate({"videos": [{
        "video_id": "v1", "device": "cpu", "depth_model_family": "depth-anything-v2",
        "pose_source": "rgb_metric_depth_pnp", "intrinsics_source": "approximate_phone_fov",
        "scale": {"status": scale},
        "objects": [{
            "track_id": "t1", "label": label, "status": "estimated",
            "dimensions": {k: ({**estimate, "cm": v} if v else None) for k, v in dims.items()},
            "frames": [1, 2, 3], "qualifiers": [],
        }],
        "poses": [], "depth_timing": {"frames": 3, "total_s": 1.0, "median_s": 0.3},
    }]})


def test_posterior_moves_toward_typical_size_but_keeps_measurement_signal():
    sizes = size_prior.posterior({"width": 80, "depth": 80, "height": 85}, CHAIRS, ERRORS)
    assert 50 < sizes["width"] < 80 and 50 < sizes["depth"] < 80
    typical = size_prior.posterior({"width": 50, "depth": 50, "height": 85}, CHAIRS, ERRORS)
    assert typical == pytest.approx({"width": 50, "depth": 50, "height": 85}, rel=1e-6)


def test_shared_scale_error_is_corrected_from_the_other_dimensions():
    # All three 30% too large: a tight height prior pulls width down more than alone.
    alone = size_prior.posterior({"width": 65}, CHAIRS, ERRORS)["width"]
    together = size_prior.posterior({"width": 65, "depth": 65, "height": 110.5}, CHAIRS,
                                    ERRORS)["width"]
    assert together < alone


def test_apply_skips_unknown_labels_beds_and_reference_scaled_videos():
    chair = size_prior.apply(response("chair", {"width": 80, "depth": 80, "height": 85}), PRIOR)
    track = chair.videos[0].objects[0]
    assert track.dimensions["width"].cm < 80
    assert size_prior.QUALIFIER in track.qualifiers
    for label, scale in (("desk", "model_metric_scale"), ("chair", "reference_scaled")):
        same = size_prior.apply(response(label, {"width": 80, "depth": 80, "height": 85},
                                         scale), PRIOR)
        assert same.videos[0].objects[0].dimensions["width"].cm == 80


def test_missing_dimension_stays_missing():
    out = size_prior.apply(response("chair", {"width": 80, "depth": None, "height": 85}), PRIOR)
    assert out.videos[0].objects[0].dimensions["depth"] is None


def test_size_distributions_skip_small_and_standard_sized_labels():
    rows = [{"label": "chair", "width_cm": 50 + i, "depth_cm": 50, "height_cm": 80 + i % 3}
            for i in range(25)]
    rows += [{"label": "bed", "width_cm": 200, "depth_cm": 150, "height_cm": 50}] * 30
    rows += [{"label": "sofa", "width_cm": 200, "depth_cm": 90, "height_cm": 80}] * 5
    sizes = size_prior.size_distributions(rows)
    assert list(sizes) == ["chair"]
    assert np.exp(sizes["chair"]["mean"][0]) == pytest.approx(61.5, abs=1)


def test_error_model_separates_shared_and_per_axis_error():
    rows = []
    rng = np.random.default_rng(0)
    for t in range(400):
        shared = rng.normal(0, 0.2)
        for axis in size_prior.AXES:
            rows.append({"video_id": "v", "track_id": str(t), "kind": "rgb", "axis": axis,
                         "residual": shared + rng.normal(0, 0.1)})
    model = size_prior.error_model(rows)["rgb"]
    assert model["tau2"] == pytest.approx(0.04, abs=0.01)
    assert model["sigma2"]["width"] == pytest.approx(0.01, abs=0.01)


def test_shipped_prior_covers_common_furniture():
    prior = load_size_prior("depth-anything-v2")
    assert {"chair", "table", "cabinet", "sofa"} <= set(prior["sizes"])
    assert set(prior["errors"]) == {"arkit", "rgb"}
    assert load_size_prior("unknown-model") is None
