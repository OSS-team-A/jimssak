from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from ai.measurement.metric_depth import MetricDepth  # noqa: E402


@pytest.mark.parametrize("family", ("depth-anything-v2", "depth-pro", "zoedepth"))
def test_model_family_returns_metric_depth_in_original_pixel_coordinates(family):
    # Stand-in inference so the regression test needs no checkpoint.
    depth = MetricDepth.__new__(MetricDepth)
    depth.family = family
    depth.torch = torch
    depth.device = "cpu"
    depth.dtype = torch.float32
    outputs = SimpleNamespace(predicted_depth=torch.full((1, 4, 5), 2.), field_of_view=None)
    input_pixels = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    depth.model = Mock(return_value=outputs)
    depth.processor = Mock()
    shape = (8, 10)
    metric_prediction = torch.full(shape, 3.)
    depth.processor.post_process_depth_estimation.return_value = [
        {"predicted_depth": metric_prediction}
    ]

    if family == "depth-anything-v2":
        prediction = depth._infer({"pixel_values": input_pixels}, shape)
        assert prediction.shape == shape
        assert np.allclose(prediction.numpy(), 2.0)
        depth.processor.post_process_depth_estimation.assert_not_called()
    elif family == "depth-pro":
        assert depth._infer({"pixel_values": input_pixels}, shape) is metric_prediction
        depth.model.assert_called_once()
        depth.processor.post_process_depth_estimation.assert_called_once_with(
            outputs, target_sizes=[shape]
        )
    else:
        assert depth._infer({"pixel_values": input_pixels}, shape) is metric_prediction
        assert depth.model.call_count == 2
        assert torch.equal(depth.model.call_args.kwargs["pixel_values"], input_pixels.flip(-1))
        depth.processor.post_process_depth_estimation.assert_called_once_with(
            outputs, source_sizes=[shape], outputs_flipped=outputs
        )
