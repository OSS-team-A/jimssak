"""Load an optional local metric-depth model for reuse across videos."""

from pathlib import Path
from threading import Lock

import cv2
import numpy as np

MODEL_FAMILIES = {
    "depth_anything": "depth-anything-v2",
    "depth_pro": "depth-pro",
    "zoedepth": "zoedepth",
}


OFFICIAL_CHECKPOINTS = {
    "depth-anything-v2": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "depth-pro": "apple/DepthPro-hf",
    "zoedepth": "Intel/zoedepth-nyu",
}
DEFAULT_DEPTH_FAMILY = "depth-anything-v2"


def resolve_checkpoint(model: str | Path | None) -> str | Path:
    if model is None:
        family = DEFAULT_DEPTH_FAMILY
    elif isinstance(model, str) and model in OFFICIAL_CHECKPOINTS and not Path(model).exists():
        family = model
    else:
        return model

    root = Path(__file__).resolve().parents[2]
    local = root / "outputs" / "depth-checkpoints" / family
    files = ("config.json", "preprocessor_config.json", "model.safetensors")
    if all((local / file).is_file() for file in files):
        return local
    return OFFICIAL_CHECKPOINTS[family]


class MetricDepth:
    def __init__(
        self, model_path: str | Path | None = None, spatial_head: Path | None = None,
        sensor_head: Path | None = None, *, device: str = "auto",
    ):
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        available = {"cpu": True, "mps": torch.backends.mps.is_available(),
                     "cuda": torch.cuda.is_available()}
        self.device = next(name for name in ("mps", "cuda", "cpu") if available[name]) if (
            device == "auto"
        ) else device
        if not available.get(self.device, False):
            raise ValueError(f"Requested depth device is unavailable: {self.device}")
        self.allow_cpu_fallback = device == "auto"
        checkpoint = resolve_checkpoint(model_path)
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        self.family = MODEL_FAMILIES.get(config.model_type)
        if self.family is None or (
            self.family == "depth-anything-v2" and config.depth_estimation_type != "metric"
        ):
            raise ValueError("Use a metric Depth Anything, Depth Pro, or ZoeDepth checkpoint")
        has_tuned_head = spatial_head is not None or sensor_head is not None
        if has_tuned_head and self.family != "depth-anything-v2":
            raise ValueError("Existing tuning heads apply only to Depth Anything")
        self.dtype = (
            torch.float16 if self.family == "depth-pro" and self.device != "cpu" else torch.float32
        )
        self.processor = AutoImageProcessor.from_pretrained(
            checkpoint, local_files_only=True, use_fast=False
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            checkpoint, config=config, local_files_only=True, use_safetensors=True, dtype=self.dtype
        )
        if spatial_head is not None:
            weights = torch.load(spatial_head, map_location="cpu", weights_only=True)
            for layer in ("conv2", "conv3"):
                getattr(self.model.head, layer).load_state_dict({
                    key.split(".", 1)[1]: value
                    for key, value in weights.items() if key.startswith(layer + ".")
                })
        if sensor_head is not None:
            weights = torch.load(sensor_head, map_location="cpu", weights_only=True)
            self.model.head.conv3.load_state_dict(weights)
        self.model.to(self.device).eval()
        self.lock = Lock()

    def _infer(self, inputs, shape):
        inputs = {
            key: tensor.to(
                device=self.device, dtype=self.dtype if tensor.is_floating_point() else None
            )
            for key, tensor in inputs.items()
        }
        outputs = self.model(**inputs)
        if self.family == "zoedepth":
            flipped = self.model(pixel_values=inputs["pixel_values"].flip(-1))
            return self.processor.post_process_depth_estimation(
                outputs, source_sizes=[shape], outputs_flipped=flipped,
            )[0]["predicted_depth"]
        if self.family == "depth-pro":
            # The raw head predicts inverse depth. Use the official conversion to meters.
            outputs.predicted_depth = outputs.predicted_depth.float()
            if outputs.field_of_view is not None:
                outputs.field_of_view = outputs.field_of_view.float()
            return self.processor.post_process_depth_estimation(
                outputs, target_sizes=[shape],
            )[0]["predicted_depth"]
        return self.torch.nn.functional.interpolate(
            outputs.predicted_depth[:, None], size=shape, mode="bilinear", align_corners=False
        )[0, 0]

    def __call__(self, image: np.ndarray, turns: int) -> np.ndarray:
        upright = np.rot90(image, turns)
        rgb = cv2.cvtColor(upright, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        with self.lock, self.torch.inference_mode():
            try:
                depth = self._infer(inputs, rgb.shape[:2])
            except (RuntimeError, NotImplementedError) as error:
                if not self.allow_cpu_fallback or self.device != "mps" or (
                    "mps" not in str(error).lower() and not isinstance(error, NotImplementedError)
                ):
                    raise
                self.device = "cpu"
                self.dtype = self.torch.float32
                self.model.to(device=self.device, dtype=self.dtype)
                depth = self._infer(inputs, rgb.shape[:2])
            depth = depth.float().cpu().numpy()
        return np.rot90(depth, -turns)
