"""Load an optional local metric-depth model for reuse across videos."""

from pathlib import Path
from threading import Lock

import cv2
import numpy as np


class MetricDepth:
    def __init__(
        self, model_path: Path, spatial_head: Path | None = None,
        sensor_head: Path | None = None, *, device: str = "auto",
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        available = {"cpu": True, "mps": torch.backends.mps.is_available(),
                     "cuda": torch.cuda.is_available()}
        self.device = next(name for name in ("mps", "cuda", "cpu") if available[name]) if (
            device == "auto"
        ) else device
        if not available.get(self.device, False):
            raise ValueError(f"Requested depth device is unavailable: {self.device}")
        self.allow_cpu_fallback = device == "auto"
        self.processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True, use_fast=False
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True
        )
        if getattr(self.model.config, "depth_estimation_type", None) != "metric":
            raise ValueError("Size estimation requires a metric depth checkpoint")
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

    def __call__(self, image: np.ndarray, turns: int) -> np.ndarray:
        upright = np.rot90(image, turns)
        rgb = cv2.cvtColor(upright, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        with self.lock, self.torch.inference_mode():
            try:
                estimated = self.model(**inputs.to(self.device)).predicted_depth
            except (RuntimeError, NotImplementedError) as error:
                if not self.allow_cpu_fallback or self.device != "mps" or (
                    "mps" not in str(error).lower() and not isinstance(error, NotImplementedError)
                ):
                    raise
                self.device = "cpu"
                self.model.to(self.device)
                estimated = self.model(**inputs.to(self.device)).predicted_depth
            depth = self.torch.nn.functional.interpolate(
                estimated[:, None], size=rgb.shape[:2], mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()
        return np.rot90(depth, -turns)
