"""Load an optional local metric-depth model once per server process."""

from pathlib import Path
from threading import Lock

import cv2
import numpy as np


class MetricDepth:
    def __init__(self, model_path: Path, spatial_head: Path | None, sensor_head: Path | None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True, use_fast=False
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True
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

    def __call__(self, image: np.ndarray, turns: int) -> np.ndarray:
        upright = np.rot90(image, turns)
        rgb = cv2.cvtColor(upright, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.lock, self.torch.inference_mode():
            estimated = self.model(**inputs).predicted_depth
            depth = self.torch.nn.functional.interpolate(
                estimated[:, None], size=rgb.shape[:2], mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()
        return np.rot90(depth, -turns)
