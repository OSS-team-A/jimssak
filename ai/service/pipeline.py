"""One video in, one v2.0 result out: recognize, measure, calibrate, snap beds, reshape.

Recognition (finding furniture and its masks per frame) belongs to the recognition stage
and is passed in as a *recognizer*: a function that takes the video path and returns a
:class:`Recognition`. Until that stage is connected, ``prepared_job`` replays the
observations of a measurement job file, which is how development videos are tested.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ai.measurement import calibration, standard_sizes
from ai.measurement.moving_contract import MeasurementJob, MeasurementResponse, VideoInput
from ai.service import result_v2

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "measurement" / "calibration"
# Calibrated, and about 0.06 s per frame on a Mac; Depth Pro takes about 10 s.
SERVICE_DEPTH_MODEL = "depth-anything-v2"


@dataclass
class Recognition:
    """Observations for one video. ``video.path`` is replaced with the analyzed file."""

    video: VideoInput
    confidences: dict[str, float] = field(default_factory=dict)


Recognizer = Callable[[Path], Recognition]


class NoFurniture(Exception):
    """The video shows no furniture the recognizer could follow."""


def prepared_job(manifest: Path, index: int = 0) -> Recognizer:
    """Recognizer that reuses the observations of one video in a measurement job file."""
    clip = MeasurementJob.model_validate_json(Path(manifest).read_text()).videos[index]

    def recognize(_path: Path) -> Recognition:
        return Recognition(video=clip)

    return recognize


def load_recognizer(spec: str) -> Recognizer:
    """``package.module:function`` → that function (called with the video path)."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError("Use package.module:function")
    return getattr(importlib.import_module(module), name)


def load_calibration(family: str | None, directory: Path = CALIBRATION_DIR) -> dict | None:
    path = directory / f"{family}.json"
    return json.loads(path.read_text()) if family and path.exists() else None


def analyze(path: Path, recognition: Recognition, measure, *, analysis_id: str,
            video_id: str, region: str = "kr", calibration_dir: Path = CALIBRATION_DIR,
            progress: Callable[[float], None] | None = None) -> dict:
    """Measure ``path`` with ``measure(VideoInput) -> VideoResult`` and build the result."""
    clip = recognition.video.model_copy(update={"path": Path(path), "video_id": video_id})
    if not clip.views:
        raise NoFurniture
    result = measure(clip)
    if progress:
        progress(0.9)
    warnings = []
    response = MeasurementResponse(videos=[result])
    factors = load_calibration(result.depth_model_family, calibration_dir)
    if factors is None:
        warnings.append(result_v2.WARNINGS["size_not_calibrated"])
    else:
        response = calibration.apply(response, factors)
    response = standard_sizes.snap(response, region)
    fps, duration = result_v2.video_info(Path(path))
    return result_v2.completed(
        response.videos[0], analysis_id=analysis_id, video_id=video_id, fps=fps,
        duration_s=duration, confidences=recognition.confidences, warnings=warnings,
    )
