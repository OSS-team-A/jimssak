"""Run the explicit fixture pipeline: python -m ai.pipeline."""

from ai.contracts import AnalysisResult, Frame
from ai.detection.detector import DemoDetector
from ai.measurement.estimator import estimate
from ai.tracking.tracker import DemoTracker


def run_demo() -> AnalysisResult:
    frame = Frame(video_id="demo-room", frame_id=0, timestamp_ms=0, width_px=1280, height_px=720)
    tracked = DemoTracker().update(frame, DemoDetector().detect(frame))
    return AnalysisResult(
        mode="demo", frame=frame,
        detections=[estimate(frame, item) for item in tracked],
    )


if __name__ == "__main__":
    print(run_demo().model_dump_json(indent=2))
