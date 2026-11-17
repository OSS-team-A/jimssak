"""Replace DemoDetector with a YOLO adapter when model integration is ready."""

from typing import Protocol

from ai.contracts import BoundingBox, Detection, Frame


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...


class DemoDetector:
    def detect(self, frame: Frame) -> list[Detection]:
        return [Detection(
            track_id=f"{frame.video_id}:demo:1",
            label="chair",
            confidence=0.95,
            bbox=BoundingBox(
                x1=frame.width_px * 0.25, y1=frame.height_px * 0.2,
                x2=frame.width_px * 0.65, y2=frame.height_px * 0.9,
            ),
        )]
