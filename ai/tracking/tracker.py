"""Tracking interface. Demo IDs only identify the synthetic demo object."""

from typing import Protocol

from ai.contracts import Detection, Frame


class Tracker(Protocol):
    def update(self, frame: Frame, detections: list[Detection]) -> list[Detection]: ...


class DemoTracker:
    def update(self, frame: Frame, detections: list[Detection]) -> list[Detection]:
        return [item.model_copy(deep=True) for item in detections]
