"""Shared v1 contract. Pixel coordinates refer to the original, unresized frame."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BoundingBox(Contract):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(gt=0)
    y2: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must have positive width and height")
        return self


class Camera(Contract):
    fx_px: float = Field(gt=0)
    fy_px: float = Field(gt=0)


class Frame(Contract):
    video_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    camera: Camera | None = None


class Dimensions(Contract):
    width_cm: float = Field(gt=0)
    height_cm: float = Field(gt=0)
    depth_cm: float | None = Field(default=None, gt=0)
    method: Literal["pinhole_bbox"] = "pinhole_bbox"


class Detection(Contract):
    track_id: str = Field(min_length=1)
    object_id: str | None = None
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    # Optional same-frame polygon in original pixel coordinates.
    mask_polygon_px: list[tuple[float, float]] | None = None
    depth_m: float | None = Field(default=None, gt=0)
    dimensions: Dimensions | None = None
    measurement_status: Literal["needs_calibration", "estimated"] = "needs_calibration"


class AnalysisResult(Contract):
    schema_version: Literal["1.0"] = "1.0"
    mode: Literal["demo", "model"]
    frame: Frame
    detections: list[Detection]

    @model_validator(mode="after")
    def within_frame(self):
        ids = [item.track_id for item in self.detections]
        if len(ids) != len(set(ids)):
            raise ValueError("track_id must be unique within a frame")
        for item in self.detections:
            if item.bbox.x2 > self.frame.width_px or item.bbox.y2 > self.frame.height_px:
                raise ValueError("bbox exceeds original frame")
            for x, y in item.mask_polygon_px or []:
                if not (0 <= x <= self.frame.width_px and 0 <= y <= self.frame.height_px):
                    raise ValueError("mask exceeds original frame")
        return self
