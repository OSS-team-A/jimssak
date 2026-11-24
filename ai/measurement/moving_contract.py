"""Inputs shared with the recognition stage for offline video measurement."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ai.contracts import BoundingBox, Contract

Point = tuple[float, float]
Dimension = Literal["width", "depth", "height", "top_width", "top_depth", "overall_width"]


class Intrinsics(Contract):
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float = Field(ge=0)
    cy: float = Field(ge=0)
    approximate: bool = False


class ObjectView(Contract):
    frame_index: int = Field(ge=0)
    track_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    bbox: BoundingBox | None = None
    mask_polygon_px: list[Point] | None = Field(default=None, min_length=3)
    tabletop_px: tuple[Point, Point, Point, Point] | None = None
    floor_contacts_px: list[Point] = Field(default_factory=list)
    overall_width_px: tuple[Point, Point] | None = None
    camera_up: tuple[float, float, float] | None = None
    complete: bool = True

    @model_validator(mode="after")
    def check_observation(self):
        if self.bbox is None and self.mask_polygon_px is None and self.tabletop_px is None:
            raise ValueError("Provide a box, silhouette polygon, or tabletop")
        if self.camera_up is not None and sum(v * v for v in self.camera_up) < 1e-8:
            raise ValueError("camera_up must have a nonzero length")
        return self


class ScaleReference(Contract):
    track_id: str
    dimension: Dimension
    cm: float = Field(gt=0)


class VideoInput(Contract):
    video_id: str = Field(min_length=1)
    path: Path
    annotation_size_px: tuple[int, int]
    intrinsics: Intrinsics | None = None
    pose_stride: int = Field(default=12, ge=1)
    views: list[ObjectView] = Field(min_length=1)
    scale_reference: ScaleReference | None = None

    @model_validator(mode="after")
    def check_video(self):
        width, height = self.annotation_size_px
        if not (1 <= width <= 1920 and 1 <= height <= 1920):
            raise ValueError("annotation dimensions must be between 1 and 1920 pixels")
        if self.intrinsics and not (
            0 <= self.intrinsics.cx < width and 0 <= self.intrinsics.cy < height
        ):
            raise ValueError("Principal point must be inside the annotation image")
        ids = set()
        for view in self.views:
            key = (view.frame_index, view.track_id)
            if key in ids:
                raise ValueError("Each (frame_index, track_id) needs a unique observation")
            ids.add(key)
            bbox = view.bbox
            if bbox and (bbox.x2 > width or bbox.y2 > height):
                raise ValueError("Bounding box exceeds annotation dimensions")
            for x, y in (
                list(view.mask_polygon_px or []) + list(view.tabletop_px or [])
                + view.floor_contacts_px + list(view.overall_width_px or [])
            ):
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError("Points must be inside the annotation image")
        if self.scale_reference and not any(
            view.track_id == self.scale_reference.track_id for view in self.views
        ):
            raise ValueError("Scale reference track_id must be present in this video")
        return self


class MeasurementJob(Contract):
    videos: list[VideoInput] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_video_ids(self):
        if len({clip.video_id for clip in self.videos}) != len(self.videos):
            raise ValueError("video_id must be unique within a job")
        return self


class DimensionEstimate(Contract):
    cm: float = Field(gt=0)
    method: Literal["metric_depth_view_median", "registered_multiview"]
    views: int = Field(gt=0)


class MeasuredTrack(Contract):
    track_id: str
    label: str
    status: Literal["estimated", "partial", "unavailable"]
    dimensions: dict[Dimension, DimensionEstimate | None]
    frames: list[int]
    qualifiers: list[str]


class ScaleResult(Contract):
    status: Literal["model_metric_scale", "reference_scaled", "unavailable"]
    reference: ScaleReference | None = None
    factor: float | None = Field(default=None, gt=0)


class PoseResult(Contract):
    frame_index: int
    component: int
    parent_frame: int | None
    inliers: int
    median_reprojection_error_px: float | None
    parallax_deg: float
    camera_center_m: tuple[float, float, float]


class DepthTiming(Contract):
    frames: int
    total_s: float
    median_s: float


class VideoResult(Contract):
    video_id: str
    device: str
    intrinsics_source: Literal["provided", "provided_approximate", "approximate_phone_fov"]
    scale: ScaleResult
    objects: list[MeasuredTrack]
    poses: list[PoseResult]
    depth_timing: DepthTiming
    source_sha256: str | None = None


class MeasurementResponse(Contract):
    videos: list[VideoResult]
