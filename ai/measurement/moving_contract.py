"""Inputs shared with the recognition stage for offline video measurement."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ai.contracts import BoundingBox, Contract

Point = tuple[float, float]
Vector = tuple[float, float, float]
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
    mask_rle: list[int] | None = None
    tabletop_px: tuple[Point, Point, Point, Point] | None = None
    floor_contacts_px: list[Point] = Field(default_factory=list)
    overall_width_px: tuple[Point, Point] | None = None
    camera_up: tuple[float, float, float] | None = None
    complete: bool = True

    @model_validator(mode="after")
    def check_observation(self):
        if all(value is None for value in (self.bbox, self.mask_polygon_px, self.mask_rle,
                                           self.tabletop_px)):
            raise ValueError("Provide a box, silhouette, or tabletop")
        if self.mask_polygon_px is not None and self.mask_rle is not None:
            raise ValueError("Provide only one mask representation")
        if self.camera_up is not None and sum(v * v for v in self.camera_up) < 1e-8:
            raise ValueError("camera_up must have a nonzero length")
        return self

    @property
    def has_mask(self):
        return self.mask_polygon_px is not None or self.mask_rle is not None


class ScaleReference(Contract):
    track_id: str
    dimension: Dimension
    cm: float = Field(gt=0)


class CapturePose(Contract):
    frame_index: int = Field(ge=0)
    world_to_camera_rotation: tuple[Vector, Vector, Vector]
    world_to_camera_translation_m: Vector


class VideoInput(Contract):
    video_id: str = Field(min_length=1)
    path: Path
    annotation_size_px: tuple[int, int]
    intrinsics: Intrinsics | None = None
    pose_stride: int = Field(default=12, ge=1)
    views: list[ObjectView] = Field(min_length=1)
    scale_reference: ScaleReference | None = None
    capture_poses: list[CapturePose] = Field(default_factory=list)

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
            runs = view.mask_rle
            if runs is not None and (
                not runs or runs[0] < 0 or any(count <= 0 for count in runs[1:])
                or sum(runs) != width * height
            ):
                raise ValueError("Mask RLE must cover every pixel in the annotation image")
            bbox = view.bbox
            if bbox and (bbox.x2 > width or bbox.y2 > height):
                raise ValueError("Bounding box exceeds annotation dimensions")
            for x, y in (
                list(view.mask_polygon_px or []) + list(view.tabletop_px or [])
                + view.floor_contacts_px + list(view.overall_width_px or [])
            ):
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError("Points must be inside the annotation image")
        if self.capture_poses:
            provided = {pose.frame_index for pose in self.capture_poses}
            if len(provided) != len(self.capture_poses):
                raise ValueError("Capture pose frame indices must be unique")
            if any(view.frame_index not in provided for view in self.views):
                raise ValueError("Capture poses must cover every object observation")
            if self.intrinsics is None:
                raise ValueError("Capture poses require the corresponding camera intrinsics")
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
    scale_source: Literal["pretrained_depth", "capture_poses", "reference_dimension"]
    views: int = Field(gt=0)


class MeasuredTrack(Contract):
    track_id: str
    label: str
    status: Literal["estimated", "partial", "unavailable"]
    dimensions: dict[Dimension, DimensionEstimate | None]
    frames: list[int]
    qualifiers: list[str]


class ScaleResult(Contract):
    status: Literal["model_metric_scale", "capture_metric_scale", "reference_scaled", "unavailable"]
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
    pose_source: Literal["rgb_metric_depth_pnp", "capture_poses"]
    intrinsics_source: Literal["provided", "provided_approximate", "approximate_phone_fov"]
    scale: ScaleResult
    objects: list[MeasuredTrack]
    poses: list[PoseResult]
    depth_timing: DepthTiming
    source_sha256: str | None = None


class MeasurementResponse(Contract):
    videos: list[VideoResult]
