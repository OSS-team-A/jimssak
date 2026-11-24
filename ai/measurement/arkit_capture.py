"""Turn an ARKit video, camera sidecar and manual or tracked masks into a measurement job."""

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import Field, model_validator

from ai.contracts import Contract
from ai.measurement.moving_contract import (
    CapturePose,
    Intrinsics,
    MeasurementJob,
    ObjectView,
    VideoInput,
)

Row3 = tuple[float, float, float]
Row4 = tuple[float, float, float, float]


class ARFrameMetadata(Contract):
    frame_index: int = Field(ge=0)
    timestamp_us: int = Field(ge=0)
    camera_to_world: tuple[Row4, Row4, Row4, Row4]
    intrinsics: tuple[Row3, Row3, Row3]


class ARCaptureMetadata(Contract):
    schema_version: Literal[1]
    video_file: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    image_size_px: tuple[int, int]
    pixel_rotation_ccw: Literal[0, 1, 2, 3] = 0
    frames: list[ARFrameMetadata] = Field(min_length=1)

    @model_validator(mode="after")
    def verify_sequence(self):
        if Path(self.video_file).name != self.video_file or self.video_file in {".", ".."}:
            raise ValueError("video_file must be a filename next to the sidecar")
        if min(self.image_size_px) <= 0:
            raise ValueError("Invalid captured image size")
        if any(frame.frame_index != i for i, frame in enumerate(self.frames)):
            raise ValueError("ARKit metadata must have one contiguous index per written frame")
        if self.frames[0].timestamp_us != 0 or any(
            a.timestamp_us >= b.timestamp_us for a, b in zip(self.frames, self.frames[1:])
        ):
            raise ValueError("Frame timestamps must start at zero and strictly increase")
        return self


class CaptureObservations(Contract):
    annotation_size_px: tuple[int, int]
    views: list[ObjectView] = Field(min_length=1)
    pose_stride: int = Field(default=12, ge=1)


def _pixel_rotation(turns, width, height):
    # Maps native pixel centers to the same coordinates as np.rot90.
    return np.array((
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, width - 1), (0, 0, 1)),
        ((-1, 0, width - 1), (0, -1, height - 1), (0, 0, 1)),
        ((0, -1, height - 1), (1, 0, 0), (0, 0, 1)),
    )[turns], dtype=float)


def make_job(metadata, observations, path, *, compare=False):
    width, height = metadata.image_size_px
    turns = metadata.pixel_rotation_ccw
    shape = (height, width) if turns % 2 else (width, height)
    if min(observations.annotation_size_px) <= 0:
        raise ValueError("Annotation dimensions must be positive")
    sx, sy = np.divide(observations.annotation_size_px, shape)
    if abs(sx / sy - 1) > 0.015:
        raise ValueError("Annotation size must preserve the aspect ratio of the rotated video")
    pixels = _pixel_rotation(turns, width, height)
    axes = pixels.copy()
    axes[:2, 2] = 0
    resize = np.array(((sx, 0, (sx - 1) / 2), (0, sy, (sy - 1) / 2), (0, 0, 1)))
    arkit_to_cv = axes @ np.diag([1, -1, -1])
    saved = {}
    cameras = {}
    for frame in metadata.frames:
        i = frame.frame_index
        transform = np.asarray(frame.camera_to_world)
        orientation = transform[:3, :3]
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5) or (
            not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(orientation), 1, atol=1e-3)
        ):
            raise ValueError(f"Invalid ARKit camera transform at frame {i}")
        native_k = np.asarray(frame.intrinsics)
        if not np.allclose(native_k[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1]):
            raise ValueError(f"Unsupported camera intrinsics at frame {i}")
        k = resize @ pixels @ native_k @ axes.T
        cameras[i] = Intrinsics(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])
        rotation = arkit_to_cv @ orientation.T
        saved[i] = CapturePose(
            frame_index=i, world_to_camera_rotation=rotation.tolist(),
            world_to_camera_translation_m=(-rotation @ transform[:3, 3]).tolist(),
        )
    ids = {view.frame_index for view in observations.views}
    if not ids <= saved.keys():
        raise ValueError("Observation references a frame outside the camera sidecar")
    # ARKit's gravity-aligned world's +Y is up. Give both comparison runs identical up vectors.
    views = [view.model_copy(update={
        "camera_up": tuple(row[1] for row in saved[view.frame_index].world_to_camera_rotation),
    }) for view in observations.views]
    common = dict(
        path=path, annotation_size_px=observations.annotation_size_px, views=views,
        frame_intrinsics=cameras, pixel_rotation_ccw=turns, pose_stride=observations.pose_stride,
    )
    videos = [VideoInput(video_id=metadata.video_id, capture_poses=[saved[i] for i in sorted(ids)],
                         **common)]
    if compare:
        videos.insert(0, VideoInput(video_id=f"{metadata.video_id}-rgb", **common))
    return MeasurementJob(videos=videos)


def verify_movie(path, metadata):
    """Check decoded dimensions and frame count without relying on the movie duration."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open the original capture: {path}")
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    try:
        ok, first = capture.read()
        if not ok or first.shape[:2][::-1] != metadata.image_size_px:
            raise ValueError("Movie dimensions do not match the camera sidecar")
        for frame in metadata.frames[1:]:
            if not capture.grab():
                raise ValueError(f"Movie ended before camera frame {frame.frame_index}")
        if capture.grab():
            raise ValueError("Movie has more frames than the camera sidecar")
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera_sidecar", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    meta = ARCaptureMetadata.model_validate_json(args.camera_sidecar.read_text())
    observations = CaptureObservations.model_validate_json(args.observations.read_text())
    path = args.camera_sidecar.resolve().parent / meta.video_file
    verify_movie(path, meta)
    job = make_job(meta, observations, path, compare=args.compare)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(job.model_dump_json(indent=2, exclude_none=True) + "\n")


if __name__ == "__main__":
    main()
