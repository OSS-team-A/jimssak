"""Offline, recognition-independent measurement for moving-camera videos."""

import argparse
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai.measurement.depth_geometry import (
    floor_up,
    object_dimensions,
    polygon_mask,
    sample_point,
    table_dimensions,
    upright_basis,
)
from ai.measurement.metric_depth import DEFAULT_DEPTH_FAMILY, MetricDepth
from ai.measurement.moving_camera import CameraPose, CameraTracker, rays
from ai.measurement.moving_contract import (
    MeasurementJob,
    MeasurementResponse,
    ObjectView,
    VideoInput,
    VideoResult,
)
from ai.measurement.multiview import MaskView, measure_track, project, triangulate

MAX_KEYFRAMES = 180


@dataclass
class ObservedView:
    view: ObjectView
    pose: CameraPose
    camera: np.ndarray
    image: np.ndarray
    depth: np.ndarray
    mask: np.ndarray | None
    up: np.ndarray | None
    dimensions: dict[str, float]


def _camera(clip, frame_index=None):
    w, h = clip.annotation_size_px
    k = clip.frame_intrinsics.get(frame_index, clip.intrinsics)
    if k is not None:
        return np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]])
    # Approximate 74-degree diagonal field of view for a standard phone camera.
    focal = 0.66 * np.hypot(w, h)
    return np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])


def _video_frames(clip, first, last, selected):
    video = cv2.VideoCapture(str(clip.path))
    if not video.isOpened():
        raise ValueError(f"Cannot open video: {clip.path}")
    video.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    try:
        # Decode in presentation order rather than seeking by frame number in a VFR movie.
        for index in range(last + 1):
            if not video.grab():
                raise ValueError(f"Video {clip.video_id} ended before required frame {index}")
            if index < first or index not in selected:
                continue
            ok, image = video.retrieve()
            if not ok:
                raise ValueError(f"Cannot decode {clip.video_id} frame {index}")
            image = np.rot90(image, clip.pixel_rotation_ccw)
            width, height = clip.annotation_size_px
            raw_h, raw_w = image.shape[:2]
            if abs(raw_w / raw_h / (width / height) - 1) > 0.015:
                raise ValueError("Decoded video and annotation image have different aspect ratios")
            yield index, cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    finally:
        video.release()


def _shape_mask(view, shape):
    if view.mask_polygon_px is not None:
        return polygon_mask(shape, view.mask_polygon_px)
    if view.mask_rle is not None:
        counts = view.mask_rle
        return np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts).reshape(shape)
    if view.bbox is not None:
        mask = np.zeros(shape, np.uint8)
        box = view.bbox
        mask[int(box.y1):int(np.ceil(box.y2)), int(box.x1):int(np.ceil(box.x2))] = 1
        return mask
    return None


def _observe(view, pose, image, depth, camera):
    shape = _shape_mask(view, depth.shape)
    dimensions = {}
    up = None
    if view.camera_up is not None:
        up = np.asarray(view.camera_up, dtype=float)
        up /= np.linalg.norm(up)
    if view.tabletop_px is not None:
        dimensions, up = table_dimensions(view, depth, camera, up)
    if shape is not None:
        if up is None:
            up = floor_up(depth, camera, shape)
        generic = object_dimensions(view, shape, depth, camera, up)
        dimensions = generic | dimensions
    if view.overall_width_px is not None and view.complete:
        endpoints = [sample_point(p, depth, camera) for p in view.overall_width_px]
        if all(point is not None for point in endpoints):
            diff = endpoints[0] - endpoints[1]
            if up is not None:
                diff -= (diff @ up) * up
            dimensions["overall_width"] = float(np.linalg.norm(diff) * 100)
    return ObservedView(view, pose, camera, image, depth, shape, up, dimensions)


def _table_from_motion(observations):
    """Use multiview top corners only when motion and model depth both support them."""
    tables = [item for item in observations if item.view.tabletop_px is not None]
    if len(tables) < 3:
        return None
    projections = [
        item.camera @ np.c_[item.pose.rotation, item.pose.translation] for item in tables
    ]
    world_corners = []
    for i in range(4):
        pixels = [item.view.tabletop_px[i] for item in tables]
        directions = np.array([item.pose.rotation.T @ rays([px], item.camera)[0]
                               for item, px in zip(tables, pixels)])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        if np.min(directions @ directions.T) > np.cos(np.deg2rad(2.0)):
            return None
        try:
            point = triangulate(zip(projections, pixels))
        except (ValueError, np.linalg.LinAlgError):
            return None
        if not np.isfinite(point).all():
            return None
        for item, matrix, pixel in zip(tables, projections, pixels):
            projected, z = project(point[None], matrix)
            sample = sample_point(pixel, item.depth, item.camera)
            if z[0] <= 0 or np.linalg.norm(projected[0] - pixel) > 4.0 or sample is None:
                return None
            if abs(z[0] / sample[2] - 1) > 0.30:
                return None
        world_corners.append(point)
    corners = np.asarray(world_corners)
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1) * 100
    if any(abs(a - b) > 0.25 * max(a, b) for a, b in zip(edges[:2], edges[2:])):
        return None
    return {"top_width": float(np.mean(edges[::2])), "top_depth": float(np.mean(edges[1::2]))}


def _registered_mask_size(items, has_capture_poses):
    candidates = [item for item in items if item.view.has_mask and item.view.complete]
    if len(candidates) < 3:
        return {}
    up_hints = [item.pose.rotation.T @ item.up for item in candidates if item.up is not None]
    if not up_hints:
        return {}
    up = np.median(up_hints, axis=0)
    if np.linalg.norm(up) < 0.5:
        return {}
    basis = upright_basis(up)
    registered = [
        MaskView(item.view.frame_index, item.mask,
                 item.camera @ np.c_[item.pose.rotation @ basis, item.pose.translation],
                 image=item.image, depth=item.depth)
        for item in candidates
    ]
    first = candidates[0].view
    result = measure_track(
        first.track_id, first.label, registered, metric_capture_poses=has_capture_poses
    )
    return {
        key: {"cm": value, "method": "registered_multiview", "views": len(ids)}
        for key, value, ids in (
            ("width", result.width_cm, result.frame_ids),
            ("depth", result.depth_cm, result.frame_ids),
            ("height", result.height_cm, result.height_frame_ids),
        )
        if value is not None
    }


def _estimate_track(observations, has_capture_poses):
    component_views = defaultdict(list)
    per_dimension = defaultdict(list)
    for item in observations:
        component_views[item.pose.component].append(item)
        for key, value in item.dimensions.items():
            if np.isfinite(value) and value > 0:
                per_dimension[key].append(value)
    dimensions = {key: {"cm": float(np.median(values)),
                        "method": "metric_depth_view_median", "views": len(values),
                        "scale_source": "pretrained_depth"}
                  for key, values in per_dimension.items()}
    best_first = sorted(component_views.values(), key=len, reverse=True)
    for items in best_first:
        registered = _registered_mask_size(items, has_capture_poses)
        tabletop = _table_from_motion(items)
        if registered or tabletop:
            if "height" in dimensions and any(item.view.floor_contacts_px for item in items):
                registered.pop("height", None)
            dimensions.update(registered)
            tables = sum(item.view.tabletop_px is not None for item in items)
            for key, value in (tabletop or {}).items():
                dimensions[key] = {"cm": value, "method": "registered_multiview", "views": tables}
            scale_source = "capture_poses" if has_capture_poses else "pretrained_depth"
            for item in dimensions.values():
                if item["method"] == "registered_multiview":
                    item["scale_source"] = scale_source
            break
    first = observations[0].view
    fields = ("width", "depth", "height", "top_width", "top_depth", "overall_width")
    has_shape = any(item.mask is not None for item in observations)
    requested = {"width", "depth", "height"} if has_shape else {"top_width", "top_depth", "height"}
    if any(item.view.overall_width_px is not None for item in observations):
        requested.add("overall_width")
    status = "estimated" if requested <= dimensions.keys() else "partial"
    if not dimensions:
        status = "unavailable"
    qualifiers = set()
    if any(item.view.bbox and not item.view.has_mask for item in observations):
        qualifiers.add("bounding_box_is_approximate")
    if any(item["method"] == "metric_depth_view_median" for item in dimensions.values()):
        qualifiers.add("single_view_depth_scale_used")
    return {"track_id": first.track_id, "label": first.label, "status": status,
            "dimensions": {key: dimensions.get(key) for key in fields},
            "frames": [item.view.frame_index for item in observations],
            "qualifiers": sorted(qualifiers)}


def _calibrate(clip, objects):
    reference = clip.scale_reference
    if reference is None:
        return {"status": "capture_metric_scale" if clip.capture_poses else "model_metric_scale"}
    target = next(item for item in objects if item["track_id"] == reference.track_id)
    dimension = target["dimensions"][reference.dimension]
    if dimension is None:
        return {"status": "unavailable", "reference": reference.model_dump()}
    factor = reference.cm / dimension["cm"]
    for item in objects:
        for estimate in item["dimensions"].values():
            if estimate is not None:
                estimate["cm"] *= factor
                estimate["scale_source"] = "reference_dimension"
    return {"status": "reference_scaled", "reference": reference.model_dump(), "factor": factor}


def estimate_video(clip: VideoInput, model: MetricDepth) -> VideoResult:
    camera = _camera(clip)
    tracker = CameraTracker(camera) if not clip.capture_poses else None
    capture_poses = {}
    for item in clip.capture_poses:
        rotation = np.asarray(item.world_to_camera_rotation)
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-3
        ):
            raise ValueError(f"Invalid capture rotation at frame {item.frame_index}")
        capture_poses[item.frame_index] = CameraPose(
            rotation, np.asarray(item.world_to_camera_translation_m), component=0
        )
    lookup = defaultdict(list)
    for view in clip.views:
        lookup[view.frame_index].append(view)
    first, last = min(lookup), max(lookup)
    if len(lookup) > MAX_KEYFRAMES:
        raise ValueError(f"At most {MAX_KEYFRAMES} annotated frames are allowed per video")
    slots = max(1, MAX_KEYFRAMES - len(lookup))
    stride = max(clip.pose_stride, int(np.ceil((last - first + 1) / slots)))
    keyframes = set(lookup)
    if tracker is not None:
        keyframes.update(range(first, last + 1, stride))
    observations = defaultdict(list)
    poses = []
    timings = []
    for index, image in _video_frames(clip, first, last, keyframes):
        start = time.perf_counter()
        depth = model(image, 0)
        timings.append(time.perf_counter() - start)
        frame_camera = _camera(clip, index)
        pose = (
            tracker.add(index, image, depth, frame_camera)
            if tracker is not None else capture_poses[index]
        )
        poses.append({
            "frame_index": index, "component": pose.component, "parent_frame": pose.parent_frame,
            "inliers": pose.inliers, "median_reprojection_error_px": pose.error_px,
            "parallax_deg": pose.parallax_deg, "camera_center_m": pose.center.tolist(),
        })
        for view in lookup.get(index, ()):
            observations[view.track_id].append(
                _observe(view, pose, image, depth, frame_camera)
            )

    objects = [
        _estimate_track(items, bool(capture_poses)) for items in observations.values()
    ]
    calibration = _calibrate(clip, objects)
    source = "approximate_phone_fov"
    if clip.frame_intrinsics:
        source = "provided_per_frame"
    elif clip.intrinsics is not None:
        source = "provided_approximate" if clip.intrinsics.approximate else "provided"
    return VideoResult(
        video_id=clip.video_id, device=model.device, intrinsics_source=source,
        depth_model_family=getattr(model, "family", None),
        pose_source="capture_poses" if capture_poses else "rgb_metric_depth_pnp",
        scale=calibration, objects=objects, poses=poses,
        depth_timing={"frames": len(timings), "total_s": sum(timings),
                      "median_s": float(np.median(timings))},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--model",
        help=f"Model family, local checkpoint or cached ID (default: {DEFAULT_DEPTH_FAMILY})",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    job = MeasurementJob.model_validate_json(args.manifest.read_text())
    cv2.setNumThreads(2)
    model = MetricDepth(args.model, device=args.device)
    model.torch.set_num_threads(4)
    results = []
    for clip in job.videos:
        if not clip.path.is_absolute():
            clip.path = args.manifest.parent / clip.path
        result = estimate_video(clip, model)
        with clip.path.open("rb") as stream:
            result.source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(MeasurementResponse(videos=results).model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
