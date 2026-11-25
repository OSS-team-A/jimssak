"""Metric multiview measurement for externally supplied instance masks."""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import cv2
import numpy as np

from ai.measurement.depth_alignment import align_depth_to_geometry
from ai.measurement.mask_height import tabletop_and_contacts, vertical_height

GRID_METERS = 0.015
MASK_MARGIN_KERNEL = np.ones((3, 3), np.uint8)


@dataclass
class MaskView:
    """One view of the same object in native image coordinates.

    The 3×4 matrix projects meter coordinates (+Z up) into calibrated pixels.
    Depth is in meters along the optical axis; image is optional BGR.
    """

    frame_id: int
    mask: np.ndarray
    matrix: np.ndarray | None = None
    turns: int = 0
    image: np.ndarray | None = None
    depth: np.ndarray | None = None

    def __post_init__(self):
        if self.mask.ndim != 2 or not self.mask.size:
            raise ValueError("Expected a nonempty two-dimensional mask array")
        self.mask = (self.mask != 0).astype(np.uint8)
        if self.turns not in (0, 1, 2, 3):
            raise ValueError("turns must be a number of counterclockwise quarter turns")
        if self.image is not None and self.image.shape != (*self.mask.shape, 3):
            raise ValueError("Image must be BGR and match the mask shape")
        if self.depth is not None and self.depth.shape != self.mask.shape:
            raise ValueError("Depth must match the mask shape")
        if self.matrix is not None:
            matrix = np.asarray(self.matrix, dtype=float)
            if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
                raise ValueError("Expected a finite 3×4 camera matrix")
            if np.linalg.cond(matrix[:, :3]) > 1e10:
                raise ValueError("Camera matrix must be invertible")
            self.matrix = matrix / np.linalg.norm(matrix[2, :3])


@dataclass
class TrackMeasurement:
    track_id: str
    label: str
    status: Literal[
        "estimated", "partial", "needs_calibration", "insufficient_views", "unavailable"
    ]
    method: Literal["multiview_geometry", "multiview_aligned_depth"] | None = None
    width_cm: float | None = None
    depth_cm: float | None = None
    height_cm: float | None = None
    frame_ids: list[int] = field(default_factory=list)
    height_frame_ids: list[int] = field(default_factory=list)
    depth_scales_by_frame: dict[int, float] = field(default_factory=dict)


def triangulate(rays):
    equations = [row for matrix, (x, y) in rays
                 for row in (x * matrix[2] - matrix[0], y * matrix[2] - matrix[1])]
    _, _, right = np.linalg.svd(np.asarray(equations), full_matrices=False)
    if abs(right[-1, 3]) < 1e-10:
        raise ValueError("Mask-center rays do not intersect at finite distance")
    return right[-1, :3] / right[-1, 3]


def project(points, matrix):
    values = points @ matrix[:, :3].T + matrix[:, 3]
    return values[:, :2] / np.maximum(values[:, 2:3], 1e-6), values[:, 2]


def mask_feature_bounds(views):
    """Use matched image features to bound poorly intersecting silhouette rays."""
    sift = cv2.SIFT_create(contrastThreshold=0.008)
    detector = cv2.BFMatcher()
    features = [
        sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask * 255)
        if image is not None else ([], None)
        for _, mask, image in views
    ]
    anchors = []
    for i, ((matrix, _, _), (keypoints, descriptors)) in enumerate(zip(views, features)):
        if descriptors is None:
            continue
        center = np.linalg.solve(matrix[:, :3], -matrix[:, 3])
        for j in range(i + 1, len(views)):
            other = views[j][0]
            other_center = np.linalg.solve(other[:, :3], -other[:, 3])
            other_keypoints, other_descriptors = features[j]
            if other_descriptors is None or np.linalg.norm(center - other_center) < 0.20:
                continue
            candidates = detector.knnMatch(descriptors, other_descriptors, k=2)
            matches = [
                pair[0] for pair in candidates
                if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance
            ]
            if not matches:
                continue
            left = np.array([keypoints[m.queryIdx].pt for m in matches])
            right = np.array([other_keypoints[m.trainIdx].pt for m in matches])
            xyz = cv2.triangulatePoints(matrix, other, left.T, right.T).T
            finite = np.abs(xyz[:, 3]) > 1e-9
            xyz = xyz[finite, :3] / xyz[finite, 3, None]
            left_px, left_z = project(xyz, matrix)
            right_px, right_z = project(xyz, other)
            good = ((left_z > 0) & (right_z > 0)
                    & (np.linalg.norm(left_px - left[finite], axis=1) < 1.8)
                    & (np.linalg.norm(right_px - right[finite], axis=1) < 1.8))
            anchors.extend(xyz[good])
    if len(anchors) < 8:
        raise ValueError("Silhouette ambiguous and too few triangulated mask features")
    lo, hi = np.percentile(anchors, [5, 95], axis=0)
    margin = np.maximum(0.08, 0.2 * (hi - lo))
    return lo - margin, hi + margin


def visual_hull(views):
    rays = []
    for matrix, mask, _ in views:
        y, x = np.where(mask)
        rays.append((matrix, (x.mean(), y.mean())))
    center = triangulate(rays)
    if not np.isfinite(center).all():
        raise ValueError("Mask centers do not provide a finite search region")
    # Scale the search window from observed pixel spans, camera focal lengths and
    # the triangulated center. No reference dimensions enter the reconstruction.
    extents = []
    for matrix, mask, _ in views:
        _, z = project(center[None], matrix)
        if z[0] <= 0:
            raise ValueError("Triangulated object center lies behind a selected camera")
        y, x = np.where(mask)
        fx = np.linalg.norm(np.cross(matrix[0, :3], matrix[2, :3]))
        fy = np.linalg.norm(np.cross(matrix[1, :3], matrix[2, :3]))
        extents.append(max(np.ptp(x) / fx, np.ptp(y) / fy) * z[0])
    radius = np.clip(2.0 * np.median(extents), 0.8, 3.0)
    stride = GRID_METERS
    offsets = np.arange(-radius, radius + stride * 0.5, stride, dtype=np.float32)
    axis = [values + offsets for values in center]
    # Carve in slices so large perspective masks do not dominate peak memory.
    gx, gy = np.meshgrid(*axis[:2], indexing="ij")
    xy = np.c_[gx.ravel(), gy.ravel()]
    hull = []
    expanded = [cv2.dilate(mask, MASK_MARGIN_KERNEL) for _, mask, _ in views]
    for level in axis[2]:
        xyz = np.c_[xy, np.full(len(xy), level)]
        for (camera, _, _), mask in zip(views, expanded):
            if not xyz.size:
                break
            uv, z = project(xyz, camera)
            x, y = np.rint(uv).astype(int).T
            h, w = mask.shape
            visible = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            xyz = xyz[visible][mask[y[visible], x[visible]] != 0]
        if len(xyz):
            hull.append(xyz)
    if not hull:
        raise ValueError("Selected masks do not intersect in 3D")
    points = np.concatenate(hull)
    if np.any(np.abs(points - center).max(axis=0) >= radius - 2 * stride):
        lo, hi = mask_feature_bounds(views)
        if np.any(lo < center - radius) or np.any(hi > center + radius):
            raise ValueError("Triangulated feature bounds extend outside the grid")
        points = points[((points >= lo) & (points <= hi)).all(axis=1)]
        if not len(points):
            raise ValueError("Triangulated feature bounds do not overlap mask volume")
    return points


def same_rays(hull, matrix, mask, depth):
    h, w = mask.shape
    uv, z = project(hull, matrix)
    x, y = np.rint(uv).astype(int).T
    in_image = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    zbuffer = np.full(h * w, np.inf)
    np.minimum.at(zbuffer, y[in_image] * w + x[in_image], z[in_image])
    zbuffer = zbuffer.reshape(h, w)
    y, x = np.where((mask != 0) & np.isfinite(zbuffer))
    if len(x) < 40:
        raise ValueError("Fewer than 40 valid shared mask rays")
    points = np.c_[x, y, np.ones(len(x))]
    origin = np.linalg.solve(matrix[:, :3], -matrix[:, 3])
    directions = np.linalg.solve(matrix[:, :3], points.T).T
    geometry = origin + directions * zbuffer[y, x, None]
    if depth is None:
        return np.c_[x, y], geometry, None
    measured = depth[y, x]
    valid = np.isfinite(measured) & (measured > 0)
    # Keep geometry available wherever depth is missing. Only valid model
    # depths contribute to the frame's scale estimate.
    if valid.sum() < 40:
        return np.c_[x, y], geometry, None
    predicted = origin + directions[valid] * measured[valid, None]
    aligned, scale = align_depth_to_geometry(matrix, geometry[valid], predicted)
    return np.c_[x, y], geometry, (aligned, scale, valid)


def _consistent_depth(geometry, aligned, matrix, silhouettes):
    """Keep depth corrections supported by both the residuals and other views."""
    residual_m = (aligned - geometry) @ matrix[2, :3]
    low, high = np.percentile(residual_m, [25, 75])
    # The IQR fence rejects the long background tail left by a single scale fit.
    # A two-voxel floor avoids rejecting corrections at the grid resolution.
    margin = max(1.5 * (high - low), 2 * GRID_METERS)
    supported = (residual_m >= low - margin) & (residual_m <= high + margin)
    indices = np.flatnonzero(supported)
    votes = np.zeros(len(indices), dtype=int)
    for projection, silhouette in silhouettes:
        pixels, z = project(aligned[indices], projection)
        x, y = np.rint(pixels).astype(int).T
        height, width = silhouette.shape
        visible = (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        visible[visible] = silhouette[y[visible], x[visible]] != 0
        votes += visible
    # Require three agreeing views and tolerate one disagreeing mask with four or more.
    supported[indices] = votes >= max(3, len(silhouettes) - 1)
    return supported


def _projected_footprint(pixels, camera_depth, shape):
    pixels = pixels[camera_depth > 0]
    footprint = np.zeros(shape, dtype=np.uint8)
    if len(pixels) >= 3:
        polygon = cv2.convexHull(pixels.astype(np.float32))
        cv2.fillConvexPoly(footprint, np.rint(polygon).astype(np.int32), 1)
    return footprint


def _recover_silhouette_edges(filtered, geometry, views):
    """Restore geometry where independent masks show that filtering cut off the object."""
    votes = np.zeros(len(geometry), dtype=int)
    for matrix, mask, _ in views:
        geometry_pixels, geometry_z = project(geometry, matrix)
        geometry_footprint = _projected_footprint(geometry_pixels, geometry_z, mask.shape)
        # Evaluate coverage against the input silhouette, not against the depth model.
        # Incomplete geometry cannot vouch for a missing edge.
        if np.mean(geometry_footprint[mask != 0]) < 0.95:
            continue
        pixels, z = project(filtered, matrix)
        footprint = _projected_footprint(pixels, z, mask.shape)
        footprint = cv2.dilate(footprint, np.ones((5, 5), dtype=np.uint8))
        uncovered = (mask != 0) & (footprint == 0)
        if not uncovered.any():
            continue
        x, y = np.rint(geometry_pixels).astype(int).T
        height, width = mask.shape
        visible = (geometry_z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        votes[visible] += uncovered[y[visible], x[visible]]
    # One mask can be wrong or partially occluded. Restore only corroborated points.
    return geometry[votes >= 2]


def sizes(points):
    # Use the identical bounding-box rule for both variants. The 1% tail trim
    # limits isolated mask/depth artifacts while preserving the observed extents.
    xy = points[:, :2]
    _, _, angle = cv2.minAreaRect(xy.astype(np.float32))
    rad = np.deg2rad(angle)
    directions = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    axes = np.c_[xy @ directions, points[:, 2]]
    span = np.diff(np.percentile(axes, [1, 99], axis=0), axis=0)[0] * 100
    return tuple(sorted(span[:2], reverse=True))


def _supported_geometric_depth(surfaces, centers):
    """Estimate the short horizontal extent when the masks constrain it across viewpoints."""
    if len(surfaces) < 3:
        return None
    object_center = np.median(np.concatenate(surfaces)[:, :2], axis=0)
    directions = np.asarray(centers)[:, :2] - object_center
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= GRID_METERS):
        return None
    directions /= lengths[:, None]
    # Moving toward an object changes its apparent size without revealing its side.
    if np.min(directions @ directions.T) > np.cos(np.deg2rad(15)):
        return None
    depths = [sizes(points)[1] for points in surfaces]
    depth = float(np.median(depths))
    spread = np.ptp(np.percentile(depths, [25, 75]))
    if depth <= 0 or spread > max(2 * GRID_METERS * 100, 0.1 * depth):
        return None
    return depth


def _surface_height(mask, pixels, points, turns):
    height, width = mask.shape
    x, y = pixels.T
    upright_pixels = (
        pixels,
        np.c_[y, width - 1 - x],
        np.c_[width - 1 - x, height - 1 - y],
        np.c_[height - 1 - y, x],
    )[turns]
    top, foot = tabletop_and_contacts(np.rot90(mask, turns), upright_pixels)
    return vertical_height(points[top], points[foot])


def is_complete_mask(mask):
    return (
        np.count_nonzero(mask) >= 40
        and not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    )


def measure_track(
    track_id: str, label: str, frames: list[MaskView], *, metric_capture_poses: bool = False
) -> TrackMeasurement:
    """Fuse complete masks of one stationary object and estimate its size in cm."""
    if len({frame.frame_id for frame in frames}) != len(frames):
        raise ValueError("Each view must have a unique frame ID")
    options = {"track_id": track_id, "label": label}
    if not frames:
        return TrackMeasurement(**options, status="insufficient_views")
    if not any(frame.matrix is not None for frame in frames):
        return TrackMeasurement(**options, status="needs_calibration")

    complete = [
        frame for frame in frames if frame.matrix is not None and is_complete_mask(frame.mask)
    ]
    centers = [np.linalg.solve(frame.matrix[:, :3], -frame.matrix[:, 3])
               for frame in complete]
    if len(centers) < 3 or max(
        np.linalg.norm(left - right) for left, right in combinations(centers, 2)
    ) < 0.15:
        return TrackMeasurement(**options, status="insufficient_views")

    views = [(frame.matrix, frame.mask, frame.image) for frame in complete]
    try:
        hull = visual_hull(views)
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        return TrackMeasurement(**options, status="unavailable")
    silhouettes = [(matrix, cv2.dilate(mask, MASK_MARGIN_KERNEL))
                   for matrix, mask, _ in views]
    frame_ids = []
    surfaces = []
    scales = {}
    heights = {}
    geometry_observations = []
    for frame in complete:
        mask = frame.mask
        try:
            pixels, geometry, aligned = same_rays(hull, frame.matrix, mask, frame.depth)
        except (ValueError, np.linalg.LinAlgError):
            continue
        geometry_observations.append((frame, pixels, geometry))
        points = geometry
        if aligned is not None:
            predicted, scale, valid = aligned
            supported = _consistent_depth(geometry[valid], predicted, frame.matrix, silhouettes)
            if supported.sum() >= 40:
                points = predicted[supported]
                pixels = pixels[valid][supported]
                scales[frame.frame_id] = scale
        frame_ids.append(frame.frame_id)
        surfaces.append(points)
        try:
            heights[frame.frame_id] = _surface_height(mask, pixels, points, frame.turns)
        except ValueError:
            continue

    if len(frame_ids) < 3:
        return TrackMeasurement(**options, status="unavailable", frame_ids=frame_ids)
    footprint_points = np.concatenate(surfaces)
    if scales:
        geometry_points = np.concatenate([points for _, _, points in geometry_observations])
        recovered = _recover_silhouette_edges(footprint_points, geometry_points, views)
        footprint_points = np.concatenate((footprint_points, recovered))
    width, depth = sizes(footprint_points)
    if min(width, depth) <= 0 or not np.isfinite((width, depth)).all():
        return TrackMeasurement(**options, status="unavailable", frame_ids=frame_ids)
    if metric_capture_poses and scales:
        geometry_surfaces = [points for _, _, points in geometry_observations]
        centers_by_frame = dict(zip((frame.frame_id for frame in complete), centers))
        geometry_centers = [centers_by_frame[frame.frame_id]
                            for frame, _, _ in geometry_observations]
        geometric_depth = _supported_geometric_depth(geometry_surfaces, geometry_centers)
        # A stable silhouette volume can retain a rear edge lost in aligned front surfaces.
        if geometric_depth is not None and geometric_depth - depth > 2 * GRID_METERS * 100:
            depth = min(width, geometric_depth)
    if scales and (metric_capture_poses or len(heights) < 2):
        # A single per-view depth scale does not preserve both tabletop and floor level.
        # With a measured trajectory, prioritize contact rays from silhouette geometry.
        geometry_heights = {}
        for frame, pixels, geometry in geometry_observations:
            try:
                geometry_heights[frame.frame_id] = _surface_height(
                    frame.mask, pixels, geometry, frame.turns
                )
            except ValueError:
                continue
        if len(geometry_heights) >= 2 or len(heights) < 2:
            heights = geometry_heights
    height = float(np.median(list(heights.values()))) if len(heights) >= 2 else None
    return TrackMeasurement(
        **options, status="estimated" if height is not None else "partial",
        method="multiview_aligned_depth" if scales else "multiview_geometry",
        width_cm=round(width, 2), depth_cm=round(depth, 2),
        height_cm=round(height, 2) if height is not None else None,
        frame_ids=frame_ids, height_frame_ids=list(heights), depth_scales_by_frame=scales,
    )
