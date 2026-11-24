"""Derive metric dimensions from manually selected depth observations."""

import cv2
import numpy as np

from ai.measurement.moving_camera import backproject, rays
from ai.measurement.multiview import sizes


def sample_point(pixel, depth, camera):
    # Take a small median to suppress isolated model noise at annotated landmarks.
    x, y = np.rint(pixel).astype(int)
    x, y = min(x, depth.shape[1] - 1), min(y, depth.shape[0] - 1)
    nearby = depth[max(0, y - 1):y + 2, max(0, x - 1):x + 2].ravel()
    nearby = nearby[np.isfinite(nearby) & (nearby > 0.15) & (nearby < 20.0)]
    return backproject([pixel], np.array([np.median(nearby)]), camera)[0] if len(nearby) else None


def cloud(mask, depth, camera):
    y, x = np.where(mask & np.isfinite(depth) & (depth > 0.15) & (depth < 20.0))
    step = max(1, len(x) // 8000)
    return backproject(np.c_[x[::step], y[::step]], depth[y[::step], x[::step]], camera)


def polygon_mask(shape, coordinates):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(coordinates).astype(np.int32)], 1)
    return mask


def _rectangle_normal(corner_rays):
    # The rectangle's vanishing directions give its normal without a known length.
    edges = np.cross(corner_rays, np.roll(corner_rays, -1, axis=0))
    a = np.cross(edges[0], edges[2])
    b = np.cross(edges[1], edges[3])
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-8:
        return None
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    if abs(a @ b) > 0.35:
        return None
    normal = np.cross(a, b)
    return normal / np.linalg.norm(normal)


def level_tabletop(corners, depth, camera, camera_up=None):
    corner_rays = rays(corners, camera)
    normal = _rectangle_normal(corner_rays) if camera_up is None else camera_up.copy()
    if normal is None:
        return None
    mask = polygon_mask(depth.shape, corners)
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(bool)
    points = cloud(inner, depth, camera)
    if len(points) < 30:
        return None
    if camera_up is None and normal[1] > 0:
        normal *= -1
    height_along_normal = points @ normal
    offset = np.median(height_along_normal)
    denom = corner_rays @ normal
    if np.any(np.abs(denom) < 0.02):
        return None
    z = offset / denom
    if np.any((z < 0.15) | (z > 20.0)):
        return None
    return corner_rays * z[:, None], normal, offset


def table_dimensions(view, depth, camera, up=None):
    corners = np.asarray(view.tabletop_px, dtype=float)
    table = level_tabletop(corners, depth, camera, up)
    dimensions = {}
    if table is not None:
        points, up, offset = table
        contacts = [sample_point(pixel, depth, camera) for pixel in view.floor_contacts_px]
        heights = [100 * (offset - point @ up) for point in contacts if point is not None]
        if len(heights) >= 2 and min(heights) > 0 and np.ptp(heights) < 0.30 * np.median(heights):
            dimensions["height"] = float(np.median(heights))
    else:
        return dimensions, up
    edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) * 100
    for key, sides in (("top_width", edges[::2]), ("top_depth", edges[1::2])):
        if min(sides) > 0 and min(sides) >= 0.55 * max(sides):
            dimensions[key] = float(np.mean(sides))
    return dimensions, up


def floor_up(depth, camera, occupied):
    # Infer a ground normal near the bottom of the frame for box-only input.
    region = ~occupied.astype(bool)
    region[:int(depth.shape[0] * 0.72)] = False
    points = cloud(region, depth, camera)
    if len(points) < 80:
        return None
    points = points[::max(1, len(points) // 1600)]
    rng = np.random.default_rng(568)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(80):
        a, b, c = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8 or abs(normal[1]) / length < 0.65:
            continue
        support = np.abs((points - a) @ (normal / length)) < 0.05
        if support.sum() > best.sum():
            best = support
    if best.sum() < max(50, 0.55 * len(points)):
        return None
    inliers = points[best] - np.mean(points[best], axis=0)
    _, spread, vectors = np.linalg.svd(inliers, full_matrices=False)
    if spread[1] < 0.08 * np.sqrt(len(inliers)):
        return None
    return vectors[-1] * (-1 if vectors[-1, 1] > 0 else 1)


def upright_basis(up):
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    right = np.cross(up, np.eye(3)[np.argmin(np.abs(up))])
    right /= np.linalg.norm(right)
    return np.stack((right, np.cross(up, right), up), axis=1)


def object_dimensions(view, shape_mask, depth, camera, up):
    if up is None or not view.complete:
        return {}
    foreground = shape_mask.astype(bool)
    if view.mask_polygon_px is None:
        # A detection box may include a wall or floor. Keep the depth cluster
        # around its center and report these as box-based estimates.
        x1, y1, x2, y2 = map(int, (
            view.bbox.x1, view.bbox.y1, view.bbox.x2, view.bbox.y2,
        ))
        inner = depth[y1 + (y2 - y1) // 4:y2 - (y2 - y1) // 4,
                      x1 + (x2 - x1) // 4:x2 - (x2 - x1) // 4]
        valid = inner[np.isfinite(inner) & (inner > 0.15) & (inner < 20.0)]
        if len(valid) < 20:
            return {}
        q1, q3 = np.percentile(valid, [25, 75])
        margin = max(0.1, min(1.5 * (q3 - q1), 0.2 * np.median(valid)))
        foreground &= (depth >= q1 - margin) & (depth <= q3 + margin)
    points = cloud(foreground, depth, camera)
    if len(points) < 60:
        return {}
    points = points @ upright_basis(up)
    width, length = sizes(points)
    height = float(np.ptp(np.percentile(points[:, 2], [1, 99])) * 100)
    return {"width": float(width), "depth": float(length), "height": height}
