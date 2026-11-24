"""Align a frame's predicted depths to paired metric geometry."""

import numpy as np


def align_depth_to_geometry(projection, geometry_xyz_m, depth_xyz_m):
    """Return aligned 3D points and the estimated depth scale for one frame.

    Corresponding rows are observations of the same image pixel. The camera
    projection maps world coordinates in meters to homogeneous image pixels.
    Scaling from the camera center preserves each observation's image ray.
    """
    projection = np.asarray(projection, dtype=float)
    geometry = np.asarray(geometry_xyz_m, dtype=float)
    depth = np.asarray(depth_xyz_m, dtype=float)
    if projection.shape != (3, 4) or geometry.ndim != 2 or geometry.shape[1] != 3:
        raise ValueError("Expected a 3×4 projection and N×3 world points")
    if geometry.shape != depth.shape:
        raise ValueError("Geometry and depth must have paired observations")
    if not all(np.isfinite(array).all() for array in (projection, geometry, depth)):
        raise ValueError("Projection and paired points must be finite")

    camera_matrix = projection[:, :3]
    try:
        center = np.linalg.solve(camera_matrix, -projection[:, 3])
    except np.linalg.LinAlgError as error:
        raise ValueError("Camera projection is singular") from error
    geo_depth = (geometry - center) @ projection[2, :3]
    predicted_depth = (depth - center) @ projection[2, :3]
    valid = (geo_depth > 0) & (predicted_depth > 0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("At least three paired points in front of the camera are required")
    scale = float(np.median(geo_depth[valid] / predicted_depth[valid]))
    return center + (depth - center) * scale, scale
