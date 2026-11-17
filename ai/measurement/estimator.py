"""Coarse frontal extent from calibrated metric depth, not a 3D reconstruction."""

from ai.contracts import Detection, Dimensions, Frame


def estimate(frame: Frame, detection: Detection) -> Detection:
    result = detection.model_copy(deep=True)
    result.dimensions = None
    result.measurement_status = "needs_calibration"
    if frame.camera is None or detection.depth_m is None:
        return result
    box = detection.bbox
    result.dimensions = Dimensions(
        width_cm=(box.x2 - box.x1) * detection.depth_m / frame.camera.fx_px * 100,
        height_cm=(box.y2 - box.y1) * detection.depth_m / frame.camera.fy_px * 100,
    )
    result.measurement_status = "estimated"
    return result
