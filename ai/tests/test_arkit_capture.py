import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from ai.measurement.arkit_capture import (  # noqa: E402
    ARCaptureMetadata,
    CaptureObservations,
    make_job,
    verify_movie,
)


def capture_metadata(turns, count=2):
    transform = [[0, 0, 1, .4], [0, 1, 0, 1.3], [-1, 0, 0, 2.1], [0, 0, 0, 1]]
    return {
        "schema_version": 1, "video_file": "capture.avi", "video_id": "test-capture",
        "image_size_px": [64, 48], "pixel_rotation_ccw": turns,
        "frames": [
            {"frame_index": i, "timestamp_us": 55_000 * i,
             "camera_to_world": transform, "intrinsics": [[65 + i, 0, 31], [0, 73, 24], [0, 0, 1]]}
            for i in range(count)
        ],
    }


@pytest.mark.parametrize("turns", range(4))
def test_arkit_projection_matches_rotated_and_resized_pixels(turns):
    meta = ARCaptureMetadata.model_validate(capture_metadata(turns))
    pixels = (48, 64) if turns % 2 else (64, 48)
    observation = CaptureObservations(
        annotation_size_px=tuple(v * 2 for v in pixels),
        views=[{"track_id": "chair", "label": "chair", "frame_index": 1,
                "bbox": {"x1": 10, "x2": 30, "y1": 10, "y2": 30}}],
    )
    rgb, arkit = make_job(meta, observation, "capture.avi", compare=True).videos
    assert not rgb.capture_poses
    assert rgb.frame_intrinsics == arkit.frame_intrinsics
    assert rgb.frame_intrinsics[0] != rgb.frame_intrinsics[1]
    assert rgb.views == arkit.views
    pose = arkit.capture_poses[0]
    frame = meta.frames[1]
    transform = np.asarray(frame.camera_to_world)
    image = np.zeros((48, 64), np.uint8)
    u, v = 37, 21
    image[v, u] = 1
    y, x = np.argwhere(np.rot90(image, turns))[0]
    cv_ray = np.linalg.solve(frame.intrinsics, [u, v, 1]) * 2
    camera_point = np.diag([1, -1, -1]) @ cv_ray
    world = transform[:3, :3] @ camera_point + transform[:3, 3]
    camera_cv = np.asarray(pose.world_to_camera_rotation) @ world
    camera_cv += pose.world_to_camera_translation_m
    k = arkit.frame_intrinsics[1]
    assert camera_cv[2] == pytest.approx(2)
    assert (k.fx * camera_cv[0] / camera_cv[2] + k.cx,
            k.fy * camera_cv[1] / camera_cv[2] + k.cy) == pytest.approx((2 * x + .5, 2 * y + .5))


def test_rejects_missing_sidecar_index_and_extra_decoded_frames(tmp_path):
    raw = capture_metadata(0)
    raw["frames"][1]["frame_index"] = 3
    with pytest.raises(ValueError, match="contiguous"):
        ARCaptureMetadata.model_validate(raw)
    path = tmp_path / "capture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 15, (64, 48))
    assert writer.isOpened()
    try:
        for value in (40, 80):
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    verify_movie(path, ARCaptureMetadata.model_validate(capture_metadata(0)))
    with pytest.raises(ValueError, match="more frames"):
        verify_movie(path, ARCaptureMetadata.model_validate(capture_metadata(0, count=1)))
    with pytest.raises(ValueError, match="ended before"):
        verify_movie(path, ARCaptureMetadata.model_validate(capture_metadata(0, count=3)))
