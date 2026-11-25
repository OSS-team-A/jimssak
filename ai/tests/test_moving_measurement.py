import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from ai.measurement.depth_geometry import table_dimensions  # noqa: E402
from ai.measurement.moving_camera import CameraTracker, relative_pose  # noqa: E402
from ai.measurement.moving_contract import ObjectView, VideoInput  # noqa: E402
from ai.measurement.moving_video import _calibrate  # noqa: E402


def projection(points, camera):
    projected = points @ camera.T
    return projected[:, :2] / projected[:, 2:]


@pytest.mark.parametrize("vary_intrinsics", (False, True))
def test_translation_and_rotation_only_are_distinguished(vary_intrinsics):
    rng = np.random.default_rng(553)
    points = rng.uniform([-1, -0.7, 2.0], [1, 0.7, 4.0], size=(80, 3))
    camera = np.array([[440., 0, 320], [0, 440., 240], [0, 0, 1]])
    rotation = cv2.Rodrigues(np.array([0.03, -0.08, 0.01]))[0]
    source = projection(points, camera)
    target_k = camera.copy()
    if vary_intrinsics:
        target_k[0, 0] *= 1.07
        target_k[1, 1] *= 1.05
        target_k[0, 2] += 7
    for movement in [np.array([0.30, 0, 0.05]), np.zeros(3)]:
        target = projection(points @ rotation.T + movement, target_k)
        result = relative_pose(source, target, points[:, 2], camera, target_k)
        assert result is not None
        assert result[0] == pytest.approx(rotation, abs=1e-5)
        assert result[1] == pytest.approx(movement, abs=1e-5)
        assert (result[4] > 1) == bool(np.linalg.norm(movement))


def test_tabletop_and_floor_are_distinct_metric_planes():
    camera = np.array([[320., 0, 240], [0, 320., 170], [0, 0, 1]])
    tabletop = np.array([[-.5, .9, 2.5], [.5, .9, 2.5], [.5, .9, 2], [-.5, .9, 2]])
    floor = np.array([[-.3, 1.6, 2.2], [.3, 1.6, 2.3]])
    row = np.arange(450, dtype=float)[:, None]
    depth = np.broadcast_to(.9 * 320 / np.maximum(1, row - 170), (450, 480)).copy()
    for x, y in np.rint(projection(floor, camera)).astype(int):
        depth[y - 2:y + 3, x - 2:x + 3] = 1.6 * 320 / (row[y - 2:y + 3] - 170)
    observation = ObjectView(
        frame_index=0, track_id="desk", label="desk",
        tabletop_px=projection(tabletop, camera).tolist(),
        floor_contacts_px=projection(floor, camera).tolist(),
    )
    estimates, _ = table_dimensions(observation, depth, camera, np.array([0., -1., 0.]))
    assert estimates == pytest.approx({"top_width": 100., "top_depth": 50., "height": 70.}, abs=1)


def test_scale_only_uses_the_explicit_dimension():
    view = {"frame_index": 0, "track_id": "desk", "label": "desk",
            "bbox": {"x1": 1, "y1": 1, "x2": 10, "y2": 10}}
    clip = VideoInput(
        video_id="clip", path="video.mp4", annotation_size_px=(640, 480), views=[view],
        scale_reference={"track_id": "desk", "dimension": "top_width", "cm": 59},
    )
    results = [{"track_id": "desk", "dimensions": {
        "top_width": {"cm": 118}, "top_depth": {"cm": 72}, "overall_width": None,
    }}]
    assert _calibrate(clip, results)["factor"] == 0.5
    assert results[0]["dimensions"] == {
        "top_width": {"cm": 59, "scale_source": "reference_dimension"},
        "top_depth": {"cm": 36, "scale_source": "reference_dimension"}, "overall_width": None,
    }


def test_video_emits_typed_result_from_mock_box_and_capture_pose(monkeypatch):
    from ai.measurement import moving_video

    class MockDepth:
        device = "cpu"

        def __call__(self, image, turns):
            return np.full(image.shape[:2], 2.0)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    clip = VideoInput(
        video_id="mock-move", path="mock.avi", annotation_size_px=(100, 100),
        intrinsics={"fx": 100, "fy": 100, "cx": 50, "cy": 50},
        capture_poses=[{
            "frame_index": 5,
            "world_to_camera_rotation": np.eye(3).tolist(),
            "world_to_camera_translation_m": [0, 0, 0],
        }],
        views=[{
            "frame_index": 5, "track_id": "desk", "label": "desk",
            "bbox": {"x1": 25, "x2": 75, "y1": 25, "y2": 75},
            "camera_up": [0, -1, 0],
        }],
    )
    monkeypatch.setattr(moving_video, "_video_frames", lambda *args: [(5, image)])
    result = moving_video.estimate_video(clip, MockDepth())
    assert result.pose_source == "capture_poses"
    assert result.poses[0].frame_index == 5
    assert result.objects[0].dimensions["depth"] is None
    assert result.objects[0].dimensions["width"].scale_source == "pretrained_depth"
    assert "bounding_box_is_approximate" in result.objects[0].qualifiers


@pytest.mark.parametrize("override", [None, "zoedepth"])
def test_job_routes_each_video_and_reuses_loaded_models(monkeypatch, tmp_path, override):
    from ai.measurement import moving_video

    loaded = []

    class MockDepth:
        device = "cpu"

        def __init__(self, family, *, device):
            self.family = family
            self.torch = self
            loaded.append(family)

        def set_num_threads(self, count):
            pass

        def __call__(self, image, turns):
            return np.full(image.shape[:2], 2.0)

    monkeypatch.setattr(moving_video, "MetricDepth", MockDepth)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(moving_video, "_video_frames", lambda *args: [(0, image)])
    (tmp_path / "clip.avi").write_bytes(b"mock video bytes")
    clip = {
        "path": "clip.avi", "annotation_size_px": [100, 100],
        "views": [{"frame_index": 0, "track_id": "box", "label": "box",
                   "bbox": {"x1": 20, "y1": 20, "x2": 70, "y2": 70}, "camera_up": [0, -1, 0]}],
    }
    capture = {
        "intrinsics": {"fx": 100, "fy": 100, "cx": 50, "cy": 50},
        "capture_poses": [{
            "frame_index": 0, "world_to_camera_rotation": np.eye(3).tolist(),
            "world_to_camera_translation_m": [0, 0, 0],
        }],
    }
    job = moving_video.MeasurementJob.model_validate({"videos": [
        dict(clip, video_id="first"),
        dict(clip | capture, video_id="arkit"),
        dict(clip, video_id="last"),
    ]})
    result = moving_video.estimate_job(
        job, model_override=override, device="cpu", base_directory=tmp_path
    )
    assert [v.video_id for v in result.videos] == ["first", "arkit", "last"]
    expected = [override] * 3 if override else ["depth-pro", "depth-anything-v2", "depth-pro"]
    assert [v.depth_model_family for v in result.videos] == expected
    assert loaded == list(dict.fromkeys(expected))
    assert len({v.source_sha256 for v in result.videos}) == 1
    assert not job.videos[0].path.is_absolute()


def _render_two_planes(offset_x, texture, focal=300.0, size=(320, 240)):
    """Image and depth of a textured step (Z=2 m left, Z=3 m right) from x = offset_x."""
    width, height = size
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    cx, cy = width / 2, height / 2
    near_x = (u - cx) * 2.0 / focal + offset_x
    far_x = (u - cx) * 3.0 / focal + offset_x
    use_near = near_x < 0
    depth = np.where(use_near, 2.0, 3.0).astype(np.float32)
    world_x = np.where(use_near, near_x, far_x)
    world_y = (v - cy) * depth / focal
    scale = texture[0].shape[1] / 6.0  # texture covers x, y in [-3, 3] m
    map_x = ((world_x + 3.0) * scale).astype(np.float32)
    map_y = ((world_y + 3.0) * scale).astype(np.float32)
    plane = np.where(use_near, 0, 1).astype(np.uint8)
    image = np.empty((height, width, 3), np.uint8)
    for index in (0, 1):
        warped = cv2.remap(texture[index], map_x, map_y, cv2.INTER_LINEAR)
        image[plane == index] = warped[plane == index]
    camera = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
    return image, depth, camera


@pytest.mark.parametrize("history, components", [(3, 2), (6, 1)])
def test_longer_history_bridges_unmatched_frames(history, components):
    rng = np.random.default_rng(4)
    texture = [cv2.GaussianBlur(rng.integers(0, 255, (1200, 1200, 3), dtype=np.uint8),
                                (0, 0), 1.2) for _ in range(2)]
    tracker = CameraTracker(None, history=history)
    poses = []
    for index, offset in enumerate([0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24]):
        image, depth, camera = _render_two_planes(offset, texture)
        if index in (2, 3, 4):
            image = np.zeros_like(image)  # frames that match nothing, e.g. motion blur
        poses.append(tracker.add(index, image, depth, camera))
    textured = [poses[i] for i in (0, 1, 5, 6)]
    assert len({pose.component for pose in textured}) == components
    if components == 1:
        moved = poses[5].center - poses[0].center
        assert np.allclose(moved, [0.20, 0, 0], atol=0.02)
