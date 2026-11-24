import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from ai.measurement.depth_geometry import table_dimensions  # noqa: E402
from ai.measurement.moving_camera import relative_pose  # noqa: E402
from ai.measurement.moving_contract import ObjectView, VideoInput  # noqa: E402
from ai.measurement.moving_video import _calibrate  # noqa: E402


def projection(points, camera):
    projected = points @ camera.T
    return projected[:, :2] / projected[:, 2:]


def test_translation_and_rotation_only_are_distinguished():
    rng = np.random.default_rng(553)
    points = rng.uniform([-1, -0.7, 2.0], [1, 0.7, 4.0], size=(80, 3))
    camera = np.array([[440., 0, 320], [0, 440., 240], [0, 0, 1]])
    rotation = cv2.Rodrigues(np.array([0.03, -0.08, 0.01]))[0]
    source = projection(points, camera)
    for movement in [np.array([0.30, 0, 0.05]), np.zeros(3)]:
        target = projection(points @ rotation.T + movement, camera)
        result = relative_pose(source, target, points[:, 2], camera)
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
        "top_width": {"cm": 59}, "top_depth": {"cm": 36}, "overall_width": None,
    }
