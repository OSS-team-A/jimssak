import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from ai.measurement import multiview  # noqa: E402


def test_aligned_depth_requires_local_and_multiview_support():
    camera = np.eye(3, 4)
    full = np.ones((5, 5), dtype=np.uint8)
    limited = full.copy()
    limited[:, 2:] = 0
    geometry = np.tile([1.0, 1.0, 1.0], (12, 1))
    geometry[0, 0] = 3.0
    aligned = geometry.copy()
    aligned[1, 2] += 1.0

    result = multiview._consistent_depth(
        geometry, aligned, camera, [(camera, mask) for mask in (full, full, limited)]
    )
    assert not result[0]  # Falls outside one of three masks.
    assert not result[1]  # In all masks, but projects substantially behind the surface.
    assert result[2:].all()

    result = multiview._consistent_depth(
        geometry, aligned, camera, [(camera, mask) for mask in (full, full, full, limited)]
    )
    assert result[0]  # One disagreeing mask is tolerated when three support the point.
    assert not result[1]


def test_recovers_a_filtered_edge_only_if_two_masks_support_it():
    camera = np.eye(3, 4)
    x, y = np.meshgrid(np.arange(5, 31), np.arange(5, 16))
    geometry = np.c_[x.ravel(), y.ravel(), np.ones(x.size)]
    narrow = geometry[geometry[:, 0] <= 20]

    full = np.zeros((24, 36), dtype=np.uint8)
    full[5:16, 5:31] = 1
    clipped = full.copy()
    clipped[:, 21:] = 0
    inconsistent = full.copy()
    inconsistent[5:10, 31:36] = 1

    recovered = multiview._recover_silhouette_edges(
        narrow, geometry, [(camera, mask, None) for mask in (full, full, clipped)]
    )
    assert len(recovered)
    assert recovered[:, 0].min() == 23
    assert recovered[:, 0].max() == 30

    for second in (clipped, inconsistent):
        recovered = multiview._recover_silhouette_edges(
            narrow, geometry, [(camera, mask, None) for mask in (full, second, clipped)]
        )
        assert not len(recovered)


def test_geometric_depth_needs_lateral_views_with_agreeing_short_spans():
    x, y = np.meshgrid(np.linspace(-.5, .5, 45), np.linspace(-.18, .18, 24))
    surface = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    spread = [(-1, -2, 1), (0, -2, 1), (1, -2, 1)]
    forward = [(0, -3, 1), (0, -2, 1), (0, -1, 1)]
    assert multiview._supported_geometric_depth([surface] * 3, spread) == pytest.approx(36)
    assert multiview._supported_geometric_depth([surface] * 3, forward) is None

    inconsistent = surface * [1, 2, 1]
    assert multiview._supported_geometric_depth([surface, surface, inconsistent], spread) is None


def test_measured_camera_height_keeps_geometric_contacts(monkeypatch):
    mask = np.zeros((36, 36), np.uint8)
    mask[5:30, 6:30] = 1
    y, x = np.where(mask)
    pixels = np.c_[x, y]
    geometry = np.c_[x * .03, y * .025, (29 - y) * .04]
    learned = geometry * [1, 1, .8]
    valid = np.ones(len(pixels), bool)
    frames = [
        multiview.MaskView(i, mask, np.c_[np.eye(3), [-x, 2, 0]])
        for i, x in enumerate((-.6, 0, .6))
    ]
    monkeypatch.setattr(multiview, "visual_hull", lambda *args: geometry)
    paired = (pixels, geometry, (learned, 1, valid))
    monkeypatch.setattr(multiview, "same_rays", lambda *args: paired)
    monkeypatch.setattr(multiview, "_consistent_depth", lambda *args: valid)
    monkeypatch.setattr(multiview, "_recover_silhouette_edges", lambda *args: geometry[:0])

    before = multiview.measure_track("cabinet", "cabinet", frames)
    after = multiview.measure_track("cabinet", "cabinet", frames, metric_capture_poses=True)
    assert after.width_cm == before.width_cm
    assert after.depth_cm == before.depth_cm
    assert before.height_cm == pytest.approx(.8 * after.height_cm, abs=.02)
    assert after.height_frame_ids == [0, 1, 2]


def test_geometric_depth_can_use_two_elevated_views_without_lateral_cover():
    x, y = np.meshgrid(np.linspace(-.5, .5, 45), np.linspace(-.18, .18, 24))
    surface = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    shallow = surface * [1, .5, 1]
    surfaces = [shallow, surface, surface]
    centers = [(0, -3, 1), (0, -2, 2), (0, -1.5, 2)]
    assert multiview._supported_geometric_depth(surfaces, centers) == pytest.approx(36)

    centers[-1] = (0, -2, .5)
    assert multiview._supported_geometric_depth(surfaces, centers) is None


def test_geometric_depth_rejects_extent_only_supported_below_upper_surface():
    x, y = np.meshgrid(np.linspace(-.5, .5, 45), np.linspace(-.36, .36, 24))
    bottom = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    narrow_top = bottom * [1, .6, 1] + [0, 0, .6]
    wide_top = bottom + [0, 0, .6]
    centers = [(-1, -2, 1), (0, -2, 1), (1, -2, 1)]
    table = np.vstack([bottom, narrow_top])
    solid = np.vstack([bottom, wide_top])
    assert multiview._supported_geometric_depth([table] * 3, centers) is None
    assert multiview._supported_geometric_depth([solid] * 3, centers) == pytest.approx(72)
