import pytest

np = pytest.importorskip("numpy")

from ai.measurement.depth_alignment import align_depth_to_geometry  # noqa: E402


def test_alignment_recovers_scale_along_camera_rays():
    center = np.array([1.5, -0.4, 0.7])
    intrinsic = np.array([[400.0, 0, 150], [0, 420.0, 100], [0, 0, 1]])
    camera = np.column_stack((intrinsic, -intrinsic @ center))
    geometry = center + [[0.2, 0, 1], [0.1, 0.3, 2], [-0.4, 0, 1.5], [0.4, 0.3, 2]]
    prediction = center + 2 * (geometry - center)
    prediction[-1] = center + 8 * (geometry[-1] - center)

    aligned, scale = align_depth_to_geometry(camera, geometry, prediction)

    assert scale == pytest.approx(0.5)
    np.testing.assert_allclose(aligned[:3], geometry[:3])
    np.testing.assert_allclose(aligned[-1] - center, 4 * (geometry[-1] - center))


def test_alignment_requires_paired_points_in_front_of_camera():
    camera = np.column_stack((np.eye(3), np.zeros(3)))
    points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, -1]])

    with pytest.raises(ValueError, match="At least three"):
        align_depth_to_geometry(camera, points, points)
