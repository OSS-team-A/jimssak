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
