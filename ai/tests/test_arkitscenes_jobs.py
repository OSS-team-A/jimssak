import json

import cv2
import numpy as np
import pytest

from ai.measurement.arkit_capture import verify_movie
from ai.training.arkitscenes_jobs import (
    Frame,
    annotation_mask,
    camera_to_world,
    choose_scans,
    encode_rle,
    hull_polygon,
    object_mask,
    plan_capture,
    project_box,
    relative_to_manifest,
    rotate_points,
    upright_turns,
    visible_fraction,
    write_capture,
)

K = np.array([[200.0, 0, 128], [0, 200.0, 96], [0, 0, 1]])
CORNERS = np.array([[x, y, z] for x in (-0.3, 0.3) for y in (-0.2, 0.2) for z in (0, 0.7)])


def look_at(center, target=(0, 0, 0.35)):
    """World-to-camera rotation (OpenCV axes) for a camera at center looking at target."""
    forward = np.asarray(target, float) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return rotation, -rotation @ center


def orbit(count=40, radius=2.2, step_s=0.1, start_deg=-30, sweep_deg=60):
    frames = []
    for i, angle in enumerate(np.deg2rad(np.linspace(start_deg, start_deg + sweep_deg, count))):
        center = np.array([radius * np.cos(angle), radius * np.sin(angle), 1.1])
        rotation, translation = look_at(center)
        frames.append(Frame(f"{100 + i * step_s:.3f}", rotation, translation, K, (256, 192)))
    return frames


class FakeSource:
    def __init__(self):
        self.requested = []

    def image(self, stamp):
        self.requested.append(stamp)
        rng = np.random.default_rng(int(float(stamp) * 10))
        return rng.integers(0, 255, (192, 256, 3), dtype=np.uint8)


def test_camera_to_world_round_trips_to_scene_center():
    frame = orbit()[0]
    transform = camera_to_world(frame)
    rotation = transform[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3))
    assert np.isclose(np.linalg.det(rotation), 1)
    # ARKit world y-up must be the scene's z-up.
    scene_center = -frame.rotation.T @ frame.translation
    assert np.allclose(transform[:3, 3], [scene_center[0], scene_center[2], -scene_center[1]])
    assert upright_turns(frame) == 0


def test_rotate_points_matches_numpy_rot90():
    image = np.zeros((192, 256), np.uint8)
    image[10, 20] = 1
    for turns in range(4):
        y, x = np.argwhere(np.rot90(image, turns))[0]
        assert rotate_points(np.array([[20.0, 10.0]]), turns, (256, 192)).tolist() == [[x, y]]


def test_plan_keeps_observed_views_and_fills_the_gaps():
    frames = orbit()
    written, observed = plan_capture(frames, CORNERS, views=5, step_s=0.2)
    assert len(observed) == 5 and set(observed) <= set(written)
    stamps = [float(frames[i].stamp) for i in written]
    assert max(np.diff(stamps)) <= 0.2 + 1e-6
    assert written[0] == observed[0] and written[-1] == observed[-1]
    assert all(project_box(CORNERS, frames[i]) is not None for i in observed)


def test_plan_rejects_box_never_fully_in_view():
    far = CORNERS * 20
    assert plan_capture(orbit(), far) is None


def test_written_capture_imports_as_arkit_and_rgb_videos(tmp_path):
    from ai.measurement.arkit_capture import make_job

    frames = orbit()
    written, observed = plan_capture(frames, CORNERS, views=4)
    source = FakeSource()
    metadata, observations = write_capture(
        tmp_path, "scan-abc", frames, written, observed, source, "uid1", "table", CORNERS
    )
    assert source.requested == [frames[i].stamp for i in written]
    verify_movie(tmp_path / "capture.avi", metadata)
    job = make_job(metadata, observations, tmp_path / "capture.avi", compare=True)
    rgb, arkit = job.videos
    assert (rgb.video_id, arkit.video_id) == ("scan-abc-rgb", "scan-abc")
    assert not rgb.capture_poses and len(arkit.capture_poses) == len(observed)
    assert len(arkit.views) == len(observed)
    width, height = observations.annotation_size_px
    for view in arkit.views:
        polygon = np.asarray(view.mask_polygon_px)
        assert len(polygon) >= 4
        assert (polygon >= 0).all() and (polygon[:, 0] < width).all()
        assert (polygon[:, 1] < height).all()
    saved = json.loads((tmp_path / "camera.json").read_text())
    assert saved["frames"][0]["timestamp_us"] == 0


def test_capture_poses_reproject_box_into_the_mask(tmp_path):
    from ai.measurement.arkit_capture import make_job

    frames = orbit()
    written, observed = plan_capture(frames, CORNERS, views=3)
    metadata, observations = write_capture(
        tmp_path, "scan-abc", frames, written, observed, FakeSource(), "uid1", "table", CORNERS
    )
    clip = make_job(metadata, observations, tmp_path / "capture.avi").videos[0]
    width, height = observations.annotation_size_px
    for view, pose in zip(clip.views, clip.capture_poses):
        k = clip.frame_intrinsics[view.frame_index]
        # The importer works in the measurement world, whose up axis it recovers itself;
        # the box center must still land inside its own mask.
        camera = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]])
        rotation = np.asarray(pose.world_to_camera_rotation)
        translation = np.asarray(pose.world_to_camera_translation_m)
        center_scene = np.array([0, 0, 0.35])
        center_world = np.array([center_scene[0], center_scene[2], -center_scene[1]])
        point = camera @ (rotation @ center_world + translation)
        pixel = point[:2] / point[2]
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.asarray(view.mask_polygon_px).round().astype(np.int32)], 1)
        assert mask[int(pixel[1]), int(pixel[0])] == 1


def test_hull_polygon_is_scaled_to_annotation_size():
    pixels = np.array([[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]])
    polygon = np.asarray(hull_polygon(pixels, 0, (256, 192)))
    assert np.allclose(polygon.min(axis=0), [20.5, 40.5])
    assert np.allclose(polygon.max(axis=0), [100.5, 120.5])


@pytest.mark.parametrize("seed", [1, 2])
def test_choose_scans_skips_holdout_and_used(seed):
    metadata = {f"4{i:07d}": {"video_id": f"4{i:07d}", "fold": "Validation",
                              "is_in_threedod": "True"} for i in range(30)}
    metadata["49999999"] = {"video_id": "49999999", "fold": "Training", "is_in_threedod": "True"}
    split = {"holdout_scans": ["40000001", "40000002"], "used_scans": {"40000003": ["x"]}}
    chosen = choose_scans(split, metadata, 10, seed)
    assert len(chosen) == 10 and len(set(chosen)) == 10
    assert not {"40000001", "40000002", "40000003", "49999999"} & set(chosen)
    assert chosen == choose_scans(split, metadata, 10, seed)


def test_job_paths_resolve_from_the_manifest_folder(tmp_path, monkeypatch):
    from ai.measurement.arkit_capture import make_job

    monkeypatch.chdir(tmp_path)
    output = tmp_path.relative_to(tmp_path) / "outputs" / "dev-eval"
    frames = orbit()
    written, observed = plan_capture(frames, CORNERS, views=3)
    folder = output / "scan-abc"
    metadata, observations = write_capture(
        folder, "scan-abc", frames, written, observed, FakeSource(), "uid1", "table", CORNERS
    )
    videos = make_job(metadata, observations, folder / "capture.avi", compare=True).videos
    fixed = relative_to_manifest(videos, output)
    assert {str(clip.path) for clip in fixed} == {"scan-abc/capture.avi"}
    assert all((output / clip.path).exists() for clip in fixed)


def test_visible_fraction_detects_a_nearer_occluder():
    frame = orbit()[10]
    corners_cam = CORNERS @ frame.rotation.T + frame.translation
    nearest = corners_cam[:, 2].min()
    behind = np.full((192, 256), nearest + 1.0, np.float32)
    assert visible_fraction(CORNERS, frame, behind) == 1.0
    door = np.full((192, 256), nearest - 0.5, np.float32)
    assert visible_fraction(CORNERS, frame, door) == 0.0
    assert visible_fraction(CORNERS, frame, None) is None
    assert visible_fraction(CORNERS, frame, np.zeros((192, 256), np.float32)) is None


def test_plan_skips_hidden_frames_and_gives_up_when_all_are_hidden():
    frames = orbit()
    hidden = set(range(0, 20))
    written, observed = plan_capture(
        frames, CORNERS, views=4, visibility=lambda i: 0.2 if i in hidden else 0.95
    )
    assert observed and not set(observed) & hidden
    assert plan_capture(frames, CORNERS, views=4, visibility=lambda i: 0.1) is None
    assert plan_capture(frames, CORNERS, views=4, visibility=lambda i: None) is None


def box_item(lengths=(0.6, 0.4, 0.7), center=(0, 0, 0.35)):
    return {"segments": {"obbAligned": {"centroid": list(center), "axesLengths": list(lengths),
                                        "normalizedAxes": [1, 0, 0, 0, 1, 0, 0, 0, 1]}}}


def render_depth(frame, item, wall_m=4.0):
    """Sensor depth of the box's front faces in front of a far wall, by ray casting."""
    height, width = 192, 256
    v, u = np.mgrid[0:height, 0:width]
    rays = np.linalg.inv(frame.intrinsics) @ np.vstack([u.ravel(), v.ravel(), np.ones(u.size)])
    rays = frame.rotation.T @ rays  # world directions
    origin = -frame.rotation.T @ frame.translation
    center = np.asarray(item["segments"]["obbAligned"]["centroid"])
    half = np.asarray(item["segments"]["obbAligned"]["axesLengths"]) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (center - half - origin[:, None].T).T / rays
        t2 = (center + half - origin[:, None].T).T / rays
    near = np.nanmax(np.minimum(t1, t2), axis=0)
    far = np.nanmin(np.maximum(t1, t2), axis=0)
    hit = (near <= far) & (near > 0)
    forward = frame.rotation[2] @ rays  # camera z per unit ray length
    depth = np.where(hit, near * forward, wall_m)
    return depth.reshape(height, width).astype(np.float32)


def test_object_mask_keeps_the_box_and_drops_the_background():
    frame = orbit()[10]
    item = box_item()
    depth = render_depth(frame, item)
    mask = object_mask(item, frame, depth)
    hull = np.zeros((192, 256), np.uint8)
    pixels = project_box(CORNERS, frame)
    cv2.fillPoly(hull, [cv2.convexHull(pixels.astype(np.float32)).round().astype(np.int32)], 1)
    on_box = depth < 3.9
    assert mask[on_box & (hull == 1)].mean() > 0.95
    assert mask[~on_box].sum() == 0
    assert object_mask(item, frame, None) is None


def test_rle_round_trips_through_the_annotation_mask():
    mask = np.zeros((4, 5), np.uint8)
    mask[1:3, 2:4] = 1
    big = annotation_mask(mask, 1)
    assert big.shape == (10, 8)
    runs = encode_rle(big)
    decoded = np.repeat(np.arange(len(runs)) % 2, runs).reshape(big.shape)
    assert (decoded == big).all()
    full = np.ones((2, 2), np.uint8)
    assert encode_rle(full) == [0, 4]


def test_written_views_use_the_object_mask_when_given(tmp_path):
    frames = orbit()
    written, observed = plan_capture(frames, CORNERS, views=3)
    native = np.zeros((192, 256), np.uint8)
    native[80:120, 100:150] = 1
    _, observations = write_capture(
        tmp_path, "scan-abc", frames, written, observed, FakeSource(), "uid1", "table", CORNERS,
        mask_for=lambda i: native,
    )
    width, height = observations.annotation_size_px
    for view in observations.views:
        assert view.mask_polygon_px is None and sum(view.mask_rle) == width * height


def test_plan_prefers_the_widest_sweep_and_skips_narrow_ones():
    still = orbit(count=40, start_deg=-5, sweep_deg=6)  # barely moves: 6 degrees
    sweep = orbit(count=40, start_deg=10, sweep_deg=70)
    frames = still + [
        Frame(f"{200 + i * 0.1:.3f}", f.rotation, f.translation, f.intrinsics, f.size)
        for i, f in enumerate(sweep)
    ]
    written, observed = plan_capture(frames, CORNERS, views=5)
    assert min(observed) >= len(still)
    centers = [-frames[i].rotation.T @ frames[i].translation for i in observed]
    angles = np.degrees(np.unwrap([np.arctan2(c[1], c[0]) for c in centers]))
    assert np.ptp(angles) > 60
    assert np.diff(angles).max() < 2 * np.diff(angles).min()  # roughly even by angle
    assert plan_capture(still, CORNERS, views=5) is None
