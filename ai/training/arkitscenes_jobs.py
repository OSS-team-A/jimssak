"""Build measurement jobs from ARKitScenes development scans for size evaluation.

Each selected furniture box becomes one replayed capture in the iPhone recorder format
(``capture.avi`` + ``camera.json`` + ``observations.json``). ``ai.measurement.arkit_capture``
then turns it into two videos of the same frames: ``<id>`` with the recorded ARKit poses and
``<id>-rgb`` with no poses, so both paths are scored on identical input.

Masks are the pixels whose LiDAR point lies inside the annotated 3D box: a stand-in for a
recognition mask, so the scores isolate the size estimator from detection quality. (The box
outline alone also covers the floor and walls behind the object.) Frames where the box is
hidden behind nearer surfaces (a door, another sofa) are not used.

Usage (repository root, needs network access to Apple's ARKitScenes server)::

    python -m ai.training.arkitscenes_jobs experiments/size-estimation/arkitscenes-split.json \
        --metadata outputs/arkit-depth-fix/metadata.csv --scenes 25 \
        --cache outputs/arkitscenes-cache --output outputs/dev-eval
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from ai.measurement.arkit_capture import ARCaptureMetadata, CaptureObservations, make_job
from ai.training.arkitscenes_split import (
    BASE,
    check_jobs,
    furniture_objects,
    load_split,
    read_metadata,
)

CORNERS = np.array(list(product([-0.5, 0.5], repeat=3)))
# ARKit camera axes (x right, y up, z back) from OpenCV camera axes, and ARKit's
# gravity-aligned world (y up) from the ARKitScenes world (z up).
CV_TO_ARKIT_CAMERA = np.diag([1.0, -1.0, -1.0])
ARKIT_WORLD_FROM_SCENE = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
ANNOTATION_SCALE = 2
MIN_MASK_PIXELS = 150  # native 256x192 pixels
STEP = 262144


@dataclass
class Frame:
    stamp: str
    rotation: np.ndarray  # world-to-camera, OpenCV axes
    translation: np.ndarray
    intrinsics: np.ndarray  # 3x3, native pixels
    size: tuple[int, int]  # native width, height


class RemoteZip(io.RawIOBase):
    """Read members of a remote zip with HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos, self.cache = url, 0, {}
        with self._request("-65557") as response:
            self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
            self.tail = response.read()

    def _request(self, interval: str):
        request = urllib.request.Request(self.url, headers={"Range": "bytes=" + interval})
        response = urllib.request.urlopen(request, timeout=30)
        if response.status != 206:
            response.close()
            raise OSError("Server did not return a partial response")
        return response

    def read(self, count=-1):
        stop = self.size if count < 0 else min(self.size, self.pos + count)
        chunks = []
        while self.pos < stop:
            tail_start = self.size - len(self.tail)
            if self.pos >= tail_start:
                chunks.append(self.tail[self.pos - tail_start : stop - tail_start])
                self.pos = stop
                continue
            block = self.pos // STEP
            first, last = block * STEP, min(self.size, (block + 1) * STEP) - 1
            if block not in self.cache:
                with self._request(f"{first}-{last}") as response:
                    self.cache[block] = response.read()
            end = min(stop, last + 1)
            chunks.append(self.cache[block][self.pos - first : end - first])
            self.pos = end
        return b"".join(chunks)

    def seek(self, offset, whence=0):
        self.pos = {0: 0, 1: self.pos, 2: self.size}[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True


class SceneSource:
    """Raw files of one scan, read from a local folder and fetched when missing."""

    def __init__(self, cache: Path, scan: str, fold: str = "Validation", download: bool = True):
        self.root = cache / scan
        self.scan, self.fold, self.download = scan, fold, download
        self._zip = None
        self._names = None
        self._depth_zip = None
        self._depth_names = None

    def file(self, name: str) -> Path:
        path = self.root / name
        if not path.exists():
            if not self.download:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{BASE}/{self.fold}/{self.scan}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            path.with_suffix(path.suffix + ".partial").write_bytes(data)
            path.with_suffix(path.suffix + ".partial").rename(path)
        return path

    def image(self, stamp: str) -> np.ndarray:
        path = self.root / "frames" / f"{stamp}.png"
        if not path.exists():
            if not self.download:
                raise FileNotFoundError(path)
            if self._zip is None:
                self._zip = zipfile.ZipFile(
                    RemoteZip(f"{BASE}/{self.fold}/{self.scan}/lowres_wide.zip")
                )
                self._names = {
                    Path(n).stem.rsplit("_", 1)[1]: n
                    for n in self._zip.namelist() if re.search(r"_\d+\.\d+\.png$", n)
                }
            key = next((k for k in _near(stamp) if k in self._names), None)
            if key is None:
                raise KeyError(f"{self.scan}: no frame near {stamp}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._zip.read(self._names[key]))
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read {path}")
        return image


    def depth(self, stamp: str, max_gap_s: float = 0.02) -> np.ndarray | None:
        """LiDAR depth in meters for the frame at ``stamp`` (0 where missing), or None."""
        path = self.root / "depth" / f"{stamp}.png"
        if not path.exists():
            if not self.download:
                return None
            if self._depth_zip is None:
                self._depth_zip = zipfile.ZipFile(
                    RemoteZip(f"{BASE}/{self.fold}/{self.scan}/lowres_depth.zip")
                )
                self._depth_names = {
                    float(Path(n).stem.rsplit("_", 1)[1]): n
                    for n in self._depth_zip.namelist() if re.search(r"_\d+\.\d+\.png$", n)
                }
            if not self._depth_names:
                return None
            nearest = min(self._depth_names, key=lambda t: abs(t - float(stamp)))
            if abs(nearest - float(stamp)) > max_gap_s:
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._depth_zip.read(self._depth_names[nearest]))
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        return raw.astype(np.float32) / 1000.0


def _near(stamp: str) -> list[str]:
    value = float(stamp)
    return [stamp, f"{value - 0.001:.3f}", f"{value + 0.001:.3f}"]


def load_frames(source: SceneSource, max_gap_s: float = 0.012) -> list[Frame]:
    """Pair each intrinsics file with the nearest trajectory row, in time order."""
    with zipfile.ZipFile(source.file("lowres_wide_intrinsics.zip")) as archive:
        cameras = {
            Path(name).stem.rsplit("_", 1)[1]: np.array(archive.read(name).decode().split(),
                                                       dtype=float)
            for name in archive.namelist() if name.endswith(".pincam")
        }
    trajectory = np.loadtxt(source.file("lowres_wide.traj"))
    times = trajectory[:, 0]
    frames = []
    for stamp in sorted(cameras, key=float):
        row = trajectory[np.argmin(np.abs(times - float(stamp)))]
        if abs(row[0] - float(stamp)) > max_gap_s:
            continue
        width, height, fx, fy, cx, cy = cameras[stamp]
        frames.append(Frame(
            stamp, cv2.Rodrigues(row[1:4])[0], row[4:7].copy(),
            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), (int(width), int(height)),
        ))
    return frames


def camera_to_world(frame: Frame) -> np.ndarray:
    """ARKit-style camera-to-world transform, as the iPhone recorder writes it."""
    transform = np.eye(4)
    transform[:3, :3] = ARKIT_WORLD_FROM_SCENE @ frame.rotation.T @ CV_TO_ARKIT_CAMERA
    transform[:3, 3] = ARKIT_WORLD_FROM_SCENE @ (-frame.rotation.T @ frame.translation)
    return transform


def upright_turns(frame: Frame) -> int:
    """Counter-clockwise quarter turns that put the scene's up direction at the image top."""
    x, y = frame.rotation[:2, 2]
    if abs(y) >= abs(x):
        return 0 if y < 0 else 2
    return 1 if x > 0 else 3


def box_corners(item: dict) -> np.ndarray:
    box = item["segments"]["obbAligned"]
    axes = np.reshape(box["normalizedAxes"], (3, 3))
    return (CORNERS * np.asarray(box["axesLengths"])) @ axes + np.asarray(box["centroid"])


def box_frame(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center, axis rows and full lengths of an annotation's gravity-aligned box (meters)."""
    box = item["segments"]["obbAligned"]
    return (np.asarray(box["centroid"], float), np.reshape(box["normalizedAxes"], (3, 3)),
            np.asarray(box["axesLengths"], float))


def object_mask(item: dict, frame: Frame, depth: np.ndarray | None,
                margin_m: float = 0.04) -> np.ndarray | None:
    """Native-resolution mask of pixels whose LiDAR point lies inside the 3D box.

    The projected box outline also covers floor, walls and the space between legs behind
    the object; those pixels would add far background points to the measurement.
    """
    if depth is None:
        return None
    width, height = frame.size
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    center, axes, lengths = box_frame(item)
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    rays = np.linalg.inv(frame.intrinsics) @ np.vstack([u + 0.0, v + 0.0, np.ones_like(z)])
    camera = (rays * z).T
    world = (camera - frame.translation) @ frame.rotation
    local = (world - center) @ axes.T
    inside = (np.abs(local) <= lengths / 2 + margin_m).all(axis=1)
    mask = np.zeros((height, width), np.uint8)
    mask[v[inside], u[inside]] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def encode_rle(mask: np.ndarray) -> list[int]:
    """Row-major run lengths starting with background, as ObjectView.mask_rle expects."""
    flat = mask.ravel().astype(np.uint8)
    edges = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.r_[0, edges, flat.size]).tolist()
    return ([0] + runs) if flat[0] else runs


def annotation_mask(mask: np.ndarray, turns: int) -> np.ndarray:
    upright = np.rot90(mask, turns)
    return np.repeat(np.repeat(upright, ANNOTATION_SCALE, axis=0), ANNOTATION_SCALE, axis=1)


def project_box(corners: np.ndarray, frame: Frame, margin: float = 4.0) -> np.ndarray | None:
    """Projected corners in native pixels when the whole box is in front and in view."""
    points = corners @ frame.rotation.T + frame.translation
    if (points[:, 2] < 0.15).any():
        return None
    pixels = points @ frame.intrinsics.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    width, height = frame.size
    if (pixels.min(axis=0) < margin).any() or (pixels.max(axis=0) > [width - margin,
                                                                        height - margin]).any():
        return None
    if np.ptp(pixels[:, 0]) < 24 or np.ptp(pixels[:, 1]) < 24:
        return None
    return pixels


def visible_fraction(corners: np.ndarray, frame: Frame, depth: np.ndarray | None,
                     margin_m: float = 0.10) -> float | None:
    """Share of the box outline not hidden behind something nearer, from LiDAR depth.

    A pixel is occluded when the sensor sees a surface nearer than the box's nearest corner
    by more than ``margin_m``. None when there is no usable depth for the frame.
    """
    pixels = project_box(corners, frame)
    if pixels is None or depth is None:
        return None
    width, height = frame.size
    scale = np.array([depth.shape[1] / width, depth.shape[0] / height])
    mask = np.zeros(depth.shape, np.uint8)
    hull = cv2.convexHull((pixels * scale).astype(np.float32)).round().astype(np.int32)
    cv2.fillPoly(mask, [hull], 1)
    values = depth[(mask == 1) & (depth > 0)]
    if values.size < 30:
        return None
    nearest = float((corners @ frame.rotation.T + frame.translation)[:, 2].min())
    return float(np.mean(values >= nearest - margin_m))


def rotate_points(pixels: np.ndarray, turns: int, size: tuple[int, int]) -> np.ndarray:
    """Native pixel coordinates in the image after ``np.rot90(image, turns)``."""
    width, height = size
    x, y = pixels[:, 0], pixels[:, 1]
    return {
        0: np.c_[x, y],
        1: np.c_[y, width - 1 - x],
        2: np.c_[width - 1 - x, height - 1 - y],
        3: np.c_[height - 1 - y, x],
    }[turns]


def hull_polygon(pixels: np.ndarray, turns: int, size: tuple[int, int]) -> list[list[float]]:
    upright = rotate_points(pixels, turns, size)
    scaled = (upright + 0.5) * ANNOTATION_SCALE - 0.5
    hull = cv2.convexHull(scaled.astype(np.float32)).reshape(-1, 2)
    width, height = (size[1], size[0]) if turns % 2 else size
    limit = np.array([width, height]) * ANNOTATION_SCALE - 1
    return np.clip(hull, 0, limit).round(2).tolist()


def _azimuths(frames: list[Frame], indices: list[int], center: np.ndarray) -> np.ndarray:
    """Horizontal direction (degrees) from the box center to each camera, unwrapped in time."""
    angles = []
    for i in indices:
        camera = -frames[i].rotation.T @ frames[i].translation
        offset = camera - center
        angles.append(np.degrees(np.arctan2(offset[1], offset[0])))
    return np.degrees(np.unwrap(np.radians(angles)))


def plan_capture(frames: list[Frame], corners: np.ndarray, *, views: int = 6,
                 step_s: float = 0.2, max_frames: int = 90, window_s: float = 20.0,
                 min_spread_deg: float = 15.0, max_checks: int = 60,
                 visibility=None, min_visible: float = 0.7) -> tuple[list[int], list[int]] | None:
    """Pick observed frames that see the box from the widest range of directions.

    Within any ``window_s`` seconds of video, the window whose fully-in-view frames sweep the
    widest horizontal angle around the box wins; views are then spread evenly by angle.
    Frames a few centimeters apart leave the box's extent along the view direction
    unconstrained, so a box seen over less than ``min_spread_deg`` is skipped.

    Returns (frame indices written to the replay, indices among them that carry a mask).
    The in-between frames let RGB camera tracking chain the observed ones as a phone video
    would. With ``visibility`` (frame index -> visible share or None), observed frames whose
    box is hidden behind nearer surfaces are skipped; a frame without sensor depth is not
    trusted either. At most ``max_checks`` in-view frames, spread over the video, are checked.
    """
    visible = [i for i, frame in enumerate(frames) if project_box(corners, frame) is not None]
    if len(visible) < 3:
        return None
    if visibility is not None:
        # Drop hidden views first, so the widest sweep is chosen among frames that see the box.
        sample = sorted({visible[j] for j in np.linspace(0, len(visible) - 1,
                                                         min(max_checks, len(visible)))
                         .round().astype(int)})
        visible = [i for i in sample if (visibility(i) or 0.0) >= min_visible]
        if len(visible) < 3:
            return None
    center = corners.mean(axis=0)
    times = np.array([float(frames[i].stamp) for i in visible])
    azimuth = _azimuths(frames, visible, center)
    best = None
    for a in range(len(visible)):
        b = int(np.searchsorted(times, times[a] + window_s, side="right"))
        if b - a < 3:
            continue
        spread = float(np.ptp(azimuth[a:b]))
        key = (spread, b - a)
        if best is None or key > best[0]:
            best = (key, a, b)
    if best is None or best[0][0] < min_spread_deg:
        return None
    _, a, b = best
    window = visible[a:b]
    angles = azimuth[a:b]
    targets = np.linspace(angles.min(), angles.max(), min(views, len(window)))
    observed = sorted({window[int(np.argmin(np.abs(angles - t)))] for t in targets})
    if len(observed) < 3:
        return None
    written, last = [], None
    for i in range(observed[0], observed[-1] + 1):
        stamp = float(frames[i].stamp)
        if i in observed or last is None or stamp - last >= step_s - 1e-6:
            written.append(i)
            last = stamp
    if len(written) > max_frames:
        keep = set(observed)
        others = [i for i in written if i not in keep]
        stride = int(np.ceil(len(others) / max(1, max_frames - len(keep))))
        written = sorted(keep | set(others[::stride]))
    if len(set(frames[i].size for i in written)) != 1:
        return None
    return written, observed


def write_capture(folder: Path, video_id: str, frames: list[Frame], written: list[int],
                  observed: list[int], source: SceneSource, track_id: str,
                  label: str, corners: np.ndarray,
                  mask_for=None) -> tuple[ARCaptureMetadata, CaptureObservations]:
    """Write the replay. ``mask_for(index)`` gives a native object mask; without it, or when
    it returns None, the view uses the projected box outline."""
    folder.mkdir(parents=True, exist_ok=True)
    size = frames[written[0]].size
    turns = upright_turns(frames[observed[len(observed) // 2]])
    writer = cv2.VideoWriter(str(folder / "capture.avi"), cv2.VideoWriter_fourcc(*"FFV1"),
                             5, size)
    if not writer.isOpened():
        raise RuntimeError("Cannot open the FFV1 video writer")
    sidecar_frames, views = [], []
    start = float(frames[written[0]].stamp)
    try:
        for index, i in enumerate(written):
            frame = frames[i]
            image = source.image(frame.stamp)
            if image.shape[1::-1] != size:
                raise ValueError(f"{video_id}: frame {frame.stamp} has another size")
            writer.write(image)
            sidecar_frames.append({
                "frame_index": index,
                "timestamp_us": round((float(frame.stamp) - start) * 1e6),
                "camera_to_world": camera_to_world(frame).tolist(),
                "intrinsics": frame.intrinsics.tolist(),
            })
            if i in observed:
                native = mask_for(i) if mask_for is not None else None
                view = {"frame_index": index, "track_id": track_id, "label": label}
                if native is not None and native.any():
                    view["mask_rle"] = encode_rle(annotation_mask(native, turns))
                else:
                    view["mask_polygon_px"] = hull_polygon(project_box(corners, frame), turns, size)
                views.append(view)
    finally:
        writer.release()
    rotated = (size[1], size[0]) if turns % 2 else size
    metadata = ARCaptureMetadata.model_validate({
        "schema_version": 1, "video_file": "capture.avi", "video_id": video_id,
        "image_size_px": size, "pixel_rotation_ccw": turns, "frames": sidecar_frames,
    })
    observations = CaptureObservations.model_validate({
        "annotation_size_px": [n * ANNOTATION_SCALE for n in rotated],
        "views": views, "pose_stride": 1,
    })
    (folder / "camera.json").write_text(metadata.model_dump_json(indent=1))
    (folder / "observations.json").write_text(observations.model_dump_json(indent=1))
    return metadata, observations


def build_scan(source: SceneSource, output: Path, *, objects_per_scan: int = 2,
               views: int = 6) -> tuple[list, list[dict]]:
    """Replay up to ``objects_per_scan`` fully visible furniture boxes of one scan."""
    annotation = json.loads(source.file(f"{source.scan}_3dod_annotation.json").read_text())
    truth = {row["uid"]: row for row in furniture_objects(annotation, source.scan)}
    if not truth:
        return [], []
    items = {item["uid"]: item for item in annotation["data"] if item["uid"] in truth}
    frames = load_frames(source)
    plans = []
    for uid, item in items.items():
        corners = box_corners(item)

        def visibility(index, corners=corners, item=item):
            frame = frames[index]
            depth = source.depth(frame.stamp)
            share = visible_fraction(corners, frame, depth)
            mask = object_mask(item, frame, depth)
            if share is None or mask is None or mask.sum() < MIN_MASK_PIXELS:
                return None
            return share

        plan = plan_capture(frames, corners, views=views, visibility=visibility)
        if plan is not None:
            plans.append((len(plan[1]), uid, plan))
    plans.sort(key=lambda row: (-row[0], row[1]))
    videos, tracks = [], []
    for _, uid, (written, observed) in plans[:objects_per_scan]:
        video_id = f"{source.scan}-{uid[:6]}"
        folder = output / video_id
        item = items[uid]
        metadata, observations = write_capture(
            folder, video_id, frames, written, observed, source, uid,
            truth[uid]["label"], box_corners(item),
            mask_for=lambda i, item=item: object_mask(item, frames[i],
                                                      source.depth(frames[i].stamp)),
        )
        job = make_job(metadata, observations, folder / "capture.avi", compare=True)
        videos.extend(job.videos)
        reference = {k: truth[uid][f"{k}_cm"] for k in ("width", "depth", "height")}
        for clip in job.videos:
            tracks.append({"video_id": clip.video_id, "track_id": uid,
                           "label": truth[uid]["label"], "dimensions_cm": reference})
    return videos, tracks


def relative_to_manifest(videos: list, manifest_folder: Path) -> list:
    """Video paths relative to the folder of job.json, which moving_video resolves against."""
    return [
        clip.model_copy(update={"path": Path(os.path.relpath(clip.path, manifest_folder))})
        if not Path(clip.path).is_absolute() else clip
        for clip in videos
    ]


def choose_scans(split: dict, metadata: dict, count: int, seed: int) -> list[str]:
    """Development-side Validation scans that no earlier experiment touched."""
    holdout = set(split["holdout_scans"])
    used = set(split["used_scans"])
    pool = sorted(
        scan for scan, row in metadata.items()
        if row["fold"] == "Validation" and row.get("is_in_threedod", "True") == "True"
        and scan not in holdout and scan not in used
    )
    return random.Random(seed).sample(pool, min(count, len(pool)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("split", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--scan", action="append", default=[],
                        help="build these scans instead of a random choice")
    parser.add_argument("--objects-per-scan", type=int, default=2)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="use cached files only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    split = load_split(args.split)
    metadata = read_metadata(args.metadata)
    scans = args.scan or choose_scans(split, metadata, args.scenes, args.seed)
    holdout = sorted(set(scans) & set(split["holdout_scans"]))
    if holdout:
        raise SystemExit(f"Refusing final-evaluation scans: {holdout}")
    videos, tracks, report = [], [], []
    for number, scan in enumerate(scans, 1):
        source = SceneSource(args.cache, scan, metadata[scan]["fold"], not args.offline)
        try:
            new_videos, new_tracks = build_scan(
                source, args.output, objects_per_scan=args.objects_per_scan, views=args.views
            )
        except Exception as error:  # report each failed scan and keep going
            report.append({"scan": scan, "error": repr(error)})
            print(f"[{number}/{len(scans)}] {scan}: {error!r}", flush=True)
            continue
        videos.extend(new_videos)
        tracks.extend(new_tracks)
        report.append({"scan": scan, "videos": len(new_videos) // 2})
        print(f"[{number}/{len(scans)}] {scan}: {len(new_videos) // 2} objects", flush=True)
    if not videos:
        raise SystemExit("No scan produced a measurable object")
    args.output.mkdir(parents=True, exist_ok=True)
    videos = relative_to_manifest(videos, args.output)
    job_path = args.output / "job.json"
    job_path.write_text(json.dumps(
        {"videos": [json.loads(v.model_dump_json(exclude_none=True)) for v in videos]},
        indent=1,
    ))
    (args.output / "truth.json").write_text(json.dumps({"tracks": [
        {k: t[k] for k in ("video_id", "track_id", "dimensions_cm")} for t in tracks
    ]}, indent=2))
    (args.output / "build-report.json").write_text(json.dumps(report, indent=2))
    problems = check_jobs(split, [job_path])
    if problems:
        raise SystemExit("\n".join(problems))
    print(json.dumps({"scans": len(scans), "objects": len(videos) // 2,
                      "videos": len(videos), "failed_scans": sum("error" in r for r in report)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
