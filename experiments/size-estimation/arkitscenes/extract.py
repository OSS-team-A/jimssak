"""Extract selected ARKitScenes RGB frames with HTTP ranges and project furniture boxes."""

import io
import json
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from collect import BASE, ROOT, SCANS

CLASSES = ("cabinet", "shelf", "table")
CORNERS = np.array(list(product([-0.5, 0.5], repeat=3)))
STEP = 262144


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        self.cache = {}
        response = self.request("-65557")
        with response:
            self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
            self.tail = response.read()
        self.transferred = len(self.tail)

    def request(self, interval):
        request = urllib.request.Request(
            self.url + "?jimssak_range=" + interval, headers={"Range": "bytes=" + interval}
        )
        response = urllib.request.urlopen(request, timeout=18)
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
                with self.request(f"{first}-{last}") as response:
                    expected = f"bytes {first}-{last}/{self.size}"
                    if response.headers["Content-Range"] != expected:
                        raise OSError("Unexpected byte range")
                    self.cache[block] = response.read()
                    self.transferred += len(self.cache[block])
            end = min(stop, last + 1)
            chunks.append(self.cache[block][self.pos - first : end - first])
            self.pos = end
        return b"".join(chunks)

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True


def furniture(path):
    rows = json.loads(path.read_text())["data"]
    boxes = []
    for row in rows:
        if row["label"] not in CLASSES:
            continue
        box = row["segments"]["obbAligned"]
        dims = np.array(box["axesLengths"])
        axes = np.reshape(box["normalizedAxes"], (3, 3))
        if dims.min() < 0.15 or dims.max() > 3.5 or abs(axes[2, 2]) < 0.9:
            continue
        corners = (CORNERS * dims) @ axes + box["centroid"]
        span, short = sorted(dims[:2], reverse=True)
        boxes.append((row["uid"], row["label"], [span * 100, short * 100, dims[2] * 100], corners))
    return boxes


def project(corners, rot, translation, intrinsics):
    width, height, fx, fy, cx, cy = intrinsics
    points = corners @ rot.T + translation
    if (points[:, 2] < 0.12).any():
        return None
    pixels = points[:, :2] / points[:, 2:] * [fx, fy] + [cx, cy]
    lo, hi = pixels.min(axis=0), pixels.max(axis=0)
    clipped_lo, clipped_hi = np.maximum(lo, 0), np.minimum(hi, [width, height])
    size = clipped_hi - clipped_lo
    if (size < 23).any() or np.prod(size) / np.prod(hi - lo) < 0.91:
        return None
    return np.r_[np.floor(clipped_lo), np.ceil(clipped_hi)].astype(int).tolist()


def picks(scan_root, scan, boxes):
    with zipfile.ZipFile(scan_root / "lowres_wide_intrinsics.zip") as archive:
        camera = {
            float(Path(name).stem.rsplit("_", 1)[1]): np.fromstring(
                archive.read(name).decode(), sep=" "
            )
            for name in archive.namelist()
            if name.endswith(".pincam")
        }
    available = np.array(sorted(camera))
    by_uid = {box[0]: [] for box in boxes}
    for line in (scan_root / "lowres_wide.traj").read_text().splitlines()[::9]:
        seconds, *pose = map(float, line.split())
        nearest = available[np.argmin(abs(available - seconds))]
        if abs(seconds - nearest) > 0.012:
            continue
        k = camera[nearest]
        rotation = cv2.Rodrigues(np.asarray(pose[:3]))[0]
        for uid, label, dims, corners in boxes:
            bbox = project(corners, rotation, pose[3:], k)
            if bbox is not None:
                by_uid[uid].append(
                    {
                        "uid": uid,
                        "label": label,
                        "dimensions_cm_long_short_height": dims,
                        "bbox": bbox,
                        "size": list(k[:2]),
                        "timestamp": f"{nearest:.3f}",
                        "scan": scan,
                    }
                )
    records = {}
    for rows in by_uid.values():
        for j in np.unique(np.linspace(0, len(rows) - 1, min(5, len(rows)), dtype=int)):
            row = rows[j]
            records.setdefault(row["timestamp"], []).append(row)
    return records


def collect_scene(item):
    split, scan = item
    scan_root = ROOT / "data" / split / scan
    boxes = furniture(scan_root / f"{scan}_3dod_annotation.json")
    if not boxes:
        return {"scan": scan, "frames": 0, "objects": 0}
    records = picks(scan_root, scan, boxes)
    if not records:
        return {"scan": scan, "frames": 0, "objects": 0}
    frames = scan_root / "frames"
    frames.mkdir(exist_ok=True)
    kept = []
    access = RemoteZip(f"{BASE}/{split}/{scan}/lowres_wide.zip")
    with zipfile.ZipFile(access) as archive:
        names = {
            Path(name).stem.rsplit("_", 1)[1]: name
            for name in archive.namelist()
            if re.search(r"_\d+\.\d+\.png$", name)
        }
        for stamp, rows in sorted(records.items()):
            key = next(
                (
                    value
                    for value in (
                        stamp,
                        f"{float(stamp) - 0.001:.3f}",
                        f"{float(stamp) + 0.001:.3f}",
                    )
                    if value in names
                ),
                None,
            )
            if key is None:
                continue
            path = frames / f"{key}.png"
            if not path.exists():
                path.write_bytes(archive.read(names[key]))
            kept.extend(
                {**row, "image": str(path.relative_to(ROOT)), "split": split} for row in rows
            )
    (scan_root / "samples.json").write_text(json.dumps(kept, indent=2) + "\n")
    print(split, scan, len(kept), "object views", flush=True)
    return {
        "scan": scan,
        "split": split,
        "frames": len({r["image"] for r in kept}),
        "objects": len({r["uid"] for r in kept}),
        "bytes": access.transferred,
    }


if __name__ == "__main__":
    tasks = [(split, scan) for split, scans in SCANS.items() for scan in scans]
    summaries = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(collect_scene, task) for task in tasks]
        for (_, scan), job in zip(tasks, jobs):
            try:
                summaries.append(job.result())
            except Exception as error:
                print(scan, type(error).__name__, error, flush=True)
                summaries.append({"scan": scan, "error": str(error)})
    (ROOT / "frame-collection.json").write_text(json.dumps(summaries, indent=2) + "\n")
