"""Recognition from AI 1: run its detector and tracker, then turn the run into views.

AI 1 (``ai/training/ai1``) writes, per video, ``frames.jsonl`` (every detection with its
track number, box and outline polygons), ``items.csv`` (tracks kept as items) and, after
``link_reappearance.py``, ``relink_*/items.csv`` (tracks of one object that left and came
back, merged under one group id). :func:`from_run` converts such a run into the
:class:`~ai.service.pipeline.Recognition` the measurement stage reads, following the rules
agreed with AI 1 (``docs/ai1-handoff-for-ai2.md``):

- Coordinates are display-orientation pixels, so ``pixel_rotation_ccw`` is 0. When the long
  side exceeds 1920 px (4K), coordinates and ``annotation_size_px`` are scaled together.
- One measured object per item: a relinked group id when tracks were merged automatically,
  otherwise the track's item id. Possible matches AI 1 only flagged are left separate.
- An outline in several pieces (an object split by something in front of it) is sent whole
  as ``mask_rle``, so both ends stay in the measurement; one piece is sent as a polygon.
- A detection whose box touches the image border is marked ``complete: false``.

The worker uses :func:`recognize` (``--recognizer ai.service.ai1_adapter:recognize``). It
runs ``infer.py`` and ``link_reappearance.py`` as subprocesses, so AI 1's own dependencies
(Ultralytics) only need to be installed where the worker runs. Settings come from the
environment: ``JIMSSAK_AI1_WEIGHTS`` (required, e.g. ``B1_yolo11s-seg_best.pt``),
``JIMSSAK_AI1_DEVICE`` (default ``cpu``) and ``JIMSSAK_AI1_PYTHON`` (default: this Python).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ai.measurement.moving_contract import VideoInput
from ai.service.pipeline import NoFurniture, Recognition

AI1_DIR = Path(__file__).resolve().parents[1] / "training" / "ai1"
MAX_SIDE = 1920
# Annotated frames per object, spread over its track. The measurement stage takes at most
# 180 keyframes per video, and the rest of them go to camera tracking between views.
VIEWS_PER_ITEM = 8
MAX_ANNOTATED_FRAMES = 100
BORDER_PX = 2
LISTED = {"confirmed", "needs_review"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _items_file(run_dir: Path, relink_dir: Path | None) -> Path:
    if relink_dir is not None:
        return relink_dir / "items.csv"
    relinks = sorted(run_dir.glob("relink_*/items.csv"))
    return relinks[-1] if relinks else run_dir / "items.csv"


def _image_size(run_dir: Path) -> tuple[int, int]:
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        video = json.loads(manifest.read_text(encoding="utf-8")).get("video", {})
        if video.get("decoded_width") and video.get("decoded_height"):
            return int(video["decoded_width"]), int(video["decoded_height"])
    raise ValueError(f"No decoded video size in {manifest}")


def _rle(polygons: list[np.ndarray], width: int, height: int) -> list[int]:
    """Row-major run lengths of a filled mask, starting with a background run."""
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [p.astype(np.int32) for p in polygons], 1)
    flat = mask.ravel()
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(bounds).tolist()
    return runs if flat[0] == 0 else [0, *runs]


def _view(det: dict, item_id: str, label: str, frame: int, scale: float,
          size: tuple[int, int]) -> dict | None:
    width, height = size
    pieces = []
    for polygon in det.get("mask_polygons") or []:
        points = np.asarray(polygon, dtype=float) * scale
        if len(points) >= 3:
            points[:, 0] = points[:, 0].clip(0, width - 1)
            points[:, 1] = points[:, 1].clip(0, height - 1)
            pieces.append(points)
    if not pieces:
        return None
    x1, y1, x2, y2 = (v * scale for v in det["box_xyxy"])
    complete = (x1 > BORDER_PX and y1 > BORDER_PX
                and x2 < width - 1 - BORDER_PX and y2 < height - 1 - BORDER_PX)
    view = {"frame_index": frame, "track_id": item_id, "label": label, "complete": complete}
    if len(pieces) == 1:
        view["mask_polygon_px"] = [[round(x, 1), round(y, 1)] for x, y in pieces[0]]
    else:
        view["mask_rle"] = _rle(pieces, width, height)
    return view


def _spread(entries: list[tuple[int, dict]], count: int) -> list[tuple[int, dict]]:
    """Up to ``count`` detections spread over the track, the largest mask in each stretch."""
    if len(entries) <= count:
        return entries
    stretches = np.array_split(np.arange(len(entries)), count)
    return [max((entries[i] for i in part), key=lambda e: e[1].get("mask_area_px") or 0)
            for part in stretches]


def from_run(run_dir: Path, video_path: Path, relink_dir: Path | None = None) -> Recognition:
    """Convert one AI 1 run folder into the measurement stage's input."""
    run_dir = Path(run_dir)
    items = [row for row in _read_csv(_items_file(run_dir, relink_dir))
             if row.get("status") in LISTED and row.get("category")]
    owner = {}
    for row in items:
        for track in str(row.get("track_ids", "")).split(";"):
            if track.strip():
                owner[int(track)] = row
    frames = [json.loads(line) for line in (run_dir / "frames.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    width, height = _image_size(run_dir)
    scale = min(1.0, MAX_SIDE / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))

    detections: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for record in frames:
        best = {}
        for det in record.get("detections") or record.get("dets") or []:
            row = owner.get(det.get("track_id"))
            if row is None or not det.get("mask_polygons"):
                continue
            item = row["pred_item_id"]
            # Tracks of one group never overlap in time; keep the larger mask if they do.
            if item not in best or (det.get("mask_area_px") or 0) > (
                    best[item].get("mask_area_px") or 0):
                best[item] = det
        for item, det in best.items():
            detections[item].append((int(record["frame_idx"]), det))
    if not detections:
        raise NoFurniture("AI 1 found no listed furniture with a mask")

    per_item = max(2, min(VIEWS_PER_ITEM, MAX_ANNOTATED_FRAMES // len(detections)))
    labels = {row["pred_item_id"]: row["category"] for row in items}
    views = []
    for item, entries in sorted(detections.items()):
        for frame, det in _spread(sorted(entries, key=lambda e: e[0]), per_item):
            view = _view(det, item, labels[item], frame, scale, size)
            if view is not None:
                views.append(view)
    video = VideoInput.model_validate({
        "video_id": run_dir.name, "path": str(video_path), "annotation_size_px": size,
        "pixel_rotation_ccw": 0, "views": views,
    })
    confidences = {}
    for row in items:
        if row["pred_item_id"] in detections and row.get("mean_conf"):
            confidences[row["pred_item_id"]] = float(row["mean_conf"])
    return Recognition(video=video, confidences=confidences)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=AI1_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-2000:]
        raise RuntimeError(f"AI 1 step failed ({command[1]}):\n{tail}")


def recognize(path: Path) -> Recognition:
    """Run AI 1 on one video and convert the result (the worker's ``--recognizer``)."""
    weights = os.environ.get("JIMSSAK_AI1_WEIGHTS")
    if not weights:
        raise RuntimeError("Set JIMSSAK_AI1_WEIGHTS to AI 1's segmentation weights (.pt)")
    python = os.environ.get("JIMSSAK_AI1_PYTHON", sys.executable)
    device = os.environ.get("JIMSSAK_AI1_DEVICE", "cpu")
    with tempfile.TemporaryDirectory(prefix="ai1-") as work:
        root = Path(work)
        _run([python, "infer.py", "--video", str(Path(path).resolve()), "--video-id", "upload",
              "--label", "service", "--run-name", "run", "--weights", str(Path(weights).resolve()),
              "--device", device, "--output-root", str(root), "--no-video"])
        run_dir = root / "service" / "run"
        _run([python, "link_reappearance.py", "--run", str(run_dir), "--out-name", "relink"])
        return from_run(run_dir, Path(path), run_dir / "relink")


def main(argv: list[str] | None = None) -> int:
    """Write a measurement job from a video (runs AI 1) or from an existing AI 1 run."""
    import argparse

    from ai.measurement.moving_contract import MeasurementJob

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--run", type=Path, help="existing infer.py run folder (skip AI 1)")
    parser.add_argument("--output", type=Path, required=True, help="measurement job JSON")
    args = parser.parse_args(argv)
    result = (from_run(args.run, args.video) if args.run else recognize(args.video))
    job = MeasurementJob(videos=[result.video])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(job.model_dump_json(indent=1) + "\n")
    views = result.video.views
    print(json.dumps({"objects": len({v.track_id for v in views}), "views": len(views),
                      "labels": sorted({f"{v.track_id}:{v.label}" for v in views}),
                      "confidences": result.confidences}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
