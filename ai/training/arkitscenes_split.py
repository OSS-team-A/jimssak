"""Freeze an ARKitScenes development/holdout split for furniture size evaluation.

The holdout set is drawn from the official Validation fold and split by room
(``visit_id``), so frames of the same room never appear on both sides. Rooms that
contain a scan already used by an earlier experiment always stay in development.

Usage (repository root)::

    # 1. create the split once and commit the JSON
    python -m ai.training.arkitscenes_split make metadata.csv --used-from outputs \
        --output experiments/size-estimation/arkitscenes-split.json

    # 2. on a machine that can reach Apple's server, list holdout furniture
    python -m ai.training.arkitscenes_split objects \
        experiments/size-estimation/arkitscenes-split.json --side holdout \
        --annotations outputs/arkitscenes-annotations --download \
        --output outputs/arkitscenes-holdout-objects.csv

    # 3. refuse development jobs that contain holdout scans
    python -m ai.training.arkitscenes_split check \
        experiments/size-estimation/arkitscenes-split.json outputs/job.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
SALT = "jimssak-arkitscenes-holdout-v1"
SCAN_ID = re.compile(r"(?<!\d)(\d{8})(?!\d)")
FURNITURE = (
    "bed", "cabinet", "chair", "refrigerator", "shelf", "sofa", "stool", "table",
    "tv_monitor", "washer",
)
TEXT_SUFFIXES = {".json", ".csv", ".py", ".md", ".txt"}
SKIP_FILES = {"metadata.csv", "scene-split.csv"}


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = {row["video_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def find_used_scans(root: Path, known: set[str]) -> dict[str, list[str]]:
    """Return scan ids mentioned under ``root`` with the top-level folders that mention them."""
    used: dict[str, set[str]] = defaultdict(set)
    for path in root.rglob("*"):
        source = path.relative_to(root).parts[0]
        if path.is_dir() and path.name in known:
            used[path.name].add(source)
        elif (
            path.is_file()
            and path.suffix in TEXT_SUFFIXES
            and path.name not in SKIP_FILES
            and path.stat().st_size < 5_000_000
        ):
            for scan in set(SCAN_ID.findall(path.read_text(errors="ignore"))) & known:
                used[scan].add(source)
    return {scan: sorted(sources) for scan, sources in sorted(used.items())}


def _bucket(visit: str) -> float:
    digest = hashlib.sha256(f"{SALT}:{visit}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def make_split(
    metadata: dict[str, dict[str, str]],
    used: dict[str, list[str]],
    holdout_fraction: float = 0.4,
) -> dict:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    rooms: dict[str, list[str]] = defaultdict(list)
    for scan, row in metadata.items():
        if row["fold"] == "Validation" and row.get("is_in_threedod", "True") == "True":
            visit = row["visit_id"] if row["visit_id"] not in ("", "NA") else f"scan-{scan}"
            rooms[visit].append(scan)
    contaminated = {visit for visit, scans in rooms.items() if any(s in used for s in scans)}
    holdout_rooms = sorted(
        visit for visit in rooms
        if visit not in contaminated and _bucket(visit) < holdout_fraction
    )
    holdout = sorted(scan for visit in holdout_rooms for scan in rooms[visit])
    leaked = sorted(set(holdout) & set(used))
    if leaked:
        raise AssertionError(f"used scans entered holdout: {leaked}")
    return {
        "version": 1,
        "rule": (
            "ARKitScenes Validation fold with 3DOD annotations, grouped by visit_id. "
            "Rooms with any scan in used_scans stay in development. Other rooms go to "
            f"holdout when sha256('{SALT}:<visit_id>') falls below holdout_fraction."
        ),
        "holdout_fraction": holdout_fraction,
        "counts": {
            "validation_rooms": len(rooms),
            "validation_scans": sum(len(s) for s in rooms.values()),
            "contaminated_rooms": len(contaminated),
            "holdout_rooms": len(holdout_rooms),
            "holdout_scans": len(holdout),
            "used_scans": len(used),
        },
        "holdout_rooms": holdout_rooms,
        "holdout_scans": holdout,
        "used_scans": used,
    }


def load_split(path: Path) -> dict:
    split = json.loads(path.read_text())
    if split.get("version") != 1:
        raise ValueError(f"unsupported split file: {path}")
    return split


def scan_ids_in_job(job: dict) -> set[str]:
    """Collect ARKitScenes scan ids from a MeasurementJob's video ids and paths."""
    found: set[str] = set()
    for video in job.get("videos", []):
        for key in ("video_id", "path"):
            found.update(SCAN_ID.findall(str(video.get(key, ""))))
    return found


def check_jobs(split: dict, paths: list[Path], final: bool = False) -> list[str]:
    holdout = set(split["holdout_scans"])
    problems = []
    for path in paths:
        scans = scan_ids_in_job(json.loads(path.read_text()))
        inside, outside = sorted(scans & holdout), sorted(scans - holdout)
        if not final and inside:
            problems.append(f"{path}: holdout scans in a development job: {inside}")
        if final and outside:
            problems.append(f"{path}: non-holdout scans in a final evaluation job: {outside}")
    return problems


def annotation_path(root: Path, scan: str) -> Path:
    return root / scan / f"{scan}_3dod_annotation.json"


def download_annotation(root: Path, scan: str, fold: str = "Validation") -> Path:
    path = annotation_path(root, scan)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{fold}/{scan}/{path.name}"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()
    json.loads(data)
    path.write_bytes(data)
    return path


def furniture_objects(annotation: dict, scan: str) -> list[dict]:
    """Gravity-aligned furniture boxes in cm: width >= depth on the floor plane."""
    rows = []
    for item in annotation.get("data", []):
        if item.get("label") not in FURNITURE:
            continue
        box = item["segments"]["obbAligned"]
        lengths = [float(v) for v in box["axesLengths"]]
        axes = [float(v) for v in box["normalizedAxes"]]
        upright = abs(axes[8]) >= 0.9
        if not upright or min(lengths) < 0.15 or max(lengths) > 3.5:
            continue
        width, depth = sorted(lengths[:2], reverse=True)
        rows.append(
            {
                "scan": scan,
                "uid": item["uid"],
                "label": item["label"],
                "width_cm": round(width * 100, 2),
                "depth_cm": round(depth * 100, 2),
                "height_cm": round(lengths[2] * 100, 2),
            }
        )
    return rows


def collect_objects(
    split: dict, side: str, root: Path, download: bool, metadata: dict | None = None
) -> tuple[list[dict], list[str]]:
    if side == "holdout":
        scans = split["holdout_scans"]
    elif metadata is None:
        raise ValueError("--metadata is required for the development side")
    else:
        holdout = set(split["holdout_scans"])
        scans = sorted(
            s for s, r in metadata.items() if r["fold"] == "Validation" and s not in holdout
        )
    rows, missing = [], []
    for scan in scans:
        path = annotation_path(root, scan)
        try:
            if download:
                path = download_annotation(root, scan)
            if not path.exists():
                missing.append(scan)
                continue
            rows.extend(furniture_objects(json.loads(path.read_text()), scan))
        except Exception as error:  # report every failed scan, keep going
            missing.append(f"{scan} ({error})")
    return rows, missing


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["scan", "uid", "label", "width_cm", "depth_cm", "height_cm"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _new_file(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; the split is frozen. Pass --force to replace it.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make", help="create the frozen split JSON")
    make.add_argument("metadata", type=Path)
    make.add_argument("--used-from", type=Path, action="append", default=[])
    make.add_argument("--holdout-fraction", type=float, default=0.4)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--force", action="store_true")

    objects = sub.add_parser("objects", help="list furniture boxes from 3DOD annotations")
    objects.add_argument("split", type=Path)
    objects.add_argument("--side", choices=("holdout", "development"), default="holdout")
    objects.add_argument("--metadata", type=Path)
    objects.add_argument("--annotations", type=Path, required=True)
    objects.add_argument("--download", action="store_true")
    objects.add_argument("--output", type=Path, required=True)

    check = sub.add_parser("check", help="fail when a job mixes in the wrong scans")
    check.add_argument("split", type=Path)
    check.add_argument("jobs", type=Path, nargs="+")
    check.add_argument("--final", action="store_true", help="job must use holdout scans only")

    args = parser.parse_args(argv)
    if args.command == "make":
        _new_file(args.output, args.force)
        metadata = read_metadata(args.metadata)
        used: dict[str, set[str]] = defaultdict(set)
        for root in args.used_from:
            for scan, sources in find_used_scans(root, set(metadata)).items():
                used[scan].update(f"{root.name}/{s}" for s in sources)
        split = make_split(
            metadata, {k: sorted(v) for k, v in used.items()}, args.holdout_fraction
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(split["counts"], indent=2))
        return 0
    if args.command == "objects":
        metadata = read_metadata(args.metadata) if args.metadata else None
        rows, missing = collect_objects(
            load_split(args.split), args.side, args.annotations, args.download, metadata
        )
        write_csv(rows, args.output)
        per_label: dict[str, int] = defaultdict(int)
        for row in rows:
            per_label[row["label"]] += 1
        print(json.dumps({
            "objects": len(rows),
            "scans_with_objects": len({r["scan"] for r in rows}),
            "by_label": dict(sorted(per_label.items())),
            "missing_annotations": missing[:20],
            "missing_count": len(missing),
        }, indent=2))
        return 0 if rows or not missing else 1
    problems = check_jobs(load_split(args.split), args.jobs, args.final)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
