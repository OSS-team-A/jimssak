"""Learn and apply per-dimension size corrections from development-set measurements.

Plain RGB video tends to overestimate furniture: on 100 development scans the median
estimate was 1.39x the true size (width 1.54x, depth 1.51x, height 1.19x). A multiplicative
factor per (video kind, dimension, measurement method), learned as the median of
true/estimate in log space, removes most of that bias.

    python -m ai.measurement.calibration fit outputs/dev-eval-100/truth.json \
        outputs/dev-eval-100/dav2.json --metadata outputs/arkit-depth-fix/metadata.csv \
        --output ai/measurement/calibration/depth-anything-v2.json
    python -m ai.measurement.calibration apply outputs/result.json \
        --calibration ai/measurement/calibration/depth-anything-v2.json \
        --output outputs/result-calibrated.json

``fit`` also reports leave-one-room-out errors (rooms from ARKitScenes ``visit_id``), so
the reported gain never comes from a room the factors were learned on.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from ai.measurement.moving_contract import MeasurementResponse

AXES = ("width", "depth", "height")
KINDS = {"capture_poses": "arkit", "rgb_metric_depth_pnp": "rgb"}
MIN_SAMPLES = 10
SCAN_ID = re.compile(r"(?<!\d)(\d{8})(?!\d)")
QUALIFIER = "size_calibrated"


def _key(kind: str, axis: str, method: str | None = None) -> str:
    return "/".join(part for part in (kind, axis, method) if part)


def samples(response: MeasurementResponse, truth: dict) -> list[dict]:
    """One row per measured primary dimension that has a true length."""
    references = {(t["video_id"], t["track_id"]): t["dimensions_cm"] for t in truth["tracks"]}
    rows = []
    for clip in response.videos:
        kind = KINDS[clip.pose_source]
        if clip.scale.status == "reference_scaled":
            continue  # already scaled by a known length
        for track in clip.objects:
            reference = references.get((clip.video_id, track.track_id))
            if reference is None:
                continue
            for axis in AXES:
                estimate = track.dimensions.get(axis)
                actual = reference.get(axis)
                if estimate is None or not actual:
                    continue
                rows.append({
                    "video_id": clip.video_id, "kind": kind, "axis": axis,
                    "method": estimate.method, "estimate": estimate.cm, "actual": actual,
                })
    return rows


def fit(rows: list[dict], min_samples: int = MIN_SAMPLES) -> dict:
    """Median log(true/estimate) per group, with coarser groups as fallbacks."""
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        ratio = math.log(row["actual"] / row["estimate"])
        groups[row["kind"]].append(ratio)
        groups[_key(row["kind"], row["axis"])].append(ratio)
        groups[_key(row["kind"], row["axis"], row["method"])].append(ratio)
    factors = {key: {"factor": round(math.exp(median(values)), 4), "samples": len(values)}
               for key, values in sorted(groups.items()) if len(values) >= min_samples}
    return {"version": 1, "min_samples": min_samples, "factors": factors}


def factor_for(calibration: dict, kind: str, axis: str, method: str) -> float | None:
    factors = calibration["factors"]
    for key in (_key(kind, axis, method), _key(kind, axis), kind):
        if key in factors:
            return factors[key]["factor"]
    return None


def apply(response: MeasurementResponse, calibration: dict) -> MeasurementResponse:
    """Scale width, depth and height; videos scaled by a known length are left alone."""
    videos = []
    for clip in response.videos:
        kind = KINDS[clip.pose_source]
        if clip.scale.status == "reference_scaled":
            videos.append(clip)
            continue
        objects = []
        for track in clip.objects:
            dimensions = dict(track.dimensions)
            changed = False
            for axis in AXES:
                estimate = dimensions.get(axis)
                if estimate is None:
                    continue
                factor = factor_for(calibration, kind, axis, estimate.method)
                if factor is not None:
                    dimensions[axis] = estimate.model_copy(
                        update={"cm": round(estimate.cm * factor, 2)})
                    changed = True
            qualifiers = sorted(set(track.qualifiers) | ({QUALIFIER} if changed else set()))
            objects.append(track.model_copy(
                update={"dimensions": dimensions, "qualifiers": qualifiers}))
        videos.append(clip.model_copy(update={"objects": objects}))
    return MeasurementResponse(videos=videos)


def rooms_from_metadata(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {row["video_id"]: row["visit_id"] for row in csv.DictReader(handle)
                if row.get("visit_id") not in (None, "", "NA")}


def room_of(video_id: str, rooms: dict[str, str]) -> str:
    match = SCAN_ID.search(video_id)
    scan = match.group(1) if match else video_id
    return rooms.get(scan, f"scan-{scan}")


def cross_validate(rows: list[dict], rooms: dict[str, str],
                   min_samples: int = MIN_SAMPLES) -> dict:
    """Leave-one-room-out absolute errors, before and after calibration, per video kind."""
    by_room: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_room[room_of(row["video_id"], rooms)].append(row)
    errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for room, held_out in by_room.items():
        model = fit([r for r in rows if room_of(r["video_id"], rooms) != room], min_samples)
        for row in held_out:
            factor = factor_for(model, row["kind"], row["axis"], row["method"]) or 1.0
            for name, value in (("raw", row["estimate"]), ("calibrated", row["estimate"] * factor)):
                error = abs(value - row["actual"])
                errors[row["kind"]][name].append(error)
                errors[row["kind"]][f"{name}/{row['axis']}"].append(error)
                errors[row["kind"]][f"{name}/within10"].append(
                    float(error <= 0.10 * row["actual"]))
    report = {"rooms": len(by_room), "kinds": {}}
    for kind, values in sorted(errors.items()):
        report["kinds"][kind] = {
            "dimensions": len(values["raw"]),
            **{name: round(mean(v), 3) for name, v in sorted(values.items())},
        }
    return report


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fit_cmd = sub.add_parser("fit", help="learn factors from measured development videos")
    fit_cmd.add_argument("truth", type=Path)
    fit_cmd.add_argument("results", type=Path, nargs="+")
    fit_cmd.add_argument("--metadata", type=Path, help="ARKitScenes metadata.csv for rooms")
    fit_cmd.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    fit_cmd.add_argument("--output", type=Path, required=True)
    apply_cmd = sub.add_parser("apply", help="scale a measurement result")
    apply_cmd.add_argument("result", type=Path)
    apply_cmd.add_argument("--calibration", type=Path, required=True)
    apply_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "fit":
        truth = _load_json(args.truth)
        rows = []
        for path in args.results:
            rows += samples(MeasurementResponse.model_validate_json(path.read_text()), truth)
        if not rows:
            raise SystemExit("No measured dimension matches the truth file")
        model = fit(rows, args.min_samples)
        rooms = rooms_from_metadata(args.metadata) if args.metadata else {}
        model["trained_on"] = {
            "dimensions": len(rows),
            "videos": len({r["video_id"] for r in rows}),
            "rooms": len({room_of(r["video_id"], rooms) for r in rows}),
        }
        model["leave_one_room_out"] = cross_validate(rows, rooms, args.min_samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(model, indent=2) + "\n")
        print(json.dumps(model["leave_one_room_out"], indent=2))
        return 0
    calibration = _load_json(args.calibration)
    response = MeasurementResponse.model_validate_json(args.result.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(apply(response, calibration).model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
