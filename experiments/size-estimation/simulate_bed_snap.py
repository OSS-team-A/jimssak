"""Estimate the gain of bed standard-size snapping on development beds.

No development scene video shows a whole bed in frame, so beds cannot be measured end to
end yet. Instead every true bed size from the development split is perturbed with
calibrated errors observed on real development videos (other furniture, leave-one-room-out
calibration), then snapped with the US standards. The frame allowance is refit without
the bed's own scan.

    python experiments/size-estimation/simulate_bed_snap.py \
        outputs/arkitscenes-dev-objects.csv outputs/dev-eval-scene/truth.json \
        outputs/dev-eval-scene/dav2.json --metadata outputs/arkit-depth-fix/metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ai.measurement import calibration, standard_sizes  # noqa: E402
from ai.measurement.moving_contract import MeasurementResponse  # noqa: E402

AXES = ("width", "depth")


def calibrated_errors(truth: dict, response: MeasurementResponse, rooms: dict) -> dict:
    """log(calibrated / true) per (kind, axis), calibration fit without the row's room."""
    rows = calibration.samples(response, truth)
    errors: dict = {}
    for room in {calibration.room_of(r["video_id"], rooms) for r in rows}:
        model = calibration.fit([r for r in rows
                                 if calibration.room_of(r["video_id"], rooms) != room])
        for row in rows:
            if calibration.room_of(row["video_id"], rooms) != room:
                continue
            factor = calibration.factor_for(model, row["kind"], row["axis"], row["method"])
            value = math.log(row["estimate"] * (factor or 1.0) / row["actual"])
            errors.setdefault((row["kind"], row["axis"]), []).append(value)
    return errors


def fit_allowance(beds: list[dict]) -> dict[str, float]:
    allowance = {"width": 0.0, "length": 0.0}
    for _ in range(5):
        standard_sizes.ALLOWANCE_CM = allowance
        gaps = {"width": [], "length": []}
        for bed in beds:
            best = min(standard_sizes.STANDARDS["us"], key=lambda s: sum(
                math.log(bed[a] / standard_sizes.outer_size(s)[a]) ** 2 for a in AXES))
            gaps["width"].append(bed["depth"] - best[2])
            gaps["length"].append(bed["width"] - best[3])
        allowance = {k: median(v) for k, v in gaps.items()}
    return allowance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("objects", type=Path, help="arkitscenes_split objects CSV (dev)")
    parser.add_argument("truth", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args(argv)

    beds = [{"scan": r["scan"], "width": float(r["width_cm"]), "depth": float(r["depth_cm"])}
            for r in csv.DictReader(args.objects.open()) if r["label"] == "bed"]
    errors = calibrated_errors(json.loads(args.truth.read_text()),
                               MeasurementResponse.model_validate_json(args.result.read_text()),
                               calibration.rooms_from_metadata(args.metadata))
    rng = random.Random(args.seed)
    report = {"beds": len(beds)}
    original = standard_sizes.ALLOWANCE_CM
    for kind in ("rgb", "arkit"):
        rel = {m: {a: [] for a in AXES} for m in ("calibrated", "snapped")}
        nonstandard = 0
        for bed in beds:
            others = [b for b in beds if b["scan"] != bed["scan"]]
            standard_sizes.ALLOWANCE_CM = fit_allowance(others)
            for _ in range(args.draws):
                measured = {a: bed[a] * math.exp(rng.choice(errors[(kind, a)])) for a in AXES}
                best = standard_sizes.match(measured, "us", kind)
                snapped = standard_sizes.outer_size(best) if best else measured
                nonstandard += best is None
                for a in AXES:
                    rel["calibrated"][a].append(abs(measured[a] / bed[a] - 1))
                    rel["snapped"][a].append(abs(snapped[a] / bed[a] - 1))
        report[kind] = {
            m: {a: {"median_relative_error": round(median(v), 3),
                    "within_10_percent": round(sum(x <= 0.1 for x in v) / len(v), 3)}
                for a, v in axes.items()}
            for m, axes in rel.items()
        }
        report[kind]["nonstandard_share"] = round(nonstandard / (len(beds) * args.draws), 3)
    standard_sizes.ALLOWANCE_CM = original
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
