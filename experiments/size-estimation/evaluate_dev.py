"""Compare measurement runs on the development set, split by ARKit and plain RGB videos.

    PYTHONPATH=. python experiments/size-estimation/evaluate_dev.py outputs/dev-eval/truth.json \
        dav2=outputs/dev-eval/dav2.json dp=outputs/dev-eval/dp.json \
        fused=outputs/dev-eval/fused.json --output outputs/dev-eval/evaluation.json

Errors are averaged over the dimensions that every run measured, so a run that skips hard
objects does not look better. Coverage and the share of multiview estimates are reported
beside them, and so is the per-dimension best run on this set (for choosing a fusion
policy on development data only).
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median

from score_moving_video import EvaluationTruth, score

from ai.measurement.moving_contract import MeasurementResponse

AXES = ("width", "depth", "height")
SOURCES = {"capture_poses": "arkit", "rgb_metric_depth_pnp": "rgb"}


def collect(response: MeasurementResponse, truth: EvaluationTruth) -> dict:
    """(video_id, track_id, axis) -> (abs error cm, relative error, method)."""
    clips = {clip.video_id: clip for clip in response.videos}
    rows = {}
    for track in score(response, truth)["tracks"]:
        clip = clips[track["video_id"]]
        measured = next(o for o in clip.objects if o.track_id == track["track_id"])
        for axis in AXES:
            entry = track["dimensions"].get(axis)
            if not entry or entry["error_cm"] is None:
                continue
            rows[(track["video_id"], track["track_id"], axis)] = (
                entry["error_cm"], entry["error_cm"] / entry["reference_cm"],
                measured.dimensions[axis].method, SOURCES[clip.pose_source],
            )
    return rows


def summarize(runs: dict[str, dict], truth: EvaluationTruth) -> dict:
    eligible = {(t.video_id, t.track_id, axis) for t in truth.tracks for axis in AXES
                if axis in t.dimensions_cm}
    report = {}
    for source in ("arkit", "rgb"):
        keys = {k for k in eligible
                if any(k in rows and rows[k][3] == source for rows in runs.values())}
        common = sorted(k for k in keys if all(k in rows for rows in runs.values()))
        section = {"eligible_dimensions": len(keys), "common_dimensions": len(common),
                   "runs": {}}
        for name, rows in runs.items():
            measured = [k for k in keys if k in rows]
            by_axis = {axis: mean(rows[k][0] for k in common if k[2] == axis)
                       for axis in AXES if any(k[2] == axis for k in common)}
            section["runs"][name] = {
                "mae_cm": mean(rows[k][0] for k in common) if common else None,
                "mae_by_axis_cm": by_axis,
                "median_relative_error": median(rows[k][1] for k in common) if common else None,
                "within_10_percent": (sum(rows[k][1] <= 0.10 for k in common) / len(common)
                                      if common else None),
                "coverage": len(measured) / len(keys) if keys else None,
                "multiview_share": (sum(rows[k][2] == "registered_multiview" for k in measured)
                                    / len(measured) if measured else None),
            }
        if common:
            section["best_run_by_axis"] = {
                axis: min(runs, key=lambda name: section["runs"][name]["mae_by_axis_cm"].get(
                    axis, float("inf")))
                for axis in AXES if any(k[2] == axis for k in common)
            }
        report[source] = section
    return report


def table(report: dict) -> str:
    lines = []
    for source, section in report.items():
        lines.append(f"\n[{source}] common dimensions: {section['common_dimensions']}"
                     f" / {section['eligible_dimensions']}")
        lines.append(f"{'run':<14}{'width':>8}{'depth':>8}{'height':>8}{'all':>8}"
                     f"{'<=10%':>8}{'cover':>8}{'multi':>8}")
        for name, run in section["runs"].items():
            axes = run["mae_by_axis_cm"]

            def cell(value, percent=False):
                if value is None:
                    return f"{'-':>8}"
                return f"{value * 100:>7.0f}%" if percent else f"{value:>8.2f}"

            lines.append(
                f"{name:<14}" + "".join(cell(axes.get(a)) for a in AXES) + cell(run["mae_cm"])
                + cell(run["within_10_percent"], True) + cell(run["coverage"], True)
                + cell(run["multiview_share"], True)
            )
        if "best_run_by_axis" in section:
            lines.append("best by axis: " + ", ".join(
                f"{a}={n}" for a, n in section["best_run_by_axis"].items()))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("truth", type=Path)
    parser.add_argument("runs", nargs="+", help="NAME=PATH of a moving_video or refine result")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = EvaluationTruth.model_validate_json(args.truth.read_text())
    runs = {}
    for item in args.runs:
        name, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"Use NAME=PATH, got {item}")
        runs[name] = collect(MeasurementResponse.model_validate_json(Path(path).read_text()),
                             truth)
    report = summarize(runs, truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(table(report))


if __name__ == "__main__":
    main()
