"""Compare depth checkpoints on matching object dimensions and report coverage."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

from score_moving_video import EvaluationTruth, score

from ai.measurement.moving_contract import MeasurementResponse

PRIMARY_AXES = ("width", "depth", "height")


def compare(responses, truth):
    reference_inputs = [
        (clip.video_id, clip.source_sha256, clip.pose_source) for clip in responses[0].videos
    ]
    for response in responses[1:]:
        candidate_inputs = [
            (v.video_id, v.source_sha256, v.pose_source) for v in response.videos
        ]
        if candidate_inputs != reference_inputs:
            raise ValueError("Every model must run the same source videos in the same order")
    captured = {v.video_id for v in responses[0].videos if v.pose_source == "capture_poses"}
    reports = [score(response, truth)["tracks"] for response in responses]
    errors = []
    for rows in reports:
        errors.append({
            (track["video_id"], track["track_id"], axis): entry["error_cm"]
            for track in rows if track["video_id"] in captured
            for axis, entry in track["dimensions"].items() if axis in PRIMARY_AXES
        })
    common = {key for key in errors[0] if all(mapping[key] is not None for mapping in errors)}
    if not common:
        raise ValueError("No shared ARKit dimensions are available for comparison")
    summaries = []
    families = set()
    for response, rows, mapping in zip(responses, reports, errors):
        tags = {clip.depth_model_family for clip in response.videos}
        if len(tags) != 1 or None in tags:
            raise ValueError("Each result file must identify one depth model family")
        family = tags.pop()
        if family in families:
            raise ValueError(f"Duplicate model result: {family}")
        families.add(family)
        axis_mae = {}
        for axis in PRIMARY_AXES:
            values = [mapping[key] for key in sorted(common) if key[2] == axis]
            axis_mae[axis] = mean(values) if values else None
        timings = [clip.depth_timing.median_s for clip in response.videos
                   if clip.video_id in captured]
        summaries.append({
            "family": family,
            "device": sorted({clip.device for clip in response.videos}),
            "mae_cm": mean(mapping[key] for key in sorted(common)),
            "mae_by_axis_cm": axis_mae,
            "measured_dimensions": sum(value is not None for value in mapping.values()),
            "eligible_dimensions": len(mapping),
            "common_dimensions": len(common),
            "median_clip_depth_time_s": median(timings),
            "tracks": rows,
        })
    ranked = sorted(summaries, key=lambda model: (model["mae_cm"], -model["measured_dimensions"]))
    return {"criterion": "ARKit MAE on dimensions measured by every model", "ranking": ranked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.results) < 2:
        parser.error("Provide results from at least two distinct models")
    truth = EvaluationTruth.model_validate_json(args.truth.read_text())
    results = [MeasurementResponse.model_validate_json(path.read_text()) for path in args.results]
    report = compare(results, truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
