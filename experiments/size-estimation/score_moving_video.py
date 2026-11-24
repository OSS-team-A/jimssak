"""Score centimeter estimates using a separate physical truth file."""

import argparse
import json
from pathlib import Path
from statistics import mean

from pydantic import Field

from ai.contracts import Contract
from ai.measurement.moving_contract import Dimension, MeasurementResponse


class TrackTruth(Contract):
    video_id: str
    track_id: str
    dimensions_cm: dict[Dimension, float]


class EvaluationTruth(Contract):
    tracks: list[TrackTruth] = Field(min_length=1)


def score(results: MeasurementResponse, truth: EvaluationTruth):
    videos = {item.video_id: item for item in results.videos}
    rows = []
    for reference in truth.tracks:
        clip = videos.get(reference.video_id)
        if clip is None:
            raise ValueError(f"No result found for video {reference.video_id}")
        track = next((x for x in clip.objects if x.track_id == reference.track_id), None)
        if track is None:
            raise ValueError(f"No result found for track {reference.track_id}")
        anchor = clip.scale.reference
        dimensions = {}
        for key, actual in reference.dimensions_cm.items():
            if actual <= 0:
                raise ValueError("Ground truth lengths must be positive")
            value = track.dimensions.get(key)
            calibrated = anchor and anchor.track_id == track.track_id and anchor.dimension == key
            status = "calibration_input" if calibrated else "measured" if value else "unavailable"
            dimensions[key] = {
                "status": status,
                "reference_cm": actual,
                "estimated_cm": value.cm if value else None,
                "error_cm": abs(value.cm - actual) if value and not calibrated else None,
            }
        errors = [x["error_cm"] for x in dimensions.values() if x["error_cm"] is not None]
        rows.append({
            "video_id": clip.video_id,
            "track_id": track.track_id,
            "scale_status": clip.scale.status,
            "dimensions": dimensions,
            "mae_cm": mean(errors) if errors else None,
            "scored_dimensions": len(errors),
            "eligible_dimensions": sum(x["status"] != "calibration_input"
                                       for x in dimensions.values()),
        })
    return {"tracks": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimates", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = MeasurementResponse.model_validate_json(args.estimates.read_text())
    truth = EvaluationTruth.model_validate_json(args.truth.read_text())
    scored = score(results, truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scored, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
