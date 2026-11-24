"""Single-frame measurement baseline; no model inference or video processing."""

import argparse
import csv
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai.contracts import AnalysisResult
from ai.measurement.estimator import estimate


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    track_id: str = Field(min_length=1)
    width_cm: float = Field(gt=0)
    height_cm: float = Field(gt=0)


def read_truth(path: Path) -> list[GroundTruth]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["track_id", "width_cm", "height_cm"]:
            raise ValueError("CSV headers must be track_id,width_cm,height_cm")
        return [GroundTruth.model_validate(row) for row in reader]


def evaluate(result: AnalysisResult, truth: list[GroundTruth]) -> dict:
    if not truth:
        raise ValueError("Ground truth must contain at least one row")
    ids = [item.track_id for item in truth]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ground-truth track_id")
    predictions = {item.track_id: item for item in result.detections}
    if set(ids) - predictions.keys():
        raise ValueError("Ground truth contains track_id missing from input frame")
    rows = []
    unavailable = []
    for target in truth:
        predicted = predictions[target.track_id]
        if predicted.dimensions is None:
            unavailable.append(target.track_id)
            continue
        for axis in ("width_cm", "height_cm"):
            actual = getattr(target, axis)
            value = getattr(predicted.dimensions, axis)
            error = abs(value - actual)
            rows.append({
                "track_id": target.track_id, "axis": axis,
                "actual_cm": actual, "estimated_cm": value,
                "absolute_error_cm": error, "absolute_percentage_error": error / actual * 100,
            })
    return {
        "ground_truth_objects": len(truth), "measured_objects": len(rows) // 2,
        "unavailable_track_ids": unavailable,
        "mae_cm": sum(row["absolute_error_cm"] for row in rows) / len(rows) if rows else None,
        "mape_percent": (
            sum(row["absolute_percentage_error"] for row in rows) / len(rows) if rows else None
        ),
        "dimensions": rows,
    }


def run_experiment(result: AnalysisResult, truth: list[GroundTruth] | None = None) -> dict:
    measured = AnalysisResult(
        mode=result.mode, frame=result.frame,
        detections=[estimate(result.frame, item) for item in result.detections],
    )
    return {
        "analysis": measured.model_dump(mode="json"),
        "evaluation": evaluate(measured, truth) if truth is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One-frame AnalysisResult JSON")
    parser.add_argument("--truth", type=Path, help="CSV: track_id,width_cm,height_cm")
    parser.add_argument(
        "--output", type=Path, help="New JSON report path; existing files are refused",
    )
    args = parser.parse_args()
    try:
        result = AnalysisResult.model_validate_json(args.input.read_text(encoding="utf-8"))
        report = run_experiment(result, read_truth(args.truth) if args.truth else None)
        output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(output)
        else:
            print(output, end="")
    except (OSError, ValueError, ValidationError) as exc:
        parser.exit(1, f"Measurement experiment failed: {exc}\n")


if __name__ == "__main__":
    main()
