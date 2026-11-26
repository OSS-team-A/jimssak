"""Snap measured beds to the nearest standard mattress size.

Bed sizes come in a handful of standards, so a bed measured from plain video is usually
closer to the truth when it is replaced by the most likely standard size. Width (the
longer floor side) is the bed's length and depth its width, as elsewhere in measurement.
Height varies with the frame and is left as measured.

    python -m ai.measurement.standard_sizes apply outputs/result-calibrated.json \
        --region kr --output outputs/result-beds.json

Run it after ``ai.measurement.calibration apply``: the error spread below is that of
calibrated estimates. A bed that is far from every standard keeps its measured size and
is marked ``bed_size_nonstandard`` for the user to check.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ai.measurement.moving_contract import MeasurementResponse
from ai.measurement.refine import canonical_label

# (key, Korean name, mattress width cm, mattress length cm)
STANDARDS = {
    "kr": (
        ("single", "싱글", 100, 200), ("super_single", "슈퍼싱글", 110, 200),
        ("double", "더블", 140, 200), ("queen", "퀸", 150, 200),
        ("king", "킹", 160, 200), ("large_king", "라지킹", 180, 200),
    ),
    # ARKitScenes rooms are mostly in the US; used for development evaluation.
    "us": (
        ("twin", "트윈", 97, 191), ("full", "풀", 137, 191), ("queen", "퀸", 152, 203),
        ("california_king", "캘리포니아 킹", 183, 213), ("king", "킹", 193, 203),
    ),
}
# Outer size beyond the mattress (frame, headboard), median over 119 development beds
# matched to the nearest US standard: none on the width, 8 cm on the length.
ALLOWANCE_CM = {"width": 0.0, "length": 8.0}
# Spread of log(calibrated / true) per dimension on 26 development scene videos.
SIGMA = {"rgb": {"width": 0.26, "depth": 0.28}, "arkit": {"width": 0.28, "depth": 0.28}}
KINDS = {"capture_poses": "arkit", "rgb_metric_depth_pnp": "rgb"}
# Beyond this log ratio on any axis (about 1.8x, over 2 sigma) the bed is not standard.
MAX_LOG_RATIO = 0.6
QUALIFIER = "bed_standard_size"
NONSTANDARD = "bed_size_nonstandard"


def outer_size(standard: tuple) -> dict[str, float]:
    """Outer bed size for a standard: width axis = length, depth axis = mattress width."""
    _, _, width, length = standard
    return {"width": length + ALLOWANCE_CM["length"], "depth": width + ALLOWANCE_CM["width"]}


def match(estimate: dict[str, float], region: str = "kr", kind: str = "rgb") -> tuple | None:
    """Most likely standard for measured width/depth (cm), or None when none is close."""
    sigma = SIGMA[kind]
    best, best_score = None, math.inf
    for standard in STANDARDS[region]:
        size = outer_size(standard)
        ratios = {axis: math.log(estimate[axis] / size[axis]) for axis in estimate}
        if any(abs(r) > MAX_LOG_RATIO for r in ratios.values()):
            continue
        score = sum((r / sigma[axis]) ** 2 for axis, r in ratios.items())
        if score < best_score:
            best, best_score = standard, score
    return best


def snap(response: MeasurementResponse, region: str = "kr") -> MeasurementResponse:
    """Replace bed width/depth with the matched standard; other objects are unchanged."""
    videos = []
    for clip in response.videos:
        kind = KINDS.get(clip.pose_source, "rgb")
        objects = []
        for track in clip.objects:
            if canonical_label(track.label) != "bed":
                objects.append(track)
                continue
            dims = {axis: track.dimensions.get(axis) for axis in ("width", "depth")}
            measured = {axis: d.cm for axis, d in dims.items() if d is not None}
            if not measured:
                objects.append(track)
                continue
            standard = match(measured, region, kind)
            if standard is None:
                qualifiers = sorted(set(track.qualifiers) | {NONSTANDARD})
                objects.append(track.model_copy(update={"qualifiers": qualifiers}))
                continue
            size = outer_size(standard)
            dimensions = dict(track.dimensions)
            for axis in measured:
                dimensions[axis] = dims[axis].model_copy(update={"cm": size[axis]})
            qualifiers = sorted(set(track.qualifiers) | {QUALIFIER, f"{QUALIFIER}:{standard[0]}"})
            objects.append(track.model_copy(
                update={"dimensions": dimensions, "qualifiers": qualifiers}))
        videos.append(clip.model_copy(update={"objects": objects}))
    return MeasurementResponse(videos=videos)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    apply_cmd = sub.add_parser("apply", help="snap beds in a measurement result")
    apply_cmd.add_argument("result", type=Path)
    apply_cmd.add_argument("--region", choices=sorted(STANDARDS), default="kr")
    apply_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    response = MeasurementResponse.model_validate_json(args.result.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(snap(response, args.region).model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
