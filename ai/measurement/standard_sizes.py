"""Use known product sizes: snap beds to mattress standards, pull appliances to catalogs.

Bed sizes come in a handful of standards, so a bed measured from plain video is usually
closer to the truth when it is replaced by the most likely standard size. Width (the
longer floor side) is the bed's length and depth its width, as elsewhere in measurement.
Height varies with the frame and is left as measured.

Washers and refrigerators come in many models, but their sizes cluster. Their measured
size is replaced by the average of catalog models weighted by how well each matches the
measurement (all three dimensions together), which pulls a noisy measurement toward
sizes that real products have.

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
APPLIANCE_QUALIFIER = "appliance_catalog_prior"

# Official sizes (W x D x H, mm) of models sold in Korea, from lge.co.kr, samsung.com and
# retailer spec tables (checked 2026-11). Sources are listed in docs/appliance-sizes.md.
APPLIANCES_MM = {
    "washer": {
        "lg_tr10wl": (590, 606, 960), "lg_tr10wv4": (590, 610, 965),
        "lg_f10src": (600, 615, 850), "lg_f9wp": (600, 550, 850),
        "lg_fr9wpb": (600, 550, 825), "samsung_ww90t3000kw": (595, 565, 855),
        "samsung_wa10cg5441bw": (610, 675, 1059), "samsung_wa10j5710sw1": (610, 675, 1075),
        "samsung_wa16cg6441by": (637, 701, 1093), "samsung_wa16n6780bg": (637, 701, 1095),
        "samsung_wd16j7800ks": (686, 787, 990), "samsung_wf24b9600kv": (686, 850, 984),
        "lg_f21vdu": (700, 770, 990), "lg_f21vbtm": (700, 835, 990),
    },
    "refrigerator": {
        "lg_90l_bar": (463, 500, 820), "lg_b182w13": (555, 585, 1400),
        "lg_b245w22": (555, 637, 1573), "lg_d312mbe31": (600, 710, 1640),
        "samsung_rb30d4051s9": (595, 663, 1700), "lg_q336mee151": (597, 674, 1860),
        "lg_q343mhhf33": (595, 676, 1860), "lg_q405mee151": (697, 696, 1860),
        "lg_d605mps52": (831, 741, 1860), "lg_ls19gbbdi": (913, 743, 1790),
        "lg_gs_j5982ms": (913, 735, 1790), "lg_m875aaa031": (914, 918, 1860),
        "lg_gr_x33fgngl": (912, 929, 1784),
    },
}
# Spread of log(calibrated / true) height on development scene videos (rgb, arkit).
HEIGHT_SIGMA = {"rgb": 0.23, "arkit": 0.20}
# Width of each catalog model's bump in log space, so neighbouring models blend smoothly.
CATALOG_BANDWIDTH = 0.05


def catalog_cm(label: str, exclude: str | None = None) -> list[dict[str, float]]:
    """Catalog sizes in measurement axes: width = longer floor side, depth = shorter."""
    sizes = []
    for model, (w, d, h) in APPLIANCES_MM.get(label, {}).items():
        if model != exclude:
            sizes.append({"width": max(w, d) / 10, "depth": min(w, d) / 10, "height": h / 10})
    return sizes


def pull_to_catalog(estimate: dict[str, float], label: str, kind: str = "rgb",
                    exclude: str | None = None) -> dict[str, float] | None:
    """Catalog average weighted by agreement with the measured dimensions (log space)."""
    catalog = catalog_cm(label, exclude)
    if not catalog or not estimate:
        return None
    sigma = {**SIGMA[kind], "height": HEIGHT_SIGMA[kind]}
    spread = {axis: math.hypot(sigma[axis], CATALOG_BANDWIDTH) for axis in estimate}
    scores = [-0.5 * sum((math.log(estimate[a] / size[a]) / spread[a]) ** 2 for a in estimate)
              for size in catalog]
    top = max(scores)
    weights = [math.exp(score - top) for score in scores]
    total = sum(weights)
    return {axis: math.exp(sum(w * math.log(size[axis]) for w, size in zip(weights, catalog))
                           / total) for axis in estimate}


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


def _pull_track(track, kind: str):
    label = canonical_label(track.label)
    dims = {axis: track.dimensions.get(axis) for axis in ("width", "depth", "height")}
    measured = {axis: d.cm for axis, d in dims.items() if d is not None}
    pulled = pull_to_catalog(measured, label, kind)
    if pulled is None:
        return track
    dimensions = dict(track.dimensions)
    for axis, cm in pulled.items():
        dimensions[axis] = dims[axis].model_copy(update={"cm": round(cm, 1)})
    qualifiers = sorted(set(track.qualifiers) | {APPLIANCE_QUALIFIER})
    return track.model_copy(update={"dimensions": dimensions, "qualifiers": qualifiers})


def snap(response: MeasurementResponse, region: str = "kr") -> MeasurementResponse:
    """Snap beds to a standard and pull appliances to catalog sizes; others unchanged."""
    videos = []
    for clip in response.videos:
        kind = KINDS.get(clip.pose_source, "rgb")
        objects = []
        for track in clip.objects:
            label = canonical_label(track.label)
            if label in APPLIANCES_MM:
                objects.append(_pull_track(track, kind))
                continue
            if label != "bed":
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
