"""Pull a noisy measurement toward the sizes that furniture of its kind usually has.

After calibration, plain-video estimates still scatter widely: on the development set the
log error has a standard deviation of about 0.3, and about half of it is a scale error
shared by all three dimensions of one object. Real furniture sizes of one kind cluster
(chairs are about 45-90 cm tall) and their dimensions are correlated, so the most likely
true size combines the measurement with that distribution.

The model, in log centimetres for one object with measured dimensions ``y``::

    true x ~ N(mu, S)             sizes of that kind (from ARKitScenes 3D boxes)
    y = x + s + e                 s ~ N(0, tau^2) shared, e_axis ~ N(0, sigma_axis^2)

and the estimate is the posterior mean ``mu + S (S + tau^2 J + D)^-1 (y - mu)``. A shared
error can then be corrected using the other dimensions, and a missing dimension is skipped.
Beds, washers and refrigerators are left to ``ai.measurement.standard_sizes``.

    python -m ai.measurement.size_prior fit outputs/arkitscenes-dev-objects.csv \
        outputs/dev-eval-200/truth.json outputs/dev-eval-200/dav2.json \
        --calibration ai/measurement/calibration/depth-anything-v2.json \
        --metadata outputs/arkit-depth-fix/metadata.csv \
        --output ai/measurement/calibration/size-prior-depth-anything-v2.json

Run it after ``ai.measurement.calibration apply``: the error model is that of calibrated
estimates. ``fit`` reports leave-one-room-out errors where the held-out room is excluded
from the calibration, the size distribution and the error model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from ai.measurement import calibration
from ai.measurement.moving_contract import MeasurementResponse
from ai.measurement.refine import canonical_label

AXES = ("width", "depth", "height")
MIN_OBJECTS = 20
# Sized from product standards and catalogs instead (ai.measurement.standard_sizes).
HANDLED_ELSEWHERE = {"bed", "washer", "refrigerator"}
QUALIFIER = "size_prior_applied"


def size_distributions(objects: list[dict], min_objects: int = MIN_OBJECTS) -> dict:
    """Mean and covariance of log sizes per label, from rows with ``label`` and ``*_cm``."""
    logs: dict[str, list[list[float]]] = defaultdict(list)
    for row in objects:
        sizes = [float(row[f"{axis}_cm"]) for axis in AXES]
        if min(sizes) > 0:
            logs[canonical_label(row["label"])].append([math.log(v) for v in sizes])
    return {
        label: {"objects": len(values), "mean": np.mean(values, axis=0).round(5).tolist(),
                "cov": np.cov(np.array(values).T).round(6).tolist()}
        for label, values in sorted(logs.items())
        if len(values) >= min_objects and label not in HANDLED_ELSEWHERE
    }


def _residuals(rows: list[dict], factors: dict) -> list[dict]:
    out = []
    for row in rows:
        factor = calibration.factor_for(factors, row["kind"], row["axis"], row["method"]) or 1.0
        out.append(dict(row, calibrated=row["estimate"] * factor,
                        residual=math.log(row["actual"] / (row["estimate"] * factor))))
    return out


def error_model(residuals: list[dict]) -> dict:
    """Shared (tau^2) and per-axis (sigma^2) log error variance per video kind."""
    model = {}
    for kind in sorted({r["kind"] for r in residuals}):
        rows = [r for r in residuals if r["kind"] == kind]
        tracks: dict[tuple, dict[str, float]] = defaultdict(dict)
        for r in rows:
            tracks[(r["video_id"], r["track_id"])][r["axis"]] = r["residual"]
        center = mean(r["residual"] for r in rows)
        products = [
            (values[a] - center) * (values[b] - center)
            for values in tracks.values()
            for i, a in enumerate(AXES) for b in AXES[i + 1:]
            if a in values and b in values
        ]
        tau2 = max(mean(products) if products else 0.0, 1e-4)
        sigma2 = {}
        for axis in AXES:
            values = [r["residual"] for r in rows if r["axis"] == axis]
            sigma2[axis] = max((float(np.var(values)) if values else 0.0) - tau2, 1e-3)
        model[kind] = {"tau2": round(tau2, 5), "sigma2": {a: round(v, 5) for a, v in
                                                          sigma2.items()}, "samples": len(rows)}
    return model


def posterior(measured: dict[str, float], distribution: dict, errors: dict) -> dict[str, float]:
    """Most likely true size (cm) for the measured axes of one object."""
    axes = [axis for axis in AXES if axis in measured]
    index = [AXES.index(axis) for axis in axes]
    mu = np.asarray(distribution["mean"])[index]
    cov = np.asarray(distribution["cov"])[np.ix_(index, index)]
    noise = errors["tau2"] * np.ones((len(axes), len(axes))) + np.diag(
        [errors["sigma2"][axis] for axis in axes])
    y = np.log([measured[axis] for axis in axes])
    x = mu + cov @ np.linalg.solve(cov + noise, y - mu)
    return {axis: float(value) for axis, value in zip(axes, np.exp(x))}


def apply(response: MeasurementResponse, prior: dict,
          skip: frozenset[str] = frozenset()) -> MeasurementResponse:
    """Pull toward the label's size distribution; labels in ``skip`` are left as they are."""
    videos = []
    for clip in response.videos:
        kind = calibration.KINDS[clip.pose_source]
        errors = prior["errors"].get(kind)
        if clip.scale.status == "reference_scaled" or errors is None:
            videos.append(clip)
            continue
        objects = []
        for track in clip.objects:
            label = canonical_label(track.label)
            distribution = None if label in skip else prior["sizes"].get(label)
            measured = {a: d.cm for a in AXES if (d := track.dimensions.get(a)) is not None}
            if distribution is None or not measured:
                objects.append(track)
                continue
            sizes = posterior(measured, distribution, errors)
            dimensions = dict(track.dimensions)
            for axis, cm in sizes.items():
                dimensions[axis] = dimensions[axis].model_copy(update={"cm": round(cm, 2)})
            objects.append(track.model_copy(update={
                "dimensions": dimensions,
                "qualifiers": sorted(set(track.qualifiers) | {QUALIFIER}),
            }))
        videos.append(clip.model_copy(update={"objects": objects}))
    return MeasurementResponse(videos=videos)


def cross_validate(objects: list[dict], rows: list[dict], rooms: dict[str, str],
                   scan_rooms: dict[str, str]) -> dict:
    """Leave-one-room-out: the room is left out of calibration, sizes and error model."""
    room = {id(r): calibration.room_of(r["video_id"], rooms) for r in rows}
    errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for held in sorted(set(room.values())):
        train = [r for r in rows if room[id(r)] != held]
        factors = calibration.fit(train)
        model = error_model(_residuals(train, factors))
        sizes = size_distributions(
            [o for o in objects if scan_rooms.get(o["scan"], f"scan-{o['scan']}") != held])
        tracks: dict[tuple, list[dict]] = defaultdict(list)
        for r in _residuals([r for r in rows if room[id(r)] == held], factors):
            tracks[(r["video_id"], r["track_id"], r["kind"], r["label"])].append(r)
        for (_, _, kind, label), items in tracks.items():
            measured = {r["axis"]: r["calibrated"] for r in items}
            adjusted = (posterior(measured, sizes[label], model[kind])
                        if label in sizes and kind in model else measured)
            for r in items:
                for name, value in (("calibrated", r["calibrated"]),
                                    ("prior", adjusted[r["axis"]])):
                    error = abs(value - r["actual"])
                    errors[kind][name].append(error)
                    errors[kind][f"{name}/{r['axis']}"].append(error)
                    errors[kind][f"{name}/within10"].append(float(error <= 0.1 * r["actual"]))
    return {"rooms": len(set(room.values())), "kinds": {
        kind: {"dimensions": len(v["calibrated"]),
               **{name: round(mean(x), 3) for name, x in sorted(v.items())}}
        for kind, v in sorted(errors.items())}}


def labelled_samples(response: MeasurementResponse, truth: dict) -> list[dict]:
    """Like ``calibration.samples``, with the track and its label kept on each row."""
    rows = []
    reference = {(t["video_id"], t["track_id"]): t["dimensions_cm"] for t in truth["tracks"]}
    for clip in response.videos:
        if clip.scale.status == "reference_scaled":
            continue
        for track in clip.objects:
            actual = reference.get((clip.video_id, track.track_id))
            if actual is None:
                continue
            for axis in AXES:
                estimate = track.dimensions.get(axis)
                if estimate is None or not actual.get(axis):
                    continue
                rows.append({
                    "video_id": clip.video_id, "track_id": track.track_id,
                    "kind": calibration.KINDS[clip.pose_source], "axis": axis,
                    "method": estimate.method, "estimate": estimate.cm,
                    "actual": actual[axis], "label": canonical_label(track.label),
                })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fit_cmd = sub.add_parser("fit", help="learn size distributions and the error model")
    fit_cmd.add_argument("objects", type=Path, help="arkitscenes_split objects CSV")
    fit_cmd.add_argument("truth", type=Path)
    fit_cmd.add_argument("results", type=Path, nargs="+")
    fit_cmd.add_argument("--calibration", type=Path, required=True)
    fit_cmd.add_argument("--metadata", type=Path, help="ARKitScenes metadata.csv for rooms")
    fit_cmd.add_argument("--output", type=Path, required=True)
    apply_cmd = sub.add_parser("apply", help="adjust a calibrated measurement result")
    apply_cmd.add_argument("result", type=Path)
    apply_cmd.add_argument("--prior", type=Path, required=True)
    apply_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "apply":
        prior = json.loads(args.prior.read_text())
        response = MeasurementResponse.model_validate_json(args.result.read_text())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(apply(response, prior).model_dump_json(indent=2) + "\n")
        return 0

    with args.objects.open(newline="") as handle:
        objects = list(csv.DictReader(handle))
    truth = json.loads(args.truth.read_text())
    rows = []
    for path in args.results:
        rows += labelled_samples(MeasurementResponse.model_validate_json(path.read_text()), truth)
    factors = json.loads(args.calibration.read_text())
    rooms = calibration.rooms_from_metadata(args.metadata) if args.metadata else {}
    prior = {
        "version": 1,
        "sizes": size_distributions(objects),
        "errors": error_model(_residuals(rows, factors)),
        "leave_one_room_out": cross_validate(objects, rows, rooms, rooms),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prior, indent=2) + "\n")
    print(json.dumps(prior["leave_one_room_out"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
