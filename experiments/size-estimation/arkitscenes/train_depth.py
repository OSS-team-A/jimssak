"""Learn furniture footprint-depth and height ratios from real scan image crops."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from collect import ROOT
from extract import CLASSES

ALPHAS = (1.0, 4.0, 16.0, 64.0, 256.0)


def upright(patch, gravity):
    gx, gy = gravity
    if abs(gy) >= abs(gx):
        turns = 0 if gy < 0 else 2
    else:
        turns = 1 if gx > 0 else 3
    return np.rot90(patch, turns)


def featurize(image, bbox, gravity, width_cm, category):
    x1, y1, x2, y2 = bbox
    crop = upright(image[y1:y2, x1:x2], gravity)
    h, w = crop.shape[:2]
    if min(h, w) == 0:
        raise ValueError("The furniture crop is empty")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float) / 255
    patch = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA)
    patch -= patch.mean()
    gradients = [np.mean(abs(np.diff(gray, axis=axis))) for axis in (0, 1)]
    return np.r_[
        np.log(width_cm / 100),
        np.log(w / h),
        np.log(np.sqrt(w * h) / 64),
        patch.ravel(),
        gradients,
        [category == label for label in CLASSES],
    ]


def load_rows():
    rows = []
    for p in sorted((ROOT / "data").glob("*/*/samples.json")):
        rotations = {}
        for line in (p.parent / "lowres_wide.traj").read_text().splitlines():
            timestamp, *pose = map(float, line.split())
            rotations[timestamp] = cv2.Rodrigues(np.asarray(pose[:3]))[0][:2, 2]
        times = np.array(list(rotations))
        for row in json.loads(p.read_text()):
            image = cv2.imread(str(ROOT / row["image"]))
            if image is None or image.shape[1::-1] != tuple(row["size"]):
                continue
            timestamp = float(row["timestamp"])
            closest = times[np.argmin(abs(times - timestamp))]
            gravity = rotations[closest]
            if np.linalg.norm(gravity) < 0.35:
                continue
            width = row["dimensions_cm_long_short_height"][0]
            row["features"] = featurize(image, row["bbox"], gravity, width, row["label"])
            rows.append(row)
    return rows


def groups(rows):
    by_object = {}
    for index, row in enumerate(rows):
        by_object.setdefault((row["scan"], row["uid"]), []).append(index)
    return list(by_object.values())


def fit(rows, alpha):
    object_groups = groups(rows)
    unique = [rows[indexes[0]] for indexes in object_groups]
    ratios = np.array([r["dimensions_cm_long_short_height"] for r in unique])
    ratios = np.log(ratios[:, 1:] / ratios[:, :1])
    category_priors = {}
    for category in CLASSES:
        mask = [r["label"] == category for r in unique]
        category_priors[category] = np.median(ratios[mask] if any(mask) else ratios, axis=0)
    x = np.stack([row["features"] for row in rows])
    dims = np.array([row["dimensions_cm_long_short_height"] for row in rows])
    target = np.log(dims[:, 1:] / dims[:, :1])
    baseline = np.stack([category_priors[row["label"]] for row in rows])
    weights = np.ones(len(rows))
    for indexes in object_groups:
        weights[indexes] = 1.0 / len(indexes)
    mean = np.average(x, axis=0, weights=weights)
    std = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)).clip(min=0.02)
    z = np.c_[np.ones(len(x)), (x - mean) / std]
    penalty = np.diag([0, *([alpha] * x.shape[1])])
    coefficients = np.linalg.solve(
        z.T @ (weights[:, None] * z) + penalty, z.T @ (weights[:, None] * (target - baseline))
    )
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coefficients": coefficients.tolist(),
        "priors": {key: value.tolist() for key, value in category_priors.items()},
    }


def evaluate(rows, model):
    x = np.stack([row["features"] for row in rows])
    z = np.c_[np.ones(len(x)), (x - model["mean"]) / model["std"]]
    base = np.array([model["priors"][row["label"]] for row in rows])
    pred = np.exp(base + z @ model["coefficients"])
    output = []
    for indices in groups(rows):
        row = rows[indices[0]]
        width, depth, height = row["dimensions_cm_long_short_height"]
        base_depth, base_height = np.exp(base[indices[0]]) * width
        est_depth, est_height = np.median(pred[indices], axis=0) * width
        output.append(
            {
                "scan": row["scan"],
                "object": row["uid"],
                "category": row["label"],
                "views": len(indices),
                "width_input_cm": width,
                "depth_label_cm": depth,
                "height_label_cm": height,
                "depth_baseline_cm": base_depth,
                "height_baseline_cm": base_height,
                "depth_learned_cm": est_depth,
                "height_learned_cm": est_height,
            }
        )
    return output


def mae(predictions, method):
    return {
        axis: float(
            np.mean([abs(p[f"{axis}_{method}_cm"] - p[f"{axis}_label_cm"]) for p in predictions])
        )
        for axis in ("depth", "height")
    }


def train():
    cv2.setNumThreads(1)
    rows = load_rows()
    train = [r for r in rows if r["split"] == "Training"]
    test = [r for r in rows if r["split"] == "Validation"]
    scans = sorted({r["scan"] for r in train})
    if len(scans) < 3 or not test:
        raise ValueError("Insufficient distinct training scans or no held-out scan")
    choices = []
    for alpha in ALPHAS:
        predictions = []
        for scan in scans:
            model = fit([r for r in train if r["scan"] != scan], alpha)
            predictions.extend(evaluate([r for r in train if r["scan"] == scan], model))
        choices.append((np.mean(list(mae(predictions, "learned").values())), alpha))
    _, alpha = min(choices)
    model = fit(train, alpha)
    predictions = evaluate(test, model)
    result = {
        "training": {
            "scans": scans,
            "objects": len(groups(train)),
            "views": len(train),
            "unique_frames": len({r["image"] for r in train}),
        },
        "held_out": {
            "scans": sorted({r["scan"] for r in test}),
            "objects": len(groups(test)),
            "views": len(test),
            "unique_frames": len({r["image"] for r in test}),
        },
        "axis_convention": "Long horizontal input, short horizontal extent (depth proxy), height",
        "alpha_selected_within_training_scans": alpha,
        "category_median_baseline_mae_cm": mae(predictions, "baseline"),
        "learned_mae_cm": mae(predictions, "learned"),
    }
    (ROOT / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "held-out-predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--width-cm", type=float)
    parser.add_argument("--category", choices=CLASSES)
    parser.add_argument(
        "--rotate",
        type=int,
        choices=range(4),
        default=0,
        help="quarter turns counterclockwise needed to make the crop upright",
    )
    args = parser.parse_args()
    if args.image is None:
        train()
        return
    if args.bbox is None or args.width_cm is None or args.width_cm <= 0 or args.category is None:
        parser.error("inference needs an image, bounding box, positive width and category")
    im = cv2.imread(str(args.image))
    if im is None or im.shape[:2] != (192, 256):
        parser.error("this pilot expects a decoded 256 × 192 frame")
    x1, y1, x2, y2 = args.bbox
    if not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 192):
        parser.error("bounding box must lie in the image")
    gravity = [(0, -1), (1, 0), (0, 1), (-1, 0)][args.rotate]
    feature = featurize(im, args.bbox, gravity, args.width_cm, args.category)
    model = json.loads((ROOT / "model.json").read_text())
    z = np.r_[1, (feature - model["mean"]) / model["std"]]
    depth, height = np.exp(model["priors"][args.category] + z @ model["coefficients"])
    print(
        json.dumps(
            {
                "width_input_cm": args.width_cm,
                "depth_estimate_cm": float(depth * args.width_cm),
                "height_estimate_cm": float(height * args.width_cm),
            }
        )
    )


if __name__ == "__main__":
    main()
