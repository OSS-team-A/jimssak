"""Learn a residual for rectangular-plane dimensions from simulated camera views."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
IMAGE_SIZE = np.array([1080.0, 1920.0])
IMAGE_CENTER = IMAGE_SIZE / 2
SEED = 824
EPOCHS = 40


def make_views(seed, count):
    rng = np.random.default_rng(seed)
    proposals = count * 3
    aspect = np.exp(rng.uniform(np.log(0.45), np.log(2.8), proposals))
    focal = rng.uniform(525, 1500, proposals)
    angles = np.radians(rng.uniform([-34, -56, -12], [34, 56, 12], (proposals, 3)))
    rotations = np.stack([cv2.Rodrigues(angles_i)[0] for angles_i in angles])
    corners = (SQUARE - 0.5)[None] * np.c_[np.ones(proposals), aspect][:, None]
    world = np.concatenate([corners, np.zeros((proposals, 4, 1))], axis=2)
    center = np.c_[
        rng.uniform(-0.95, 0.95, proposals),
        rng.uniform(-1.1, 1.1, proposals),
        rng.uniform(1.55, 5.4, proposals),
    ]
    camera = world @ rotations.transpose(0, 2, 1) + center[:, None]
    rays = camera[:, :, :2] / camera[:, :, 2:]
    radial = np.sum(rays**2, axis=2, keepdims=True)
    radial_curve = rng.uniform(-0.07, 0.035, (proposals, 1, 1))
    shifted_center = IMAGE_CENTER + rng.normal(0, [17, 23], (proposals, 2))
    pixels = (
        rays * (1 + radial_curve * radial) * focal[:, None, None]
        + shifted_center[:, None]
        + rng.normal(0, 3, (proposals, 4, 2))
    )
    hint = focal * np.exp(rng.normal(0, 0.13, proposals))
    visible = ((pixels > 4) & (pixels < IMAGE_SIZE - 4)).all(axis=(1, 2))
    big_enough = (np.ptp(pixels, axis=1) > [110, 130]).all(axis=1)
    edges = np.roll(pixels, -1, axis=1) - pixels
    following = np.roll(edges, -1, axis=1)
    crosses = edges[:, :, 0] * following[:, :, 1] - edges[:, :, 1] * following[:, :, 0]
    well_formed = (crosses > 20).all(axis=1) & (camera[:, :, 2] > 0.2).all(axis=1)
    valid = np.flatnonzero(visible & big_enough & well_formed)[:count]
    if len(valid) != count:
        raise RuntimeError("Insufficient fully visible synthetic views")
    return pixels[valid], hint[valid], aspect[valid]


def axis_features(top, focal_hint):
    corners = np.asarray(top, dtype=np.float32)
    if corners.shape != (4, 2) or not cv2.isContourConvex(corners) or focal_hint <= 0:
        raise ValueError("Enter four ordered convex corners and a positive focal estimate")
    h = cv2.getPerspectiveTransform(SQUARE, corners)
    xy = h[:2, :2] - IMAGE_CENTER[:, None] * h[2, :2]
    product = float(np.prod(h[2, :2]))
    focal_squared = -float(xy[:, 0] @ xy[:, 1]) / product if abs(product) > 1e-10 else -1
    focal_free = np.sqrt(focal_squared) if focal_squared > 0 else None

    def geometry(focal):
        axes = np.vstack((xy / focal, h[2, :2]))
        lengths = np.linalg.norm(axes, axis=0)
        cosine = float(axes[:, 0] @ axes[:, 1] / np.prod(lengths))
        return lengths[1] / lengths[0], cosine

    nominal, cosine = geometry(focal_hint)
    free = geometry(focal_free)[0] if focal_free else None
    focal_cond = np.clip(focal_free, 350, 2300) if focal_free else focal_hint
    free_clipped = geometry(focal_cond)[0]
    features = np.r_[
        ((corners - IMAGE_CENTER) / IMAGE_SIZE).ravel(),
        (
            np.log(focal_hint / 900),
            np.log(focal_cond / focal_hint),
            np.log(nominal),
            np.log(free_clipped / nominal),
            cosine,
            float(focal_free is not None),
        ),
    ]
    return features, float(nominal), free


def data(seed, count):
    corners, hints, ratios = make_views(seed, count)
    rows = [axis_features(points, f) for points, f in zip(corners, hints)]
    features = np.stack([row[0] for row in rows])
    nominal = np.array([row[1] for row in rows])
    free = np.array([row[2] if row[2] is not None else row[1] for row in rows])
    return features, nominal, free, ratios


def network(inputs):
    return nn.Sequential(
        nn.Linear(inputs, 48),
        nn.SiLU(),
        nn.Linear(48, 32),
        nn.SiLU(),
        nn.Linear(32, 1),
        nn.Flatten(0, 1),
    )


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def summarize(predicted, true):
    ratios = np.abs(predicted - true) / true * 100
    return {"mape_percent": float(ratios.mean()), "median_ape_percent": float(np.median(ratios))}


def train(real_corners=None):
    torch.manual_seed(SEED)
    splits = [data(SEED + offset, count) for offset, count in enumerate((24000, 3000, 3000))]
    x, initial, _, y = splits[0]
    val_x, val_initial, _, val_y = splits[1]
    mean, std = x.mean(axis=0), x.std(axis=0).clip(min=1e-4)
    train_x, targets = tensor((x - mean) / std), tensor(np.log(y / initial))
    val_x, val_targets = tensor((val_x - mean) / std), tensor(np.log(val_y / val_initial))
    model = network(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    criterion = nn.SmoothL1Loss(beta=0.05)
    best_loss, checkpoint = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in torch.randperm(len(train_x)).split(256):
            optimizer.zero_grad()
            loss = criterion(model(train_x[batch]), targets[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val = criterion(model(val_x), val_targets).item()
        if val < best_loss:
            best_loss = val
            checkpoint = {
                "state": {k: v.clone() for k, v in model.state_dict().items()},
                "mean": tensor(mean),
                "std": tensor(std),
                "epoch": epoch,
            }
    torch.save(checkpoint, ROOT / "length-model.pt")
    test_x, nominal, free, y = splits[2]
    model.load_state_dict(checkpoint["state"])
    results = {
        "synthetic_train_views": 24000,
        "synthetic_validation_views": 3000,
        "synthetic_test_views": 3000,
        "epochs": EPOCHS,
        "picked_epoch": checkpoint["epoch"],
        "synthetic_test": {
            "geometry_approx_focal": summarize(nominal, y),
            "geometry_self_calibrated_with_fallback": summarize(free, y),
            "learned": summarize(regress(model, checkpoint, test_x, nominal), y),
        },
    }
    if real_corners is not None:
        results.update(evaluate_real(model, checkpoint, real_corners))
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def evaluate_real(model, checkpoint, real_corners):
    rows = []
    for case in json.loads(real_corners.read_text()):
        inputs, initial, self_fit = axis_features(case["corners"], case["focal_hint_px"])
        estimate = regress(model, checkpoint, inputs[None], np.array([initial]))[0]
        rows.append(
            {
                **case,
                "geometry_self_calibrated_cm": (
                    self_fit * case["reference_width_cm"] if self_fit is not None else None
                ),
                "geometry_approx_focal_cm": initial * case["reference_width_cm"],
                "learned_cm": float(estimate * case["reference_width_cm"]),
            }
        )
    metrics = {}
    for key in ("geometry_self_calibrated_cm", "geometry_approx_focal_cm", "learned_cm"):
        errors = [abs(row[key] - row["catalog_height_cm"]) for row in rows if row[key] is not None]
        metrics[key] = float(np.mean(errors)) if errors else None
    return {"real_checks": rows, "real_mae_against_catalog_cm": metrics}


def regress(model, checkpoint, inputs, initial):
    features = (tensor(inputs) - checkpoint["mean"]) / checkpoint["std"]
    with torch.inference_mode():
        return initial * model(features).exp().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corners", type=Path, help="JSON array of TL, TR, BR, BL pixels")
    parser.add_argument("--real-corners", type=Path, help="Optional private evaluation fixture")
    parser.add_argument("--width-cm", type=float)
    parser.add_argument("--focal-px", type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if args.corners is None:
        train(args.real_corners)
        return
    if args.width_cm is None or args.width_cm <= 0 or args.focal_px is None:
        parser.error("provide a positive width and approximate focal length")
    inputs, initial, _ = axis_features(json.loads(args.corners.read_text()), args.focal_px)
    checkpoint = torch.load(ROOT / "length-model.pt", weights_only=True, map_location="cpu")
    model = network(len(inputs))
    model.load_state_dict(checkpoint["state"])
    model.eval()
    estimate = regress(model, checkpoint, inputs[None], np.array([initial]))[0]
    print(json.dumps({"height_cm_estimate": float(estimate * args.width_cm)}))


if __name__ == "__main__":
    main()
