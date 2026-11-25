"""Sample metric sensor supervision and fine-tune the depth decoder output."""

import copy

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EPOCHS = 70
PIXELS_PER_VIEW = 192


def sensor_points(sensor, matrix, origin, box, rng):
    """Sample synchronized sensor depths on smooth surfaces inside the furniture box."""
    z = sensor.astype(np.float32) / 1000
    valid = (z > 0.1) & (z <= 20)
    kernel = np.ones((3, 3), np.uint8)
    # Reject depth discontinuities, where the neighboring pixels often include
    # the background, a hole in sensor coverage, or an occluding object.
    valid &= (cv2.dilate(z, kernel) - cv2.erode(z, kernel)) <= 0.12
    y, x = np.nonzero(valid)
    uv = np.c_[x, y, np.ones(len(x))]
    camera_depth = z[y, x]
    points = origin + np.linalg.solve(matrix[:, :3], uv.T).T * camera_depth[:, None]
    axes = np.asarray(box["normalizedAxes"]).reshape(3, 3).copy()
    if axes[2, 2] < 0:
        axes[2] *= -1
    local = (points - box["centroid"]) @ axes.T
    half_size = np.asarray(box["axesLengths"]) / 2
    inside = (np.abs(local) <= half_size + 0.035).all(axis=1)
    # Floor pixels are not furniture surface measurements, even when they fall
    # inside a scan's padded footprint near the lowest annotated plane.
    inside &= local[:, 2] > -half_size[2] + 0.025
    kept = np.flatnonzero(inside)
    if len(kept) < 24:
        return None
    ids = rng.choice(kept, min(PIXELS_PER_VIEW, len(kept)), replace=False)
    return x[ids], y[ids], camera_depth[ids]


def fit_depth_head(model, groups, tune_scans):
    """Fit the last depth layer, selecting an epoch using independent tuning scans."""
    train_groups = [views for (scan, _), views in groups.items() if scan not in tune_scans]
    tune_groups = [views for (scan, _), views in groups.items() if scan in tune_scans]
    if not train_groups or not tune_groups:
        raise ValueError("Missing independent training or tuning rooms")
    head = nn.Linear(model.head.conv3.in_channels, 1)
    with torch.no_grad():
        head.weight.copy_(model.head.conv3.weight.flatten(1))
        head.bias.copy_(model.head.conv3.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=4e-4, weight_decay=1e-4)
    original = [p.detach().clone() for p in head.parameters()]

    def unpack(items):
        views = [view for item in items for view in item]
        weight = [torch.full_like(y, 1 / (len(item) * len(y)))
                  for item in items for _, y in item]
        return (
            torch.cat([x for x, _ in views]),
            torch.cat([y for _, y in views]),
            torch.cat(weight),
        )

    train_x, train_y, train_w = unpack(train_groups)
    tune_x, tune_y, tune_w = unpack(tune_groups)

    def predict(x):
        return head(x).squeeze(-1).sigmoid() * model.head.max_depth

    def validation():
        with torch.no_grad():
            return float(((predict(tune_x) - tune_y).abs() * tune_w).sum() / tune_w.sum() * 100)

    start = validation()
    best, best_epoch, state = float("inf"), None, None
    curve = []
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(predict(train_x).log(), train_y.log(), beta=0.1, reduction="none")
        drift = sum(
            (param - first).square().sum() for param, first in zip(head.parameters(), original)
        )
        ((loss * train_w).sum() / train_w.sum() + 0.2 * drift).backward()
        optimizer.step()
        error = validation()
        curve.append(error)
        if error < best:
            best, best_epoch, state = error, epoch, copy.deepcopy(head.state_dict())
        if epoch in (1, 20, 40, 60, EPOCHS):
            print(f"Sensor tune {epoch}/{EPOCHS}: {error:.2f} cm", flush=True)
    head.load_state_dict(state)
    model.head.conv3.load_state_dict({"weight": head.weight.detach()[:, :, None, None],
                                    "bias": head.bias.detach()})
    return {
        "training_objects": len(train_groups), "tuning_objects": len(tune_groups),
        "training_pixels": len(train_y), "tuning_pixels": len(tune_y),
        "tune_sensor_mae_before_cm": start, "tune_sensor_mae_after_cm": best,
        "best_epoch": best_epoch, "tune_sensor_mae_by_epoch_cm": curve,
        "updated_weights": sum(
            int((param.detach() != first).sum())
            for param, first in zip(head.parameters(), original)
        ),
    }
