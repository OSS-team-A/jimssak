"""Train a width-anchored geometric correction on annotated video views."""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEO = "https://www.youtube.com/watch?v="
CENTER = (490.0, 275.0)
AXES = ("depth", "height")

# Per frame: media ID, seconds, top FL/FR/BR/BL, observed foot by top-corner index.
# Points are in pixels on 980 × 551 playback crops.
PRODUCTS = {
    "LACK": {
        "cm": (55, 55, 45),
        "catalog": "https://www.ikea.com/kr/ko/p/lack-side-table-white-10449909/",
        "views": [
            ("ynasFcWWafg", 92.527, [[234, 112], [461, 349], [888, 183], [581, 20]],
             [[0, 305, 385], [2, 765, 460]]),
            ("ynasFcWWafg", 93.360, [[186, 165], [537, 480], [919, 195], [554, 34]],
             [[0, 276, 458], [2, 791, 484]]),
            ("DwKD3YwDZrw", 60.450, [[575, 287], [801, 334], [771, 210], [607, 177]],
             [[0, 561, 393], [1, 731, 433]]),
            ("DwKD3YwDZrw", 65.450, [[530, 327], [925, 421], [850, 171], [595, 133]],
             [[0, 532, 453], [1, 810, 545]]),
        ],
    },
    "KNARREVIK": {
        "cm": (37, 28, 45),
        "catalog": "https://www.ikea.com/kr/ko/p/knarrevik-bedside-table-black-80576319/",
        "views": [
            ("1WOHBAcVeOQ", 236.507, [[383, 124], [815, 106], [694, 85], [364, 92]],
             [[3, 385, 478], [2, 667, 445]]),
            ("1WOHBAcVeOQ", 237.373, [[373, 164], [830, 145], [698, 119], [354, 126]],
             [[3, 374, 527], [2, 673, 489]]),
        ],
    },
    "NESNA": {
        "cm": (36, 35, 45),
        "catalog": "https://www.ikea.com/kr/ko/p/nesna-bedside-table-bamboo-20247128/",
        "views": [
            ("0kexzFkS9bE", 1883.729, [[414, 233], [653, 382], [844, 205], [601, 127]],
             [[0, 470, 472]]),
            ("0kexzFkS9bE", 1884.162, [[445, 241], [636, 410], [848, 273], [634, 174]],
             [[0, 473, 484]]),
        ],
    },
}


def points(depth, height, feet):
    top = np.array([[0, 0, 0], [1, 0, 0], [1, 0, depth], [0, 0, depth]], dtype=float)
    bottom = top[np.asarray(feet, dtype=int)].copy()
    bottom[:, 1] = height
    return np.vstack((top, bottom))


def camera(focal):
    cx, cy = CENTER
    return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=float)


def prepare(views):
    sources = {video: index for index, video in enumerate(dict.fromkeys(v[0] for v in views))}
    observations = []
    for video, _, top, feet in views:
        marks = np.asarray(feet, dtype=float).reshape(-1, 3)
        target = np.vstack((top, marks[:, 1:]))
        weights = np.r_[np.full(4, 3.0), np.full(len(marks), 6.0)][:, None]
        observations.append((sources[video], target, marks[:, 0].astype(int), weights))
    return observations, len(sources)


def initial(observations, cameras, depth, height, focal):
    values = [depth, height] + [focal] * cameras
    for _, target, feet, _ in observations:
        ok, rotation, translation = cv2.solvePnP(
            points(depth, height, feet), target, camera(focal), None, flags=cv2.SOLVEPNP_SQPNP
        )
        if not ok:
            raise ValueError("Could not initialize camera pose")
        values.extend(np.r_[rotation.ravel(), translation.ravel()])
    return np.array(values)


def residual(parameters, observations, cameras):
    terms = []
    for index, (source, target, feet, weights) in enumerate(observations):
        start = 2 + cameras + 6 * index
        rotation, translation = parameters[start:start + 6].reshape(2, 3)
        intrinsics = camera(parameters[2 + source])
        projection, _ = cv2.projectPoints(
            points(*parameters[:2], feet), rotation, translation, intrinsics, None
        )
        terms.extend(((projection.reshape(-1, 2) - target) / weights).ravel())
    return np.array(terms)


def solve(params, observations, cameras):
    current = residual(params, observations, cameras)
    damping = 0.01
    step_sizes = np.maximum(np.abs(params) * 1e-5, 1e-6)
    for _ in range(100):
        jacobian = np.column_stack([
            (residual(params + step, observations, cameras) - current) / size
            for step, size in zip(np.diag(step_sizes), step_sizes)
        ])
        # Bound the influence of blurry corners and foot annotations.
        weights = np.minimum(1, 2.5 / np.maximum(np.abs(current), 1e-9))
        gram = jacobian.T @ (weights[:, None] * jacobian)
        delta = np.linalg.solve(
            gram + damping * np.diag(np.maximum(np.diag(gram), 1e-9)),
            -jacobian.T @ (weights * current),
        )
        trial = params + delta
        valid = (
            np.all((trial[:2] > 0.1) & (trial[:2] < 3))
            and np.all((trial[2:2 + cameras] > 270) & (trial[2:2 + cameras] < 1600))
        )
        if valid:
            candidate = residual(trial, observations, cameras)
            if loss(candidate) < loss(current):
                params, current = trial, candidate
                damping = max(damping / 3, 1e-8)
                if np.linalg.norm(delta) < 1e-6:
                    break
                continue
        damping *= 5
        if damping > 1e8:
            break
    return params, loss(current)


def loss(errors):
    magnitude = np.abs(errors)
    return float(np.sum(np.where(magnitude <= 2.5, 0.5 * magnitude**2, 2.5 * (magnitude - 1.25))))


def estimate(width_cm, views):
    observations, cameras = prepare(views)
    starts = ((0.45, 0.7, 440), (0.8, 1.0, 620), (1.15, 1.2, 900))
    params, cost = min(
        (solve(initial(observations, cameras, *start), observations, cameras) for start in starts),
        key=lambda fitted: fitted[1],
    )
    return params[:2] * width_cm, cost, params[2:2 + cameras].tolist()


def train_correction(rows):
    # Ridge regression with one log-scale intercept per axis. One row per SKU so
    # a product with many near-identical frames has the same weight as any other.
    offsets = np.stack([np.log(row["reference"] / row["geometry"]) for row in rows])
    return np.exp(offsets.sum(axis=0) / (len(rows) + 2))


def main():
    previous = json.loads((ROOT / "malm-reference.json").read_text())
    products = dict(PRODUCTS)
    products["MALM"] = {
        "cm": (
            previous["only_length_input_cm"],
            *(previous["comparison"][axis]["reference_cm"] for axis in AXES),
        ),
        "catalog": "https://www.ikea.com/fr/fr/p/malm-coiffeuse-blanc-10203610/",
        "views": [
            ("H_Se6AsZDIE", seconds, [marks[key][:2] for key in ("FL", "FR", "BR", "BL")],
             [[1, *marks["FOOT"][:2]]] if "FOOT" in marks else [])
            for seconds, marks, _ in previous["frames"]
        ],
    }
    rows = {}
    for product, record in products.items():
        width, *targets = record["cm"]
        prediction, cost, focals = estimate(width, record["views"])
        rows[product] = {
            "geometry": prediction, "reference": np.asarray(targets),
            "reprojection_loss": round(cost, 2), "focals_px": focals,
        }

    results = {}
    errors = []
    for held_out, row in rows.items():
        train_ids = [key for key in rows if key != held_out]
        scales = train_correction([rows[key] for key in train_ids])
        prediction = row["geometry"] * scales
        before = np.abs(row["geometry"] - row["reference"])
        after = np.abs(prediction - row["reference"])
        errors.append([before, after])
        results[held_out] = {
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in row.items()
            },
            "training_products": train_ids, "learned_scale": scales.tolist(),
            "predicted_cm": prediction.tolist(), "before_error_cm": before.tolist(),
            "after_error_cm": after.tolist(),
        }

    model = {"axes": AXES, "scale": train_correction(list(rows.values())).tolist(),
             "training_products": list(rows)}
    (ROOT / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    result = {
        "axes": AXES,
        "rows": results,
        "mae_before_after_cm": np.mean(errors, axis=0).tolist(),
        "data": products,
        "candidate_excluded": VIDEO + "jipET3dlj0w",
    }
    (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
