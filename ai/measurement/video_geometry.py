"""Analyze an annotated desk plane and test rotation-only motion on a video."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ai.training.video_inventory import capture, frames, sha256, write_image


def camera_matrix(focal, width, height):
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])


def pose(world, pixels, camera):
    _, rotations, translations, _ = cv2.solvePnPGeneric(
        world, pixels, camera, None, flags=cv2.SOLVEPNP_IPPE
    )
    candidates = []
    for rotation, translation in zip(rotations, translations):
        matrix, _ = cv2.Rodrigues(rotation)
        if (world @ matrix.T + translation.ravel())[:, 2].min() <= 0:
            continue
        projected, _ = cv2.projectPoints(world, rotation, translation, camera, None)
        rms = np.sqrt(np.mean(np.sum((projected[:, 0] - pixels) ** 2, axis=1)))
        center = -(matrix.T @ translation).ravel()
        candidates.append((float(rms), matrix, center))
    if not candidates:
        raise ValueError("No positive-depth planar pose")
    return min(candidates, key=lambda item: item[0])


def in_plane(pixels, camera, rotation, center, z):
    world_rays = (np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(camera).T) @ rotation
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (z - center[2]) / world_rays[:, 2]
        points = center + world_rays * distance[:, None]
    if (distance <= 0).any() or not np.isfinite(points).all():
        raise ValueError("Points do not intersect the plane ahead of the camera")
    return points


def planar_distances(reference, world, camera, rotation, center, floor_height):
    measurements = {}
    if "comparison_top" in reference:
        # The exact four-corner homography stays independent of the pinhole pose.
        homography = cv2.getPerspectiveTransform(
            reference["top"].astype(np.float32), world[:, :2].astype(np.float32)
        )
        comparison = reference["comparison_top"][None].astype(np.float32)
        plane = cv2.perspectiveTransform(comparison, homography)[0]
        lengths = np.linalg.norm(np.roll(plane, -1, axis=0) - plane, axis=1)
        measurements["other_top_width_cm"] = float(np.mean(lengths[::2]))
        measurements["other_top_depth_cm"] = float(np.mean(lengths[1::2]))
        measurements["other_top_opposite_edges_cm"] = lengths.tolist()
    if "floor_rail" in reference:
        floor_z = -np.sign(center[2]) * floor_height
        try:
            ends = in_plane(reference["floor_rail"], camera, rotation, center, floor_z)
        except ValueError:
            measurements["visible_floor_rail_cm"] = None
        else:
            measurements["visible_floor_rail_cm"] = float(np.linalg.norm(ends[0] - ends[1]))
    return measurements


def geometry(reference, world, camera, floor_height):
    rms, rotation, center = pose(world, reference["top"], camera)
    return {
        "reprojection_rms_px": rms,
        "camera_center_relative_to_table_cm": center.tolist(),
        "inferred_camera_height_cm": float(abs(center[2]) + floor_height),
        "camera_to_table_center_cm": float(np.linalg.norm(center - world.mean(axis=0))),
        **planar_distances(reference, world, camera, rotation, center, floor_height),
    }


def pixels_for(reference, scale):
    return {
        key: np.asarray(points, dtype=float) * scale
        for key, points in reference.items()
        if key in ("top", "comparison_top", "floor_rail")
    }


def audit_poses(references, world, camera, floor_height, jitter, rng):
    results = []
    views_by_unit = {}
    for item, reference in references:
        fit = geometry(reference, world, camera, floor_height)
        samples = []
        for _ in range(200):
            shifted = {
                key: points + rng.normal(0, jitter, size=points.shape)
                for key, points in reference.items()
            }
            try:
                samples.append(geometry(shifted, world, camera, floor_height))
            except ValueError:
                continue
        metrics = [
            "camera_to_table_center_cm",
            "inferred_camera_height_cm",
            "visible_floor_rail_cm",
            "other_top_width_cm",
            "other_top_depth_cm",
        ]
        ranges = {}
        for metric in metrics:
            numbers = [sample[metric] for sample in samples if sample.get(metric) is not None]
            if numbers:
                ranges[metric] = np.percentile(numbers, [5, 95]).tolist()
        sample_centers = np.array(
            [sample["camera_center_relative_to_table_cm"] for sample in samples]
        )
        views_by_unit.setdefault(item["unit"], []).append((item["frame"], fit, sample_centers))
        results.append(
            {
                "frame": item["frame"],
                "unit": item["unit"],
                **fit,
                "jitter_successes": len(samples),
                "jitter_5th_95th_percentile": ranges,
            }
        )
    baselines = []
    for unit, views in views_by_unit.items():
        if len(views) == 2:
            (first, one, first_samples), (second, two, second_samples) = views
            baseline = np.linalg.norm(
                np.subtract(
                    one["camera_center_relative_to_table_cm"],
                    two["camera_center_relative_to_table_cm"],
                )
            )
            count = min(len(first_samples), len(second_samples))
            sample_baselines = (
                np.linalg.norm(first_samples[:count] - second_samples[:count], axis=1)
                if count
                else None
            )
            baselines.append(
                {
                    "unit": unit,
                    "frames": [first, second],
                    "conditional_baseline_cm": float(baseline),
                    "jitter_5th_95th_percentile_cm": (
                        np.percentile(sample_baselines, [5, 95]).tolist() if count else None
                    ),
                }
            )
    return {"focal_px": float(camera[0, 0]), "views": results, "baselines": baselines}


def matches(first, second):
    sift = cv2.SIFT_create(nfeatures=3000)
    kp1, desc1 = sift.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), None)
    kp2, desc2 = sift.detectAndCompute(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), None)
    if desc1 is None or desc2 is None:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher()

    def unique(a, b):
        return {
            pair[0].queryIdx: pair[0].trainIdx
            for pair in matcher.knnMatch(a, b, k=2)
            if len(pair) == 2 and pair[0].distance < 0.74 * pair[1].distance
        }

    reverse = unique(desc2, desc1)
    mutual = [(a, b) for a, b in unique(desc1, desc2).items() if reverse.get(b) == a]
    if not mutual:
        return np.empty((0, 2)), np.empty((0, 2))
    return tuple(
        np.array([keypoints[pair[i]].pt for pair in mutual])
        for i, keypoints in enumerate((kp1, kp2))
    )


def unit_rays(points, camera):
    rays = np.c_[points, np.ones(len(points))] @ np.linalg.inv(camera).T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def projected_errors(matrix, first, second):
    predicted = cv2.perspectiveTransform(first[None], matrix)[0]
    return np.linalg.norm(predicted - second, axis=1)


def rotation_matrix(first, second, camera):
    one, two = unit_rays(first, camera), unit_rays(second, camera)
    keep = np.arange(len(first))
    for _ in range(4):
        u, _, vt = np.linalg.svd(one[keep].T @ two[keep])
        correction = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
        rotation = vt.T @ correction @ u.T
        h = camera @ rotation @ np.linalg.inv(camera)
        error = projected_errors(h, first, second)
        keep = np.argsort(error)[: max(6, int(0.75 * len(error)))]
    return rotation, h


def motion(first, second, cameras, rng):
    # Features are computed at half resolution, then measured in source pixels.
    a, b = matches(
        cv2.resize(first, None, fx=0.5, fy=0.5), cv2.resize(second, None, fx=0.5, fy=0.5)
    )
    count = len(a)
    if count < 30:
        return {"mutual_matches": count, "status": "insufficient_matches"}
    train, test = np.split(rng.permutation(count), [int(0.7 * count)])
    a, b = 2 * a, 2 * b
    h, _ = cv2.findHomography(a[train], b[train], cv2.RANSAC, 5)
    if h is None:
        return {"mutual_matches": count, "status": "homography_fit_failed"}

    def errors(matrix):
        return np.percentile(projected_errors(matrix, a[test], b[test]), [50, 90]).tolist()

    rotations = []
    for camera in cameras:
        rotation, h_rotation = rotation_matrix(a[train], b[train], camera)
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        rotations.append(
            {
                "focal_px": float(camera[0, 0]),
                "angle_deg": float(angle),
                "held_out_error_px_p50_p90": errors(h_rotation),
            }
        )
    return {
        "status": "evaluated",
        "mutual_matches": count,
        "held_out_matches": len(test),
        "free_homography_error_px_p50_p90": errors(h),
        "rotation": rotations,
    }


def annotate(image, reference):
    overlay = image.copy()
    for key, points in reference.items():
        color = {
            "top": (255, 230, 0),
            "comparison_top": (0, 210, 255),
            "floor_rail": (80, 255, 80),
        }[key]
        points = np.rint(points).astype(np.int32)
        cv2.polylines(overlay, [points], len(points) == 4, color, 3, cv2.LINE_AA)
        for index, point in enumerate(points, 1):
            cv2.circle(overlay, point, 7, color, -1)
            cv2.putText(overlay, f"{key}:{index}", point + [8, -7], 0, 0.55, (0, 0, 0), 4)
            cv2.putText(overlay, f"{key}:{index}", point + [8, -7], 0, 0.55, color, 1)
    return overlay


def run(source, settings, output):
    if settings.get("source_sha256") not in (None, sha256(source)):
        raise ValueError("Annotations were made for a different source video")
    targets = {ref["frame"] for ref in settings["references"]}
    targets.update(value for pair in settings["motion_pairs"] for value in pair)
    video = capture(source)
    images = {}
    try:
        for index, _, frame in frames(video, 1):
            if index in targets:
                images[index] = frame
            if len(images) == len(targets):
                break
    finally:
        video.release()
    if targets - images.keys():
        raise ValueError(f"Could not read frames {sorted(targets - images.keys())}")

    height, width = next(iter(images.values())).shape[:2]
    scale = np.array([width, height]) / settings["annotation_size_px"]
    sizes = settings["reference_cm"]
    w, d = sizes["top_width"], sizes["top_depth"]
    world = np.array([[0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0]], dtype=float)
    refs = [(ref, pixels_for(ref, scale)) for ref in settings["references"]]
    cameras = [
        camera_matrix(focal, width, height)
        for focal in settings["focal_candidates_px_at_source_size"]
    ]
    output.mkdir(parents=True)
    for item, ref in refs:
        index = item["frame"]
        write_image(output / f"reference_{index:06d}.jpg", annotate(images[index], ref))

    cv2.setRNGSeed(3898)
    rng = np.random.default_rng(3898)
    analysis = {
        "video_sha256": sha256(source),
        "reference_cm": sizes,
        "camera_assumptions": "Centered principal point, square pixels, no lens distortion",
        "corner_jitter_sigma_px": settings["jitter_px_at_source_size"],
        "poses": [
            audit_poses(
                refs,
                world,
                camera,
                sizes["top_to_floor"],
                settings["jitter_px_at_source_size"],
                rng,
            )
            for camera in cameras
        ],
        "motion_pairs": [
            {"frames": [a, b], **motion(images[a], images[b], cameras, rng)}
            for a, b in settings["motion_pairs"]
        ],
        "confirmed_3d_dimensions_cm": None,
    }
    (output / "geometry.json").write_text(json.dumps(analysis, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.video, json.loads(args.annotations.read_text()), args.output)


if __name__ == "__main__":
    main()
