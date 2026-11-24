"""Register moving RGB frames from feature matches and pretrained metric depth."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraPose:
    # World-to-camera pose within one connected portion of a video.
    rotation: np.ndarray
    translation: np.ndarray
    component: int
    parent_frame: int | None = None
    inliers: int = 0
    error_px: float | None = None
    parallax_deg: float = 0.0

    @property
    def center(self):
        return -self.rotation.T @ self.translation


def rays(pixels, camera):
    return np.linalg.solve(camera, np.c_[pixels, np.ones(len(pixels))].T).T


def backproject(pixels, depth, camera):
    return rays(pixels, camera) * depth[:, None]


def relative_pose(source, target, depths, camera):
    """Return source-to-target motion or None for insufficient reliable matches."""
    valid = np.isfinite(depths) & (depths > 0.15) & (depths < 20.0)
    source, target, depths = source[valid], target[valid], depths[valid]
    if len(source) < 20:
        return None
    world = backproject(source, depths, camera)
    ok, vector, translation, matches = cv2.solvePnPRansac(
        world, target, camera, None, iterationsCount=250, reprojectionError=2.5,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or matches is None or len(matches) < max(16, 0.35 * len(world)):
        return None
    selected = matches.ravel()
    span = np.ptp(target[selected], axis=0)
    if np.any(span < 0.06 * np.array([2 * camera[0, 2], 2 * camera[1, 2]])):
        return None
    vector, translation = cv2.solvePnPRefineLM(
        world[selected], target[selected], camera, None, vector, translation,
    )
    rotation = cv2.Rodrigues(vector)[0]
    projected = (world[selected] @ rotation.T + translation.ravel()) @ camera.T
    if np.any(projected[:, 2] <= 0):
        return None
    error = np.linalg.norm(
        projected[:, :2] / projected[:, 2:] - target[selected], axis=1
    )
    if np.median(error) > 2.5:
        return None
    before = rays(source[selected], camera) @ rotation.T
    after = rays(target[selected], camera)
    cosine = np.sum(before * after, axis=1) / (
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    )
    parallax = float(np.median(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))))
    shift = translation.ravel()
    if parallax < 0.6:
        # A turn in place does not provide a translation baseline for fusion.
        shift = np.zeros(3)
    return rotation, shift, len(selected), float(np.median(error)), parallax


class CameraTracker:
    def __init__(self, camera):
        self.camera = camera
        self.sift = cv2.SIFT_create(nfeatures=1800)
        self.matcher = cv2.BFMatcher()
        self.recent = []
        self.next_component = 0

    def _mutual(self, first, second):
        if first is None or second is None:
            return []

        def ratio(a, b):
            return {m.queryIdx: m.trainIdx for m, n in (
                pair for pair in self.matcher.knnMatch(a, b, k=2) if len(pair) == 2
            ) if m.distance < 0.74 * n.distance}

        reverse = ratio(second, first)
        return [(i, j) for i, j in ratio(first, second).items() if reverse.get(j) == i]

    def add(self, frame_id, image, depth):
        points, descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None
        )
        pixels = np.array([point.pt for point in points], dtype=float).reshape(-1, 2)
        x, y = np.rint(pixels).astype(int).T
        samples = depth[np.clip(y, 0, depth.shape[0] - 1), np.clip(x, 0, depth.shape[1] - 1)]
        pose = None
        for parent_id, parent, previous_pixels, previous_descriptors, previous_depth in reversed(
            self.recent
        ):
            pairs = self._mutual(previous_descriptors, descriptors)
            if len(pairs) < 20:
                continue
            source, target = np.asarray(pairs).T
            relative = relative_pose(
                previous_pixels[source], pixels[target], previous_depth[source], self.camera
            )
            if relative is None:
                continue
            rotation, shift, count, error, parallax = relative
            candidate = CameraPose(
                rotation @ parent.rotation, rotation @ parent.translation + shift,
                parent.component, parent_id, count, error, parallax,
            )
            if pose is None:
                pose = candidate
            if parallax >= 0.6 and pose.component == candidate.component:
                pose = candidate
                break
        if pose is None:
            pose = CameraPose(np.eye(3), np.zeros(3), self.next_component)
            self.next_component += 1
        self.recent.append((frame_id, pose, pixels, descriptors, samples))
        self.recent = self.recent[-3:]
        return pose
