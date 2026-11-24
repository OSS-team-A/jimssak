"""Select tabletop and floor-contact observations from an aligned furniture mask."""

import numpy as np


def tabletop_and_contacts(mask, pixels):
    """Return Boolean selectors for shared upright image pixels.

    The upper silhouette band represents the tabletop rim. The floor contacts
    are points on the lowest parts of the per-column silhouette envelope.
    """
    y, x = np.where(mask)
    if len(y) == 0:
        raise ValueError("Empty furniture mask")
    low, high = np.percentile(y, [1, 99])
    height = high - low
    if height < 12:
        raise ValueError("Furniture is too small for tabletop and foot measurements")

    px, py = pixels.T
    bottom = np.full(mask.shape[1], -1)
    top = np.full(mask.shape[1], mask.shape[0])
    np.maximum.at(bottom, x, y)
    np.minimum.at(top, x, y)
    tolerance = max(2.0, 0.04 * height)
    tabletop = (py <= low + 0.26 * height) & (py <= top[px] + 2 * tolerance)
    contacts = (bottom[px] >= low + 0.84 * height) & (py >= bottom[px] - tolerance)
    if tabletop.sum() < 8 or contacts.sum() < 8:
        raise ValueError("Too few shared tabletop or floor-contact observations")
    return tabletop, contacts


def vertical_height(top_xyz, foot_xyz):
    """Measure the separation of two independently estimated world-Z levels."""
    top = np.median(top_xyz[:, 2])
    floor = np.median(foot_xyz[:, 2])
    height = float((top - floor) * 100)
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Tabletop has no reliable positive separation from the floor")
    return height
