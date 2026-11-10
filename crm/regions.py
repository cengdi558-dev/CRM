"""Eight functional facial regions (the N_A = 8 AU groups of Sect. III-B).

The paper partitions the face with a landmark detector into eight AU regions.
Landmark indices differ between detectors, so this module keeps the *geometry*
configurable: pass a detector's 68-point output and a mapping, or fall back to
the normalised template used when no detector is available.

Biologically the eight groups follow the AU families:
    0 brow-raise L   1 brow-raise R
    2 eye-widen L    3 eye-widen R
    4 nose/cheek     5 mouth-corner L
    6 mouth-corner R 7 chin/lower-lip
"""

from __future__ import annotations

import torch

AU_REGIONS = (
    "brow_L",
    "brow_R",
    "eye_L",
    "eye_R",
    "nose",
    "mouth_L",
    "mouth_R",
    "chin",
)

#: mirrored partner of every region (the bilateral pairing used by COC eq. 7)
MIRROR = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4, 5: 6, 6: 5, 7: 7}

#: normalised template boxes (x0, y0, x1, y1) in face-crop coordinates.
#: Used only when no landmark detector is wired in.
_TEMPLATE = (
    (0.18, 0.18, 0.46, 0.34),
    (0.54, 0.18, 0.82, 0.34),
    (0.20, 0.34, 0.45, 0.48),
    (0.55, 0.34, 0.80, 0.48),
    (0.40, 0.42, 0.60, 0.62),
    (0.26, 0.62, 0.48, 0.82),
    (0.52, 0.62, 0.74, 0.82),
    (0.38, 0.78, 0.62, 0.95),
)


def region_masks(
    height: int,
    width: int,
    device: torch.device | str = "cpu",
    landmarks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a ``[8, H, W]`` float mask, one soft box per AU region (eq. 3).

    ``landmarks`` is an optional ``[B, K, 2]`` tensor in ``[0, 1]``; when given it
    is averaged over the batch to derive the eight boxes, which keeps the masks
    aligned with the actual face instead of a fixed template.
    """
    if landmarks is not None and landmarks.numel() > 0:
        pts = landmarks.detach().to("cpu").mean(dim=0)  # [K, 2]
        boxes = _boxes_from_landmarks(pts)
    else:
        boxes = _TEMPLATE

    ys = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
    masks = []
    for x0, y0, x1, y1 in boxes:
        mx0, mx1 = x0 * width, x1 * width
        my0, my1 = y0 * height, y1 * height
        inside = (xs >= mx0) & (xs < mx1) & (ys >= my0) & (ys < my1)
        masks.append(inside.float())
    return torch.stack(masks, dim=0)


def boxes_from_points(points) -> list[tuple[float, float, float, float]]:
    """Eight AU boxes from *any* dense point cloud (landmarks) in ``[0, 1]``.

    The paper derives the boxes from a landmark detector, but detectors disagree
    on indices (68, 98, 468 points). Instead of hard-coding an index table this
    splits the detected face bounding box into the same vertical bands the
    muscle groups occupy, so any detector works.
    """
    if points is None:
        return list(_TEMPLATE)
    try:
        pts = points.detach().cpu().numpy() if hasattr(points, "detach") else points
    except Exception:
        return list(_TEMPLATE)
    import numpy as np

    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 8:
        return list(_TEMPLATE)

    pts = np.clip(pts, 0.0, 1.0)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)

    def box(fx0, fy0, fx1, fy1):
        return (
            float(x0 + fx0 * w), float(y0 + fy0 * h),
            float(x0 + fx1 * w), float(y0 + fy1 * h),
        )

    return [
        box(0.02, 0.00, 0.50, 0.28),   # brow L
        box(0.50, 0.00, 0.98, 0.28),   # brow R
        box(0.05, 0.24, 0.48, 0.46),   # eye L
        box(0.52, 0.24, 0.95, 0.46),   # eye R
        box(0.30, 0.40, 0.70, 0.66),   # nose
        box(0.06, 0.62, 0.48, 0.86),   # mouth L
        box(0.52, 0.62, 0.94, 0.86),   # mouth R
        box(0.28, 0.78, 0.72, 1.00),   # chin
    ]


def _boxes_from_landmarks(points: torch.Tensor):
    """Backwards-compatible wrapper around :func:`boxes_from_points`."""
    return boxes_from_points(points)


def assign_regions(
    grid_h: int,
    grid_w: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Map every token of a ``grid_h x grid_w`` feature map to a region id.

    Returns ``[grid_h * grid_w]`` int64; ``-1`` marks tokens that fall outside
    every AU box (background, hair), which receive no anatomical attention bias.
    """
    masks = region_masks(height, width)  # [8, H, W]
    ys = (torch.arange(grid_h, dtype=torch.float32) + 0.5) * (height / grid_h)
    xs = (torch.arange(grid_w, dtype=torch.float32) + 0.5) * (width / grid_w)
    yy, xx = torch.meshgrid(ys.long(), xs.long(), indexing="ij")
    hit = masks[:, yy.reshape(-1), xx.reshape(-1)]  # [8, N]
    ids = hit.argmax(dim=0)
    ids[hit.max(dim=0).values <= 0] = -1
    return ids
