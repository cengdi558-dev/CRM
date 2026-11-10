"""Estimate the concomitance matrix C offline (eq. 1).

C is the partial correlation of the eight regional responses on **unoccluded**
training faces, obtained by graphical lasso:

    Theta = argmin  tr(Sigma Theta) - log det Theta + rho * ||Theta||_1
    C_jk  = -Theta_jk / sqrt(Theta_jj * Theta_kk)

Run this once, keep the ``.npy``, and install it with ``CRM.load_prior``.
Requires scikit-learn only (no torch).

Usage:
    python -m crm.estimate_c --data datasets/RAF-DB/train --out c_rafdb.npy
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def regional_responses(paths, size: int = 112) -> np.ndarray:
    """Sobel-gradient responses of the eight regions for every image.

    Deliberately dependency-light: Pillow for decoding, numpy for the rest, so
    the script runs anywhere the rest of the pipeline does not.
    """
    from PIL import Image

    # reuse the same template boxes as crm.regions, in pixels
    from .regions import _TEMPLATE

    rows = []
    for p in paths:
        with Image.open(p) as im:
            g = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
        gy[1:-1, :] = g[2:, :] - g[:-2, :]
        grad = np.sqrt(gx * gx + gy * gy)

        resp = []
        for x0, y0, x1, y1 in _TEMPLATE:
            sub = grad[int(y0 * size):int(y1 * size), int(x0 * size):int(x1 * size)]
            resp.append(float(sub.mean()) if sub.size else 0.0)
        rows.append(resp)
    return np.asarray(rows, dtype=np.float64)


def estimate_c(responses: np.ndarray, alpha: float = 0.05, normalize: bool = True) -> np.ndarray:
    """Graphical-lasso partial correlation, then row-normalised (eq. 1)."""
    from sklearn.covariance import GraphicalLasso

    model = GraphicalLasso(alpha=alpha, max_iter=200, assume_centered=False)
    model.fit(responses)
    theta = model.precision_
    d = np.sqrt(np.diag(theta))
    c = -theta / np.outer(d, d)
    np.fill_diagonal(c, 0.0)
    if normalize:
        denom = np.abs(c).sum(axis=1, keepdims=True)
        c = c / np.clip(denom, 1e-8, None)
    return c.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of unoccluded training faces")
    ap.add_argument("--out", default="c.npy")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    for root, _, files in os.walk(args.data):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    paths.sort()
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images under {args.data}")

    print(f"images: {len(paths)}")
    resp = regional_responses(paths)
    print("response stats:", resp.mean(axis=0).round(4), resp.std(axis=0).round(4))

    c = estimate_c(resp, alpha=args.alpha)
    np.save(args.out, c)
    print(f"saved {args.out}  shape={c.shape}")
    print(np.round(c, 3))


if __name__ == "__main__":
    main()
