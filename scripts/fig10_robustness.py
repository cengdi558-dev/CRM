"""Fig. 10: robustness across five degraded conditions.

Columns: condition label | Origin | CRM w/o COR | ResNet-50 | Swin-T | CRM
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from torchvision import transforms

from crm.model import load_checkpoint
from crm.regions import boxes_from_points

from scripts._common import MEAN, STD, SIZE, gradcam, ours_map

DATA = os.environ.get("RAFDB_DIR", ".rafdb")
OUT = "figs"
BASELINE_DIR = os.environ.get("BASELINE_DIR", "checkpoints/compare")


def degrade(name, im):
    if name == "Occlusion":
        im = im.copy()
        w, h = im.size
        ImageDraw.Draw(im).rectangle(
            [int(w * 0.18), int(h * 0.55), int(w * 0.82), int(h * 0.95)], fill=(25, 25, 25))
        return im
    if name == "Low resolution":
        w, h = im.size
        small = im.resize((max(8, w // 8), max(8, h // 8)), Image.BILINEAR)
        return small.resize((w, h), Image.NEAREST)
    if name == "Blur":
        return im.filter(ImageFilter.GaussianBlur(3.2))
    if name == "Rotation":
        return im.rotate(24, resample=Image.BILINEAR, expand=False)
    if name == "Low light":
        return Image.fromarray((np.asarray(im).astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8))
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    conditions = ["Occlusion", "Low resolution", "Blur", "Rotation", "Low light"]
    root = os.path.join(DATA, "val")
    classes = [c for c in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, c))]
    base = []
    for ci, c in enumerate(classes):
        folder = os.path.join(root, c)
        files = sorted(os.listdir(folder))[20:80]
        for f in files:
            p = os.path.join(folder, f)
            if len(base) >= len(conditions):
                break
            im = Image.open(p).convert("RGB")
            arr = np.asarray(im.convert("L"), dtype=np.float32)
            if arr.mean() >= 70 and arr.std() >= 25:
                base.append(ImageOps.fit(im, (SIZE, SIZE)))
        if len(base) >= len(conditions):
            break
    while len(base) < len(conditions):
        base.append(ImageOps.fit(Image.open(os.path.join(root, classes[0], sorted(
            os.listdir(os.path.join(root, classes[0])))[0])).convert("RGB"), (SIZE, SIZE)))

    crm, cfg = load_checkpoint(os.environ.get("CRM_CKPT", "checkpoints/crm_best.pth"), "cpu")
    crm.eval()
    noc, _ = load_checkpoint(os.environ.get("CRM_NOCOC_CKPT", "checkpoints/crm_nococ.pth"), "cpu")
    noc.eval()
    boxes = np.asarray(boxes_from_points(None))

    tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    nets = {}
    for name in ("resnet50", "swin_tiny_patch4_window7_224"):
        ck = torch.load(os.path.join(BASELINE_DIR, f"{name}.pth"), map_location="cpu")
        m = timm.create_model(name, pretrained=False, num_classes=len(ck["classes"]))
        m.load_state_dict(ck["model"])
        m.eval()
        nets[name] = m

    rows = []
    for cond, im in zip(conditions, base):
        degraded = degrade(cond, im)
        x = tf(degraded).unsqueeze(0).requires_grad_(True)
        cells = {"Origin": degraded}
        for name, m in nets.items():
            for p in m.parameters():
                p.grad = None
            with torch.no_grad():
                pred = int(m(x).argmax(-1))
            cells[name] = gradcam(m, x, pred)
        cells["CRM w/o COR"] = ours_map(noc, degraded)
        cells["CRM"] = ours_map(crm, degraded)
        rows.append((cond, degraded, cells))
        print("row done:", cond, flush=True)

    cols = ["Origin", "CRM w/o COR", "ResNet-50", "Swin-T", "CRM"]
    keys = ["Origin", "CRM w/o COR", "resnet50", "swin_tiny_patch4_window7_224", "CRM"]
    n_rows, n_cols = len(rows), 1 + len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.7 * n_rows))
    axes = np.atleast_2d(axes)
    for r, (cond, _img, cells) in enumerate(rows):
        lab = axes[r, 0]
        lab.text(0.82, 0.5, cond, ha="right", va="center", fontsize=11, fontfamily="serif")
        lab.axis("off")
        for c, (title, key) in enumerate(zip(cols, keys), start=1):
            ax = axes[r, c]
            arr = np.asarray(cells["Origin"].resize((SIZE, SIZE)))
            ax.imshow(arr)
            if key != "Origin":
                ax.imshow(cells[key], cmap="jet", alpha=0.55, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    for c, title in enumerate([""] + cols):
        if c:
            axes[0, c].set_title(title, fontsize=11, fontfamily="serif", pad=6)
    fig.tight_layout()
    path = os.path.join(OUT, "fig10_final.png")
    fig.savefig(path, dpi=300)
    print("saved", path)


if __name__ == "__main__":
    main()
