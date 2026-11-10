"""Fig. 9, rebuilt locally: 10 rows, five Grad-CAM baselines, one OURS column."""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

from crm.model import load_checkpoint
from crm.regions import boxes_from_points

from scripts._common import MEAN, STD, SIZE, gradcam, ours_map

SHORT = {
    "resnet50": "ResNet-50",
    "vit_base_patch16_224": "ViT-B/16",
    "swin_tiny_patch4_window7_224": "Swin-T",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-T",
    "OURS": "CRM",
}
WEIGHTS = os.environ.get("BASELINE_DIR", "checkpoints/compare")
CRM = os.environ.get("CRM_CKPT", "checkpoints/crm_best.pth")
DATA = os.environ.get("RAFDB_DIR", ".rafdb")


def pick_samples(boxes_off=0):
    root = os.path.join(DATA, "val")
    classes = [c for c in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, c))]
    picks = []
    # up to three distinct faces per class, then filter and keep the first ten
    for c in classes:
        folder = os.path.join(root, c)
        files = sorted(os.listdir(folder))
        for step in (0, 12, 24):
            idx = boxes_off + step
            if idx < len(files):
                picks.append((os.path.join(folder, files[idx]), c))
    good = []
    for path, c in picks:
        im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        if im.mean() >= 70 and im.std() >= 25:      # drop dark / flat frames
            good.append((path, c))
    good.sort(key=lambda item: classes.index(item[1]))
    return classes, good[:10]


def main():
    classes, samples = pick_samples(6)
    model_crm, cfg = load_checkpoint(CRM, "cpu")
    model_crm.eval()
    boxes = np.asarray(boxes_from_points(None))
    tf224 = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    names = list(SHORT)[:5]
    loaded = {}
    for name in names:
        ck = torch.load(os.path.join(WEIGHTS, f"{name}.pth"), map_location="cpu")
        m = timm.create_model(name, pretrained=False, num_classes=len(ck["classes"]))
        m.load_state_dict(ck["model"])
        m.eval()
        loaded[name] = m

    heats = {n: [] for n in names}
    heats["OURS"] = []
    for path, _c in samples:
        rgb = Image.open(path).convert("RGB")
        x = tf224(rgb).unsqueeze(0).requires_grad_(True)
        for name in names:
            m = loaded[name]
            for p in m.parameters():
                p.grad = None
            with torch.no_grad():
                pred = int(m(x).argmax(-1))
            heats[name].append(gradcam(m, x, pred))
        heats["OURS"].append(ours_map(model_crm, rgb))
        print("done", os.path.basename(path), flush=True)

    cols = names + ["OURS"]
    n_rows, n_cols = len(samples), 1 + len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))
    axes = np.atleast_2d(axes)
    for r, (path, _c) in enumerate(samples):
        arr = np.asarray(Image.open(path).convert("RGB").resize((SIZE, SIZE)))
        axes[r, 0].imshow(arr)
        for c, name in enumerate(cols, start=1):
            axes[r, c].imshow(arr)
            axes[r, c].imshow(heats[name][r], cmap="jet", alpha=0.55, vmin=0, vmax=1)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    for c, text in enumerate(["Origin"] + [SHORT[m] for m in cols]):
        axes[0, c].set_title(text, fontsize=11, fontfamily="serif", pad=6)
    fig.tight_layout()
    os.makedirs("figs-final", exist_ok=True)
    fig.savefig(os.path.join("figs-final", "fig9_final.png"), dpi=300)
    print("saved figs-final/fig9_final.png")


if __name__ == "__main__":
    main()

