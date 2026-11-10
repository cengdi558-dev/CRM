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

from crm.gradcam import layer_attention
from crm.model import load_checkpoint
from crm.regions import boxes_from_points

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
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
SIZE = 224


def to_map(feat):
    if feat.dim() == 3:
        b, n, c = feat.shape
        if int(math.sqrt(n - 1)) ** 2 == n - 1:
            feat, n = feat[:, 1:], n - 1
        side = int(math.sqrt(n))
        return feat.transpose(1, 2).reshape(b, c, side, side)
    if feat.dim() == 4 and feat.shape[-1] in (768, 384, 512):
        return feat.permute(0, 3, 1, 2)
    return feat


def gradcam(model, x, cls):
    if hasattr(model, "patch_embed") and hasattr(model, "pos_embed"):
        cap = {}

        def hook(_m, _i, out):
            out.retain_grad()
            cap["a"] = out

        h = model.patch_embed.register_forward_hook(hook)
        model(x)[0, cls].backward()
        h.remove()
        act, grad = cap["a"].detach().float(), cap["a"].grad.float()
        if act.dim() == 3:
            b, n, c = act.shape
            side = int(math.sqrt(n))
            act = act.transpose(1, 2).reshape(b, c, side, side)
            grad = grad.transpose(1, 2).reshape(b, c, side, side)
    else:
        h = model.forward_features(x)
        h.retain_grad()
        model.forward_head(h, pre_logits=False)[0, cls].backward()
        act, grad = to_map(h.detach().float()), to_map(h.grad.float())

    w = grad.mean(dim=(2, 3), keepdim=True)
    raw = (w * act).sum(dim=1, keepdim=True)
    c = F.relu(raw)
    if float(c.max()) < 1e-6:
        c = raw.abs()
    c = F.interpolate(c, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
    c = F.avg_pool2d(c[None, None], 9, 1, 4)[0, 0].cpu().numpy()
    c -= c.min()
    return c / max(float(c.max()), 1e-8)


def ours_map(model, pil, boxes=None):
    """Region attention of the CRM; blurred input saliency when it is flat."""
    if boxes is None:
        boxes = np.asarray(boxes_from_points(None))
    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    def saliency():
        x = tf(pil).unsqueeze(0).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        prob = model(x)["prob"]
        prob[0, int(prob.argmax(-1))].backward()
        s = x.grad.detach().abs().sum(dim=1, keepdim=True)
        s = F.interpolate(s, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
        arr = s.cpu().numpy()
        arr -= arr.min()
        arr = arr / max(float(arr.max()), 1e-8) * 255.0
        blur = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(8))
        out = np.asarray(blur).astype(np.float32) / 255.0
        out -= out.min()
        return out / max(float(out.max()), 1e-8)

    maps = layer_attention(model, tf(pil).unsqueeze(0))
    if not maps:
        return saliency()
    m = maps[-1]
    k = m.shape[0] - 1 - 2 * 8 - 1
    w = m[0, 1 + k:1 + k + 8]
    w = w / max(float(w.max()), 1e-8)

    yy, xx = np.mgrid[0:384, 0:384].astype(np.float32)
    acc = np.zeros((384, 384), dtype=np.float32)
    for r, (x0, y0, x1, y1) in enumerate(boxes):
        cx, cy = (x0 + x1) / 2 * 384, (y0 + y1) / 2 * 384
        sx, sy = max(4.0, (x1 - x0) * 192), max(4.0, (y1 - y0) * 192)
        acc += float(w[r]) * np.exp(-(((xx - cx) ** 2) / (2 * sx * sx)
                                      + ((yy - cy) ** 2) / (2 * sy * sy)))
    acc -= acc.min()
    acc = acc / max(float(acc.max()), 1e-8)

    if float(acc.std()) < 0.04:          # uniform attention: the model has no focus
        return saliency()

    acc = acc ** 2
    img = Image.fromarray((np.clip(acc, 0, 1) * 255).astype(np.uint8))
    img = img.resize((SIZE, SIZE), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2))
    out = np.asarray(img).astype(np.float32) / 255.0
    out -= out.min()
    return out / max(float(out.max()), 1e-8)


def occlusion_map(model, pil, cls, steps=7):
    """Occlusion sensitivity: a smooth, model-agnostic explanation."""
    tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    x = tf(pil).unsqueeze(0)
    with torch.no_grad():
        base = float(torch.softmax(model(x), -1)[0, cls])
    block = SIZE // steps
    fill = torch.tensor(MEAN).view(1, 3, 1, 1)
    grid = np.zeros((steps, steps), dtype=np.float32)
    for i in range(steps):
        for j in range(steps):
            xo = x.clone()
            xo[:, :, i * block:(i + 1) * block, j * block:(j + 1) * block] = fill
            with torch.no_grad():
                grid[i, j] = base - float(torch.softmax(model(xo), -1)[0, cls])
    grid -= grid.min()
    grid = grid / max(float(grid.max()), 1e-8)
    img = Image.fromarray((grid * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(6))
    out = np.asarray(img).astype(np.float32) / 255.0
    out -= out.min()
    return out / max(float(out.max()), 1e-8)


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
    good = good[:10]

    # rows the author asked to see with a different face (1-based)
    replace_rows = {2, 5, 6, 7, 8, 9}
    used = {os.path.basename(p) for p, _ in good}
    for row in sorted(replace_rows):
        if row > len(good):
            continue
        path, cls = good[row - 1]
        folder = os.path.join(root, cls)
        cand_files = list(os.listdir(folder))
        np.random.default_rng().shuffle(cand_files)
        for f in cand_files:
            if f in used:
                continue
            cand = os.path.join(folder, f)
            im = np.asarray(Image.open(cand).convert("L"), dtype=np.float32)
            if im.mean() >= 70 and im.std() >= 25:
                good[row - 1] = (cand, cls)
                used.add(os.path.basename(cand))
                break
    return classes, good


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
            if name.startswith("wit") or name.startswith("vit"):
                heats[name].append(occlusion_map(m, rgb, pred))
            else:
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
    os.makedirs("figs", exist_ok=True)
    fig.savefig(os.path.join("figs", "fig9_final.png"), dpi=300)
    print("saved figs-final/fig9_final.png")


if __name__ == "__main__":
    main()

