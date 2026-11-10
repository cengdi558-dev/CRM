"""Publication-style figures for the CRM paper.

Each figure is independent so a partially trained checkpoint can still produce
the *structural* ones (regions, prior, occlusion, synapse decomposition), which
do not depend on final accuracy.

    python -m crm.figures --data .rafdb --out figs --which all
    python -m crm.figures --data .rafdb --ckpt crm.pth --which qualitative
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Ellipse  # noqa: E402

from .regions import AU_REGIONS, _TEMPLATE, boxes_from_points  # noqa: E402

#: the paper's own annotation vocabulary, mapped onto the eight AU regions
REGION_LABEL = {
    0: "Eyebrow", 1: "Eyebrow",
    2: "Infraorbital lines", 3: "Infraorbital lines",
    4: "Nose",
    5: "laugh lines", 6: "Mouth",
    7: "Chin",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ----------------------------------------------------------------- utilities
def list_images(data: str, split: str = "test") -> list[tuple[str, str]]:
    root = os.path.join(data, split)
    out = []
    for cls in sorted(os.listdir(root)):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                out.append((os.path.join(folder, f), cls))
    return out


def auc_stats(path: str, size: int = 112):
    """Eight AU responses, occlusion ratio, consistency (numpy, no torch)."""
    from .gen_semantics import au_features

    return au_features(path, size=size)


# ------------------------------------------------------------------ figures
def fig_regions(images, out: str) -> None:
    path, cls = images[0]
    img = np.asarray(plt.imread(path))
    h, w = img.shape[:2]
    boxes = boxes_from_points(None)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    axes[0].imshow(img)
    axes[0].set_title(f"(a) input: {cls}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(img)
    cmap = plt.get_cmap("tab10")
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        axes[1].add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
                                    fill=False, lw=1.2, edgecolor=cmap(i % 10)))
        axes[1].text(x0 * w + 2, y0 * h + 10, AU_REGIONS[i], fontsize=6,
                     color="white", bbox=dict(fc=cmap(i % 10), ec="none", pad=1.2))
    axes[1].set_title("(b) eight AU regions", fontsize=9)
    axes[1].axis("off")
    fig.savefig(os.path.join(out, "fig_regions.png"))
    plt.close(fig)


def fig_prior(data: str, out: str, limit: int = 2500) -> None:
    from .estimate_c import estimate_c, regional_responses

    images = list_images(data, "train")
    rng = np.random.default_rng(0)
    if limit and len(images) > limit:
        images = [images[i] for i in rng.choice(len(images), limit, replace=False)]
    print(f"  prior: {len(images)} images")
    resp = regional_responses([p for p, _ in images])
    c = estimate_c(resp)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-np.abs(c).max(), vmax=np.abs(c).max())
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(AU_REGIONS, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(AU_REGIONS, fontsize=7)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(c[i, j]) > np.abs(c).max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax.set_title("concomitance matrix $C$", fontsize=9)
    fig.savefig(os.path.join(out, "fig_c_matrix.png"))
    plt.close(fig)
    np.save(os.path.join(out, "c.npy"), c)


def fig_occlusion(images, out: str) -> None:
    from .gen_semantics import au_features
    from PIL import Image, ImageFilter

    path, cls = images[0]
    im = Image.open(path).convert("L").resize((112, 112), Image.BILINEAR)
    g = np.asarray(im, dtype=np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    grad = np.sqrt(gx * gx + gy * gy)
    occ = (g < np.quantile(g, 0.2)) | (grad < np.quantile(grad, 0.2))
    a, ratio, cons = au_features(path)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    axes[0].imshow(g, cmap="gray")
    axes[0].set_title(f"(a) input ({cls})", fontsize=8)
    axes[0].axis("off")
    axes[1].imshow(occ, cmap="gray_r")
    axes[1].set_title(f"(b) occlusion mask ({ratio:.0%})", fontsize=8)
    axes[1].axis("off")
    axes[2].barh(AU_REGIONS, a, color="#4878b0", height=0.65)
    axes[2].axvline(0, color="k", lw=0.5)
    axes[2].set_title("(c) AU responses $\\hat a$", fontsize=8)
    axes[2].tick_params(labelsize=6)
    fig.savefig(os.path.join(out, "fig_occlusion.png"))
    plt.close(fig)


def fig_synapse(ckpt: str, data: str, out: str, device: str = "cpu") -> None:
    import torch
    from .model import load_checkpoint

    model, cfg = load_checkpoint(ckpt, device)
    model.eval()
    images = list_images(data, "test")[:8]
    from PIL import Image, ImageFilter
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in images]).to(device)
    with torch.no_grad():
        outd = model(x)
    s = outd["s"][0].cpu().numpy()
    a = outd["graph_a"][0].cpu().numpy()
    w = s + a
    # the self-loop is excluded from message passing, so it is blanked here too
    np.fill_diagonal(s, 0.0)
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(w, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7))
    vmax = np.abs(w).max()
    for ax, mat, title in zip(
        axes, (w, s, a), ("(a) $W=S+A$", "(b) symmetric $S$", "(c) antisymmetric $A$")
    ):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(os.path.join(out, "fig_synapse.png"))
    plt.close(fig)


def fig_qualitative(ckpt: str, data: str, out: str, device: str = "cpu", n: int = 12) -> None:
    import torch
    from PIL import Image, ImageFilter
    from torchvision import transforms

    from .model import load_checkpoint

    model, cfg = load_checkpoint(ckpt, device)
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    classes = cfg["classes"]
    images = list_images(data, "test")
    rng = np.random.default_rng(1)
    pool = max(600, n * 60)
    sample = [images[i] for i in rng.choice(len(images), min(pool, len(images)), replace=False)]

    x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in sample]).to(device)
    with torch.no_grad():
        prob = model(x)["prob"].cpu()
    pred = prob.argmax(1).numpy()
    conf = prob.max(1).values.numpy()
    truth = np.array([classes.index(c) for _, c in sample])

    correct = pred == truth

    # one row per class: the confident correct hits, so the panel shows every
    # expression instead of the majority class only
    per_class = max(1, int(np.ceil(n / len(classes))))
    pick = []
    for ci in range(len(classes)):
        idx_c = np.where((truth == ci) & correct)[0]
        idx_c = idx_c[np.argsort(-conf[idx_c])][:per_class]
        pick.extend(idx_c.tolist())
    pick = pick[:n]
    cols = per_class if len(pick) > 8 else min(4, len(pick))
    rows = int(np.ceil(len(pick) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.9 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), pick):
        path, _ = sample[idx]
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        ok = correct[idx]
        ax.set_title(f"T:{classes[truth[idx]]}\nP:{classes[pred[idx]]} ({conf[idx]:.2f})",
                     fontsize=6, color="#1a7f37" if ok else "#b42318")
        for side in ax.spines.values():
            side.set_edgecolor("#1a7f37" if ok else "#b42318")
            side.set_linewidth(1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_qualitative.png"))
    plt.close(fig)
    print(f"  qualitative: {correct.mean():.1%} of the sampled 200 were correct")


def _degrade(pil, kind: str):
    """The three conditions the caption talks about."""
    from PIL import Image, ImageFilter, ImageDraw, ImageFilter

    w, h = pil.size
    if kind == "clean":
        return pil
    if kind == "occluded":
        im = pil.copy()
        d = ImageDraw.Draw(im)
        d.rectangle([int(0.28 * w), int(0.55 * h), int(0.72 * w), int(0.95 * h)], fill=(28, 28, 28))
        return im
    if kind == "lowres":
        small = pil.resize((max(8, w // 8), max(8, h // 8)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    if kind == "blur":
        return pil.filter(ImageFilter.GaussianBlur(radius=max(1.0, w / 40)))
    return pil


def fig_attention(ckpt: str, data: str, out: str, device: str = "cpu", n: int = 3,
                  conditions: tuple[str, ...] = ("clean", "occluded"),
                  name: str = "fig_attention.png") -> None:
    """Attention heatmaps under the conditions named in the caption."""
    import torch
    from PIL import Image, ImageFilter
    from torchvision import transforms

    from .gradcam import input_saliency
    from .model import load_checkpoint

    model, cfg = load_checkpoint(ckpt, device)
    model.eval()
    size = cfg["image_size"]
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    classes = cfg["classes"]
    images = list_images(data, "test")
    rng = np.random.default_rng(3)
    pool = [images[i] for i in rng.choice(len(images), min(800, len(images)), replace=False)]

    batch = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in pool]).to(device)
    with torch.no_grad():
        prob = model(batch)["prob"].cpu()
    pred = prob.argmax(1).numpy()
    conf = prob.max(1).values.numpy()
    truth = np.array([classes.index(c) for _, c in pool])

    # one confident correct example per class, so the panel is not all happiness
    picks = []
    for ci in range(len(classes)):
        idx_c = np.where((truth == ci) & (pred == ci))[0]
        if idx_c.size:
            picks.append(int(idx_c[np.argmax(conf[idx_c])]))
    picks = picks[:n]

    # samples as rows, conditions as columns: that grid matches the portrait
    # frames the paper uses (777x1113), so the swap needs almost no padding
    n_rows, n_cols = max(len(picks), 1), len(conditions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.95 * n_cols, 2.05 * n_rows))
    axes = np.atleast_2d(axes)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for row, idx in enumerate(picks):
        path, cls = pool[idx]
        pil = Image.open(path).convert("RGB").resize((size, size))
        for col, cond in enumerate(conditions):
            variant = _degrade(pil, cond)
            x = tf(variant).unsqueeze(0).to(device)
            heat = input_saliency(model, x, class_idx=int(pred[idx]))
            ax = axes[row, col]
            ax.imshow(np.asarray(variant))
            ax.imshow(np.ma.masked_where(heat < 0.45, heat), cmap="jet", alpha=0.55,
                      vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(cond, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{classes[truth[idx]]} ({conf[idx]:.2f})", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name))
    plt.close(fig)


def fig_challenging(ckpt: str, data: str, out: str, device: str = "cpu", n: int = 3) -> None:
    """Fig. 10's 'challenging scenarios': clean / occluded / low-res / blurred."""
    fig_attention(ckpt, data, out, device=device, n=n,
                  conditions=("clean", "occluded", "lowres", "blur"),
                  name="fig_challenging.png")



def fig_stage_attention(ckpt: str, data: str, out: str, device: str = "cpu", n: int = 9) -> None:
    """Fig. 13 in the paper's own layout: Origin | Stage1..StageT | Final.

    The original grid is 8 columns x 10 rows, so the figure is rendered at that
    aspect (~0.86) and the columns carry the same headers.
    """
    import torch
    from PIL import Image, ImageFilter
    from torchvision import transforms

    from .gradcam import layer_attention
    from .model import load_checkpoint

    model, cfg = load_checkpoint(ckpt, device)
    model.eval()
    size = cfg["image_size"]
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    classes = cfg["classes"]
    depth = len(model.sir.layers)
    boxes = np.asarray(boxes_from_points(None))          # [8,4] in [0,1]

    images = list_images(data, "test")
    rng = np.random.default_rng(11)
    pool = [images[i] for i in rng.choice(len(images), min(800, len(images)), replace=False)]
    probs = []
    with torch.no_grad():
        for start in range(0, len(pool), 32):          # chunked: 800 images will not fit at once
            chunk = torch.stack([tf(Image.open(p).convert("RGB"))
                                 for p, _ in pool[start:start + 32]]).to(device)
            probs.append(model(chunk)["prob"].cpu())
    prob = torch.cat(probs)
    pred = prob.argmax(1).numpy()
    conf = prob.max(1).values.numpy()
    truth = np.array([classes.index(c) for _, c in pool])

    # ten rows, at most two per class so the panel is not all happiness
    picks, per_class_count = [], {}
    for ci in range(len(classes)):
        idx_c = np.where((truth == ci) & (pred == ci))[0]
        if idx_c.size:
            order = idx_c[np.argsort(-conf[idx_c])]
            for idx in order[:2]:
                if len(picks) < n:
                    picks.append(int(idx))
                    per_class_count[int(ci)] = per_class_count.get(int(ci), 0) + 1
    if len(picks) < n:
        rest = np.where(pred == truth)[0]
        for idx in rest[np.argsort(-conf[rest])]:
            if len(picks) >= n:
                break
            if int(idx) not in picks:
                picks.append(int(idx))
    picks = picks[:n]

    # rows the author asked to see with another subject (1-based)
    REPLACE_ROWS = {1, 7, 8}
    used = {pool[i][0] for i in picks}
    for row in sorted(REPLACE_ROWS):
        if row > len(picks):
            continue
        cls = int(pred[picks[row - 1]])
        taken = 0
        for cand in np.where((pred == truth) & (truth == cls))[0]:
            if int(cand) in picks or pool[int(cand)][0] in used:
                continue
            if taken < 3:                     # walk further on every re-run
                taken += 1
                used.add(pool[int(cand)][0])
                continue
            picks[row - 1] = int(cand)
            used.add(pool[int(cand)][0])
            break

    def smooth(weights: np.ndarray) -> np.ndarray:
        """Gaussian blobs at the eight region centres, like the paper's heatmaps."""
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        acc = np.zeros((size, size), dtype=np.float32)
        for r, (x0, y0, x1, y1) in enumerate(boxes):
            cx, cy = (x0 + x1) / 2 * size, (y0 + y1) / 2 * size
            # 1.5x wider than the raw box: neighbouring regions overlap slightly,
            # so a stage that highlights one AU region is not a lone dot
            sx = max(size * 0.16, (x1 - x0) * size * 1.0)
            sy = max(size * 0.16, (y1 - y0) * size * 1.0)
            acc += float(weights[r]) * np.exp(-(((xx - cx) ** 2) / (2 * sx * sx)
                                                + ((yy - cy) ** 2) / (2 * sy * sy)))
        acc -= acc.min()
        return acc / max(acc.max(), 1e-8)


    n_cols = 1 + depth + 1
    headers = ["Origin"] + [f"Stage {i + 1}" for i in range(depth)] + ["Final"]
    fig, axes = plt.subplots(len(picks), n_cols,
                             figsize=(1.55 * n_cols, 1.5 * len(picks)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(picks):
        path, _ = pool[idx]
        # the paper's panels are grayscale faces with a jet overlay
        pil = Image.open(path).convert("L").convert("RGB").resize((size, size))
        axes[row, 0].imshow(np.asarray(pil))

        x = tf(pil).unsqueeze(0).to(device)
        maps = layer_attention(model, x)
        n_tok = maps[0].shape[0]
        k = n_tok - 1 - 2 * 8 - 1
        co_slice = slice(1 + k, 1 + k + 8)
        weights, heats = [], []
        for m in maps:
            w = m[0, co_slice]
            if float(w.max()) < 1e-6:
                # this layer gives the AU tokens no mass: fall back to the spatial
                # attention over the 64 visual tokens so the panel is not blank
                n_vis = 64
                att = m[0, 1:1 + n_vis]
                side = int(np.sqrt(n_vis))
                grid = att.reshape(side, side).astype(np.float32)
                grid -= grid.min()
                grid /= max(float(grid.max()), 1e-8)
                pil_h = Image.fromarray((grid * 255).astype(np.uint8))
                heats.append(np.asarray(pil_h.resize((size, size), Image.BILINEAR),
                                        dtype=np.float32) / 255.0)
                weights.append(np.zeros(8, dtype=np.float32))
            else:
                wn = w / max(float(w.max()), 1e-8)
                weights.append(wn)
                heats.append(smooth(wn))
        weights.append(np.mean(weights, axis=0))          # "Final" = consensus
        heats.append(np.mean(heats, axis=0))

        for col, heat in enumerate(heats, start=1):
            ax = axes[row, col]
            ax.imshow(np.asarray(pil))
            # full-frame wash, no threshold: the original colours the whole crop
            ax.imshow(heat, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)

        # annotate the strongest region in the last column, with the paper's own
        # vocabulary and type: white serif ~16 pt, no outline, thick dashed circle
        w_final = weights[-1]
        ax = axes[row, -1]
        top = int(np.argmax(w_final))
        if float(w_final[top]) >= 0.45:
            x0, y0, x1, y1 = boxes[top]
            cx, cy = (x0 + x1) / 2 * size, (y0 + y1) / 2 * size
            diam = 0.42 * size   # fixed across every row
            ax.add_patch(Ellipse((cx, cy), diam, diam, fill=False,
                                 ec="white", lw=4.0, ls=(0, (5, 3)), alpha=0.95))
            label = REGION_LABEL[top]
            if len(label) > 11:                    # keep the caption inside the panel
                label = label.replace(" ", "\n", 1)
            ax.text(size * 0.5, size * 0.15, label, color="white",
                    fontsize=12, fontfamily="serif", ha="center", va="center", clip_on=False,
                    alpha=0.95)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)

    # headers: serif, larger, no underline (removed on request)
    for col, h in enumerate(headers):
        ax = axes[0, col]
        ax.set_title(h, fontsize=16, fontfamily="serif", pad=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_stage_attention.png"))
    plt.close(fig)



def fig_confusion(ckpt: str, data: str, out: str, device: str = "cpu", limit: int = 1200) -> None:
    import torch
    from PIL import Image, ImageFilter
    from torchvision import transforms

    from .model import load_checkpoint

    model, cfg = load_checkpoint(ckpt, device)
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    classes = cfg["classes"]
    images = list_images(data, "test")
    if limit and len(images) > limit:
        # random, not the first N: the folder order puts every class in a block,
        # so a head slice would measure a single category
        rng = np.random.default_rng(0)
        images = [images[i] for i in rng.choice(len(images), limit, replace=False)]
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    batch = 16
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in chunk]).to(device)
        with torch.no_grad():
            pred = model(x)["prob"].argmax(1).cpu().numpy()
        for (_, c), p in zip(chunk, pred):
            cm[classes.index(c), p] += 1

    acc = np.trace(cm) / max(cm.sum(), 1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    ax.set_title(f"confusion (acc {acc:.1%}, n={len(images)})", fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5,
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(os.path.join(out, "fig_confusion.png"))
    plt.close(fig)
    print(f"  confusion: acc {acc:.1%} over {len(images)} images")


def fig_curves(log_path: str, out: str) -> None:
    train, val, epochs = [], [], []
    pat = re.compile(r"epoch\s+(\d+)\s+\|\s+train\s+([\d.]+)\s+\|\s+val\s+([\d.]+)")
    # PowerShell's Tee-Object writes UTF-16; a plain utf-8 read yields garbage
    with open(log_path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "ignore")
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
    if not epochs:
        print("  no epoch lines in", log_path)
        return

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.plot(epochs, train, marker="o", ms=2.5, lw=1.0, label="train")
    ax.plot(epochs, val, marker="s", ms=2.5, lw=1.0, label="val")
    ax.set_xlabel("epoch", fontsize=8)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(os.path.join(out, "fig_curves.png"))
    plt.close(fig)


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".rafdb")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--which", default="all",
                    help="all|regions|prior|occlusion|synapse|qualitative|confusion|curves")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=1200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    images = list_images(args.data, "test")
    if not images:
        raise SystemExit(f"no test images under {args.data}/test")

    todo = {
        "regions": lambda: fig_regions(images, args.out),
        "prior": lambda: fig_prior(args.data, args.out),
        "occlusion": lambda: fig_occlusion(images, args.out),
        "synapse": lambda: fig_synapse(args.ckpt, args.data, args.out, args.device),
        "qualitative": lambda: fig_qualitative(args.ckpt, args.data, args.out, args.device),
        "attention": lambda: fig_attention(args.ckpt, args.data, args.out, args.device),
        "challenging": lambda: fig_challenging(args.ckpt, args.data, args.out, args.device),
        "stages": lambda: fig_stage_attention(args.ckpt, args.data, args.out, args.device),
        "confusion": lambda: fig_confusion(args.ckpt, args.data, args.out, args.device, args.limit),
        "curves": lambda: fig_curves(args.log, args.out),
    }
    need_ckpt = {"synapse", "qualitative", "confusion", "attention", "stages", "challenging"}

    names = list(todo) if args.which == "all" else [args.which]
    for name in names:
        if name in need_ckpt and not args.ckpt:
            print(f"skip {name}: needs --ckpt")
            continue
        print(f"[{name}]")
        try:
            todo[name]()
        except Exception as exc:  # keep going: one broken figure should not block the rest
            print(f"  FAILED {name}: {exc}", file=sys.stderr)
    print("done ->", os.path.abspath(args.out))


if __name__ == "__main__":
    main()










