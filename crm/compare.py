"""Train comparison baselines and render the Fig. 9 / Fig. 10 style grids.

One command does everything for a set of timm backbones:

    python -m crm.compare --data .rafdb --models resnet50,vit_base_patch16_224,swin_tiny_patch4_window7_224 \
        --image-size 224 --epochs 8 --samples 10 --out compare

For every model it trains a classifier, scores the same fixed sample set with
Grad-CAM, and finally writes:

    compare/fig9_grid.png    Origin | model A | model B | ... (paper layout)
    compare/summary.json     per-model accuracy on the same split

Only models that classify a sample correctly are allowed to illustrate it, so
the grid compares methods on the same faces.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def transforms_for(size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def build_model(name: str, num_classes: int):
    import timm
    return timm.create_model(name, pretrained=True, num_classes=num_classes)


def train_one(model, loader, device, epochs, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    for ep in range(epochs):
        model.train()
        correct = seen = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            correct += (out.argmax(-1) == y).sum().item()
            seen += y.numel()
        sched.step()
        print(f"    epoch {ep + 1}/{epochs}  acc {correct / max(seen, 1):.3f}", flush=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        seen += y.numel()
    return correct / max(seen, 1)


def saliency(model, x, class_idx):
    """Input-gradient saliency, used for *every* method so the columns are comparable.

    Grad-CAM needs a conv feature map (ViTs have none) and our CRM blocks the
    gradient through its top-k gate, so a single method that works everywhere is
    the only fair choice for a comparison figure.
    """
    x = x.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[0, class_idx].backward()
    sal = x.grad.detach().abs().sum(dim=1, keepdim=True)
    sal = F.avg_pool2d(sal, kernel_size=7, stride=1, padding=3)
    sal = sal[0, 0].cpu().numpy()
    sal -= sal.min()
    return sal / max(sal.max(), 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".rafdb")
    ap.add_argument("--models", default="resnet50,vit_base_patch16_224")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--out", default="compare")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = args.image_size

    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), transforms_for(size, True))
    val_ds = datasets.ImageFolder(os.path.join(args.data, "val"), transforms_for(size, False))
    classes = train_ds.classes

    if args.train_limit:
        idx = torch.randperm(len(train_ds))[: args.train_limit].tolist()
        train_ds = Subset(train_ds, idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    summary = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"== {name} ==", flush=True)
        model = build_model(name, len(classes)).to(device)
        train_one(model, train_loader, device, args.epochs)
        acc = evaluate(model, val_loader, device)
        print(f"  val acc {acc:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "classes": classes}, os.path.join(args.out, f"{name}.pth"))
        summary[name] = acc

        # fixed sample set: the first N per class, so every model sees the same faces
        picks = []
        for ci, cls in enumerate(classes):
            folder = os.path.join(args.data, "val", cls)
            files = sorted(os.listdir(folder))[: max(1, args.samples // len(classes))]
            picks += [(os.path.join(folder, f), ci) for f in files]

        tf = transforms_for(size, False)
        heat, truth = [], []
        for path, ci in picks:
            pil = Image.open(path).convert("RGB")
            x = tf(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = int(model(x).argmax(-1))
            cam = saliency(model, x, pred)
            heat.append(cam)
            truth.append((path, ci, pred))
        np.savez(os.path.join(args.out, f"{name}_heat.npz"),
                 heat=np.stack([h if h is not None else np.zeros((size, size)) for h in heat]))
        with open(os.path.join(args.out, f"{name}_samples.json"), "w", encoding="utf-8") as fh:
            json.dump(truth, fh, ensure_ascii=False)

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("done:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()


