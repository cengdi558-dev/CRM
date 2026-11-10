"""Training entry point, following the repo's data conventions.

The original ``utilities/train_*.py`` scripts use ``datasets.ImageFolder`` with
``train/`` and ``val/`` sub-folders and AdamW + cosine. This script keeps that
contract so an existing dataset layout works unchanged.

    python -m crm.train --data FER2013_processed --prior c.npy
    python -m crm.train --data FER2013_processed --backbone none      # CPU smoke

Optional occlusion synthesis (``--occlude 0.5``) pastes a random rectangle over
half the batch, which is how the paper's robustness setting is approximated.
"""

from __future__ import annotations

import argparse
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .losses import CRMLoss, LossWeights
from .model import CRM

# torch 2.4 moved these to torch.amp; older images (2.0-2.3, the common AutoDL
# templates) only have torch.cuda.amp. Pick whichever exists.
def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_loaders(data_root: str, image_size: int, batch_size: int, workers: int, wrapper=None):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val = datasets.ImageFolder(os.path.join(data_root, "val"), transform=val_tf)
    if wrapper is not None:
        train, val = wrapper(train), wrapper(val)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=workers),
    )


def occlude_batch(x: torch.Tensor, ratio: float, rng: random.Random) -> torch.Tensor:
    """Paste a solid rectangle over ``ratio`` of the batch (occlusion bench)."""
    if ratio <= 0:
        return x
    x = x.clone()
    b, _, h, w = x.shape
    n = int(b * ratio)
    for i in rng.sample(range(b), n):
        bh, bw = rng.randint(h // 6, h // 2), rng.randint(w // 6, w // 2)
        y0, x0 = rng.randint(0, h - bh), rng.randint(0, w - bw)
        x[i, :, y0:y0 + bh, x0:x0 + bw] = 0.0
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root containing train/ and val/")
    ap.add_argument("--prior", default="", help="C matrix .npy from crm.estimate_c")
    ap.add_argument("--backbone", default="swin_base_patch4_window12_384",
                    help="timm name, or 'none' for the CPU stem")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.5, help="0 = co-activation ablation")
    ap.add_argument("--dim", type=int, default=256, help="token width (lower = faster on CPU)")
    ap.add_argument("--depth", type=int, default=6, help="SIR stages T")
    ap.add_argument("--kappa", type=int, default=64, help="visual token budget")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the CRM head (CPU-friendly)")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation steps (effective batch = batch*accum)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after N optimiser steps (dry run / timing)")
    ap.add_argument("--train-limit", type=int, default=0,
                    help="use only N random training images (CPU-friendly)")
    ap.add_argument("--val-limit", type=int, default=0,
                    help="cap the validation set size")
    ap.add_argument("--no-coc", action="store_true", help="disable occlusion compensation (ablation)")
    ap.add_argument("--balanced", action="store_true",
                    help="class-balanced sampling (RAF-DB is heavily imbalanced)")
    ap.add_argument("--occlude", type=float, default=0.0)
    ap.add_argument("--semantics", default="", help="semantics .npz from crm.gen_semantics")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="crm_best.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    semantic_dim = 4096
    wrapper = None
    if args.semantics:
        from .semantic_cache import SemanticCache, SemanticDataset

        cache = SemanticCache(args.semantics)
        semantic_dim = cache.dim
        wrapper = lambda ds: SemanticDataset(ds, cache)  # noqa: E731
        print(f"semantics: {len(cache)} cached, {cache.levels}x{cache.dim}")

    train_loader, val_loader = build_loaders(
        args.data, args.image_size, args.batch_size, args.workers, wrapper
    )

    if args.train_limit or args.val_limit:
        import torch as _t
        from torch.utils.data import Subset

        if args.train_limit and args.train_limit < len(train_loader.dataset):
            idx = _t.randperm(len(train_loader.dataset))[: args.train_limit].tolist()
            train_loader = DataLoader(Subset(train_loader.dataset, idx), batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers, drop_last=True)
        if args.val_limit and args.val_limit < len(val_loader.dataset):
            idx = _t.randperm(len(val_loader.dataset))[: args.val_limit].tolist()
            val_loader = DataLoader(Subset(val_loader.dataset, idx), batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers)

    base = train_loader.dataset
    while hasattr(base, "dataset"):
        base = base.dataset
    num_classes = len(base.classes)

    if args.balanced:
        # RAF-DB train is ~17:1 between the largest and smallest class; without
        # rebalancing the model collapses onto happiness/neutral.
        import numpy as _np
        from torch.utils.data import WeightedRandomSampler

        targets = _np.array([label for _, label in base.samples])
        counts = _np.bincount(targets, minlength=num_classes).astype(float)
        weights = (1.0 / counts)[targets]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_loader.dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.workers, drop_last=True,
        )
        print(f"balanced sampling over {num_classes} classes: {counts.astype(int).tolist()}")

    model = CRM(
        num_classes=num_classes,
        backbone=None if args.backbone == "none" else args.backbone,
        gamma=args.gamma,
        image_size=args.image_size,
        semantic_dim=semantic_dim,
        use_coc=not args.no_coc,
        dim=args.dim,
        depth=args.depth,
        kappa=args.kappa,
    ).to(device)

    if args.freeze_backbone and args.backbone != "none":
        for p in model.backbone.parameters():
            p.requires_grad = False
        frozen = sum(p.numel() for p in model.backbone.parameters())
        print(f"backbone frozen: {frozen:,} parameters")

    if args.prior:
        import numpy as np
        model.load_prior(torch.from_numpy(np.load(args.prior)))

    criterion = CRMLoss(LossWeights())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = _make_scaler(device == "cuda")
    rng = random.Random(0)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, seen = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            x, y = batch[0].to(device), batch[1].to(device)
            sem = batch[2].to(device) if len(batch) > 2 else None
            x = occlude_batch(x, args.occlude, rng)

            with _autocast(AMP_ENABLED and device == "cuda"):
                out = model(x, semantics=sem)
                losses = criterion(out, y, model.prior)

            # gradient accumulation: step only every `accum` micro-batches
            scaler.scale(losses["loss"] / args.accum).backward()
            if step % args.accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total += losses["loss"].item() * y.numel()
            correct += (out["logits"].argmax(-1) == y).sum().item()
            seen += y.numel()
            if args.max_steps and step >= args.max_steps:
                print(f"  reached --max-steps {args.max_steps}, stopping")
                break
        # flush a trailing partial group
        if len(train_loader) % args.accum:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        model.eval()
        v_total, v_correct, v_seen = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                sem = batch[2].to(device) if len(batch) > 2 else None
                with _autocast(AMP_ENABLED and device == "cuda"):
                    out = model(x, semantics=sem)
                    losses = criterion(out, y, model.prior)
                v_total += losses["loss"].item() * y.numel()
                v_correct += (out["logits"].argmax(-1) == y).sum().item()
                v_seen += y.numel()

        train_acc = correct / max(seen, 1)
        val_acc = v_correct / max(v_seen, 1)
        print(f"epoch {epoch:3d} | train {train_acc:.4f} | val {val_acc:.4f} | "
              f"cls {losses['cls'].item():.4f} align {losses['align'].item():.4f}")

        if val_acc > best:
            best = val_acc
            torch.save({
                "model": model.state_dict(),
                "classes": base.classes,
                "config": {
                    "backbone": None if args.backbone == "none" else args.backbone,
                    "image_size": args.image_size,
                    "dim": args.dim,
                    "depth": args.depth,
                    "kappa": args.kappa,
                    "gamma": args.gamma,
                    "semantic_dim": semantic_dim,
                    "use_coc": not args.no_coc,
                    "epochs": args.epochs,
                },
            }, args.out)
            print(f"  saved {args.out} (val {best:.4f})")

    print(f"done, best val acc = {best:.4f}")


if __name__ == "__main__":
    main()



