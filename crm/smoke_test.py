"""Shape and gradient smoke test that needs no dataset and no timm.

    python -m crm.smoke_test
    python -m crm.smoke_test --size 256 --batch 4

It builds CRM with the tiny convolutional stem, runs a forward pass, checks
every tensor shape the paper implies, then back-propagates the composite loss so
a broken graph is caught before anyone touches a GPU.
"""

from __future__ import annotations

import argparse

import torch

from .losses import CRMLoss, LossWeights
from .model import CRM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--classes", type=int, default=7)
    ap.add_argument("--backbone", default="none",
                    help="'none' for the CPU stem, or a timm model name")
    ap.add_argument("--pretrained", action="store_true",
                    help="load timm weights (needs network)")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = CRM(
        num_classes=args.classes,
        backbone=None if args.backbone == "none" else args.backbone,
        pretrained=args.pretrained,
        dim=64,
        semantic_dim=32,
        depth=2,
        heads=4,
        kappa=16,
        image_size=args.size,
    )
    model.train()

    x = torch.randn(args.batch, 3, args.size, args.size)
    y = torch.randint(0, args.classes, (args.batch,))
    semantics = torch.randn(args.batch, 3, 32)

    out = model(x, semantics=semantics)
    checks = {
        "logits": (args.batch, args.classes),
        "prob": (args.batch, args.classes),
        "confidence": (args.batch,),
        "a": (args.batch, 8),
        "a_hat": (args.batch, 8),
        "rho": (args.batch, 8),
        "occ": (args.batch, None, None),
        "plan": (args.batch, None, 3),
    }
    for name, expected in checks.items():
        got = tuple(out[name].shape)
        ok = len(got) == len(expected) and all(e is None or e == g for e, g in zip(expected, got))
        print(f"{'OK ' if ok else 'BAD'} {name:12s} {got}")

    print(f"comp map      {tuple(out['comp'].shape)}  (expect 256 channels)")
    assert out["comp"].shape[1] == 256, "V_comp must carry 256 channels"

    criterion = CRMLoss(LossWeights())
    losses = criterion(out, y, model.prior)
    losses["loss"].backward()

    grads = {
        name: (p.grad is not None)
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    # ``null_semantics`` is only used when no cached description is supplied
    unused = [n for n, ok in grads.items() if not ok and "null_semantics" not in n]
    print(f"loss          {losses['loss'].item():.4f}")
    print(f"terms         " + "  ".join(f"{k}={v.item():.3f}" for k, v in losses.items() if k != "loss"))
    print(f"params with grad: {sum(grads.values())}/{len(grads)}")
    if unused:
        print("no gradient: " + ", ".join(unused))
    ok = bool(torch.isfinite(losses["loss"])) and not unused
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")


if __name__ == "__main__":
    main()
