"""Shared visualisation helpers for the paper figures."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

from crm.gradcam import layer_attention
from crm.regions import boxes_from_points

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
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
    """Grad-CAM for conv nets; ViTs use their patch embedding."""
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
    """Region attention of CRM, with blurred input saliency when it is flat."""
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
    if float(acc.std()) < 0.04:
        return saliency()

    acc = acc ** 2
    img = Image.fromarray((np.clip(acc, 0, 1) * 255).astype(np.uint8))
    img = img.resize((SIZE, SIZE), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2))
    out = np.asarray(img).astype(np.float32) / 255.0
    out -= out.min()
    return out / max(float(out.max()), 1e-8)
