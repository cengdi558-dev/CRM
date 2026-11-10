"""Grad-CAM and per-layer attention for the CRM model.

The paper's Fig. 9 / 10 / 13 are attention visualisations, so they have to be
computed from the model rather than cropped from somewhere else. Two views are
provided:

* ``GradCAM`` — gradient-weighted spatial map on the deepest backbone stage,
  the usual "where does the decision come from" picture;
* ``layer_attention`` — the SIR attention at each of the T stages, which is what
  the "self-iterative attention" figure (Fig. 13) shows.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def input_saliency(model: torch.nn.Module, x: torch.Tensor, class_idx: int | None = None,
                   blur: int = 7) -> np.ndarray:
    """Gradient saliency w.r.t. the input pixels.

    Grad-CAM through the backbone does not work on this architecture: the
    straight-through top-k gate in STM multiplies the visual tokens by a
    constant mask, so d(logit)/d(feature) is ~1e-18 and the CAM collapses to
    zero. The input gradient has no such gate and always exists, so it is the
    honest choice for an attention/saliency figure.
    """
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    logits = out["logits"]
    if class_idx is None:
        class_idx = int(logits.argmax(dim=-1)[0])
    model.zero_grad(set_to_none=True)
    logits[0, class_idx].backward()

    sal = x.grad.detach().abs().sum(dim=1, keepdim=True)      # [B,1,H,W]
    sal = F.avg_pool2d(sal, kernel_size=blur, stride=1, padding=blur // 2)
    sal = sal[0, 0].cpu().numpy()
    sal -= sal.min()
    sal /= max(sal.max(), 1e-8)
    return sal


class GradCAM:
    """Hook the backbone output and weight its channels by the class gradient."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = model.backbone.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        feat = output[-1] if isinstance(output, (list, tuple)) else output
        self.activations = feat
        feat.register_hook(self._store_grad)

    def _store_grad(self, grad) -> None:
        self.gradients = grad

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        logits = out["logits"]
        if class_idx is None:
            class_idx = int(logits.argmax(dim=-1)[0])
        logits[0, class_idx].backward()

        act = self.activations                       # timm gives NHWC
        grad = self.gradients
        if act is None or grad is None:
            raise RuntimeError("backbone hook did not fire")

        # decide the layout from the known stage width, not from shape equality
        try:
            expected = self.model.stage_dims[-1]
        except AttributeError:
            expected = None
        if expected is not None and act.shape[-1] == expected:
            act = act.permute(0, 3, 1, 2)
            grad = grad.permute(0, 3, 1, 2)
        elif act.shape[-1] != grad.shape[-1]:
            act = act.permute(0, 3, 1, 2)
            grad = grad.permute(0, 3, 1, 2)

        weights = grad.mean(dim=(2, 3), keepdim=True)          # [B, C, 1, 1]
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # [B, 1, h, w]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= max(cam.max(), 1e-8)
        return cam


def layer_attention(model: torch.nn.Module, x: torch.Tensor) -> list[np.ndarray]:
    """Attention of every SIR layer, averaged over heads, as ``[N, N]`` maps.

    ``SIRLayer`` keeps its last attention map when ``store_attention`` is set on
    the instance; this helper turns it on, runs the model and restores it.
    """
    layers = list(model.sir.layers)
    for layer in layers:
        layer.store_attention = True
    try:
        with torch.no_grad():
            model(x)
    finally:
        for layer in layers:
            layer.store_attention = False

    maps = []
    for layer in layers:
        attn = getattr(layer, "last_attention", None)
        if attn is not None:
            maps.append(attn[0].mean(dim=0).cpu().numpy())   # average heads
    return maps
