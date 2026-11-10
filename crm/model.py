"""CRM 閳?the full model (Sect. III-A overview, Fig. 3).

    x -> [Swin stages] -> COC -> SSA -> STM -> SIR -> (logits, c, s)

``backbone=None`` swaps Swin for a tiny convolutional stem with the same four
stage contract, so the whole pipeline can be smoke-tested on CPU without timm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coc import COCModule
from .regions import AU_REGIONS
from .sir import SIRStack
from .ssa import SSAModule
from .stm import TokenBuilder


class _Stem(nn.Module):
    """Tiny four-stage stand-in for Swin: NHWC outputs, same contract as timm."""

    def __init__(self, dims=(32, 64, 96, 128)) -> None:
        super().__init__()
        layers, prev = [], 3
        for d in dims:
            layers.append(nn.Sequential(
                nn.Conv2d(prev, d, 3, stride=2, padding=1),
                nn.BatchNorm2d(d),
                nn.GELU(),
            ))
            prev = d
        self.layers = nn.ModuleList(layers)
        self.dims = dims

    def forward(self, x):
        out = []
        for layer in self.layers:
            x = layer(x)
            out.append(x.permute(0, 2, 3, 1).contiguous())   # NHWC
        return out


class CRM(nn.Module):
    """Concomitance-guided expression model.

    Args:
        num_classes: 7 for RAF-DB / FERPlus, 8 for CK+.
        backbone: a timm model name, or ``None`` for the CPU-testable stem.
        pretrained: load timm weights (ignored for the stem).
        dim: common token width ``d``.
        semantic_dim: width of the cached LLM sentence embeddings.
        depth: number of SIR stages ``T``.
        gamma: disparity trade-off; 0 = the paper's co-activation ablation.
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str | None = "swin_base_patch4_window12_384",
        pretrained: bool = True,
        dim: int = 256,
        semantic_dim: int = 4096,
        depth: int = 6,
        heads: int = 8,
        kappa: int = 64,
        gamma: float = 0.5,
        dropout: float = 0.1,
        image_size: int = 384,
        use_coc: bool = True,
    ) -> None:
        super().__init__()
        if backbone is None:
            self.backbone = _Stem()
            stage_dims = self.backbone.dims
        else:
            import timm  # imported lazily so the stem path needs no timm

            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                num_classes=0,
                dynamic_img_size=True,   # allow sizes other than the variant's default
            )
            probe = torch.zeros(1, 3, image_size, image_size)
            with torch.no_grad():
                stage_dims = tuple(f.shape[-1] for f in self.backbone(probe))

        self.num_classes = num_classes
        self.dim = dim
        self.n_regions = len(AU_REGIONS)
        self.stage_dims = tuple(stage_dims)
        self.use_coc = use_coc

        self.coc = COCModule(texture_dim=stage_dims[0], n_regions=self.n_regions)
        #: inject the 256-channel compensated map back into the first stage
        self.comp_fuse = nn.Conv2d(256, stage_dims[0], kernel_size=1)
        self.vis_proj = nn.Linear(stage_dims[-1], dim)
        self.ssa = SSAModule(dim=dim, semantic_dim=semantic_dim)
        self.stm = TokenBuilder(
            stage_dims=stage_dims, dim=dim, n_regions=self.n_regions,
            kappa=kappa, semantic_dim=dim,
        )
        self.sir = SIRStack(dim=dim, depth=depth, heads=heads, n_regions=self.n_regions,
                            gamma=gamma, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, num_classes),
        )
        self.register_buffer("prior", torch.eye(self.n_regions))

    # ------------------------------------------------------------------ prior
    def load_prior(self, matrix: torch.Tensor) -> None:
        """Install the offline concomitance matrix C (see ``estimate_c.py``)."""
        assert matrix.shape == (self.n_regions, self.n_regions)
        self.prior = matrix.float()

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        semantics: torch.Tensor | None = None,
        landmarks: torch.Tensor | None = None,
        symmetry: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)                                   # 4 x NHWC

        coc = self.coc(x, feats[0], landmarks=landmarks, symmetry=symmetry,
                       compensate=self.use_coc)

        # COC output re-enters the visual stream at the first stage (Fig. 3)
        f0 = feats[0]
        if f0.shape[1] != self.stage_dims[0]:                     # NHWC -> NCHW
            f0 = f0.permute(0, 3, 1, 2)
        comp = F.interpolate(coc["comp"], size=f0.shape[-2:], mode="bilinear", align_corners=False)
        feats = [f0 + self.comp_fuse(comp)] + list(feats[1:])

        # rough region tokens for the semantic alignment (deepest stage)
        deep = feats[-1]
        if deep.shape[-1] != self.stage_dims[-1]:                  # NCHW backbone
            deep = deep.permute(0, 2, 3, 1)
        visual = self.vis_proj(deep.reshape(deep.shape[0], -1, deep.shape[-1]))
        ssa = self.ssa(visual, semantics)

        stm = self.stm(feats, coc["a_hat"], self.prior, ssa["semantic"])
        sir = self.sir(stm["tokens"], stm["regions"], self.prior)

        tokens = sir["tokens"]
        cls = tokens[:, 0]
        pooled = tokens[:, 1:].mean(dim=1)
        logits = self.head(torch.cat([cls, pooled], dim=-1))
        prob = logits.softmax(dim=-1)

        # consistency statistic s = 1 - ||A||/||W|| with W = S + A (eq. 18/20)
        a_norm = sir["a"].abs().sum()
        s_norm = (sir["s"] + sir["a"]).abs().sum().clamp_min(1e-6)
        consistency = (1.0 - a_norm / s_norm).clamp(0.0, 1.0)

        return {
            "logits": logits,
            "prob": prob,
            "confidence": prob.max(dim=-1).values,
            "consistency": consistency,
            "a": coc["a"],
            "a_hat": coc["a_hat"],
            "rho": coc["rho"],
            "occ": coc["occ"],
            "comp": coc["comp"],
            "plan": ssa["plan"],
            "align_loss": ssa["align_loss"],
            "budget_loss": stm["budget_loss"],
            "s": sir["s"],
            "graph_a": sir["a"],
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x, **kwargs)["logits"].argmax(dim=-1)


def load_checkpoint(path: str, device: str = "cpu"):
    """Rebuild a :class:`CRM` from a training checkpoint.

    ``train.py`` stores the constructor arguments under ``config``; older
    checkpoints only carry ``model`` and ``classes``, in which case the defaults
    are used.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck.get("config") or {})
    classes = list(ck.get("classes") or [str(i) for i in range(7)])

    model = CRM(
        num_classes=len(classes),
        backbone=cfg.get("backbone", None),
        pretrained=False,
        dim=cfg.get("dim", 256),
        semantic_dim=cfg.get("semantic_dim", 4096),
        depth=cfg.get("depth", 6),
        heads=cfg.get("heads", 8),
        kappa=cfg.get("kappa", 64),
        gamma=cfg.get("gamma", 0.5),
        image_size=cfg.get("image_size", 384),
        use_coc=cfg.get("use_coc", True),
    )
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    cfg.setdefault("image_size", 384)
    cfg["classes"] = classes
    return model, cfg


