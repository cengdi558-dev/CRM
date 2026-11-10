"""STM — Synapse-Aware Tokenization (Sect. III-D, eq. 13-15).

Builds the sequence

    [CLS | visual tokens | concomitance tokens | disparity tokens | semantic]

adds positional and *type* embeddings (the paper notes that omitting the type
embedding made the semantic branch get ignored), and keeps the visual budget at
``kappa`` with a differentiable top-k and a budget penalty.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regions import MIRROR, assign_regions


class TokenBuilder(nn.Module):
    """Assemble the synapse-aware token sequence.

    Args:
        stage_dims: channel count of each Swin stage (e.g. 96/192/384/768).
        dim: common token width ``d``.
        n_regions: number of AU regions.
        kappa: visual-token budget ``kappa`` (64 in the paper).
        input_size: spatial size the features came from (for region lookup).
    """

    def __init__(
        self,
        stage_dims: tuple[int, ...] = (96, 192, 384, 768),
        dim: int = 256,
        n_regions: int = 8,
        kappa: int = 64,
        semantic_dim: int = 256,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_regions = n_regions
        self.kappa = kappa
        self.stage_dims = tuple(stage_dims)

        self.stage_proj = nn.ModuleList(
            [nn.Linear(c, dim) for c in stage_dims]
        )
        self.co_proj = nn.Linear(n_regions, dim)      # phi_co(C_j:)
        self.di_proj = nn.Linear(1, dim)              # phi_di(a_j - a_pi(j))
        self.semantic_proj = nn.Linear(semantic_dim, dim)

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        self.type_embed = nn.Embedding(5, dim)        # 0 cls 1 vis 2 co 3 di 4 sem
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    # ------------------------------------------------------------------ utils
    def _to_nchw(self, feat: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Accept both timm's NHWC stages and a plain NCHW stem output."""
        expected = self.stage_dims[stage_idx]
        if feat.shape[1] == expected:
            return feat
        if feat.shape[-1] == expected:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        feats: list[torch.Tensor],
        a_hat: torch.Tensor,
        prior: torch.Tensor,
        semantics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """``feats``: four stage maps; ``a_hat``: [B, 8]; ``prior``: [8, 8]."""
        b = a_hat.shape[0]
        device = a_hat.device

        visual, region_ids = [], []
        for stage_idx, (feat, proj) in enumerate(zip(feats, self.stage_proj)):
            f = self._to_nchw(feat, stage_idx)            # [B, C, h, w]
            _, c, h, w = f.shape
            tokens = f.flatten(2).transpose(1, 2)         # [B, h*w, C]
            visual.append(proj(tokens))
            region_ids.append(assign_regions(h, w, h * 4, w * 4).to(device))
        visual_tokens = torch.cat(visual, dim=1)          # [B, Nv, d]
        visual_regions = torch.cat(region_ids, dim=0)     # [Nv]

        # --- top-kappa selection over visual tokens (eq. 15) ---------------
        scores = self.score(visual_tokens).squeeze(-1)    # [B, Nv]
        k = min(self.kappa, scores.shape[1])
        top = scores.topk(k, dim=1).indices               # [B, k]
        mask = torch.zeros_like(scores).scatter_(1, top, 1.0)
        hard = mask
        soft = hard + scores - scores.detach()            # straight-through
        budget = ((scores.sigmoid().sum(dim=1) - k).abs() / max(scores.shape[1], 1)).mean()

        gather = top.unsqueeze(-1).expand(-1, -1, self.dim)
        visual_tokens = torch.gather(visual_tokens, 1, gather) * soft.gather(1, top).unsqueeze(-1)
        # region ids follow each sample's own selection
        visual_regions = torch.gather(
            visual_regions.unsqueeze(0).expand(b, -1), 1, top
        )                                                  # [B, k]

        # --- prior / disparity tokens (eq. 13) -----------------------------
        prior_n = F.normalize(prior.clone(), p=1, dim=1)  # row-normalised
        co = self.co_proj(prior_n.to(device).unsqueeze(0).expand(b, -1, -1))   # [B, 8, d]
        mirror = torch.tensor([MIRROR[i] for i in range(self.n_regions)], device=device)
        disp = (a_hat - a_hat[:, mirror]).unsqueeze(-1)   # [B, 8, 1]
        di = self.di_proj(disp)                            # [B, 8, d]

        sem = self.semantic_proj(semantics).unsqueeze(1)   # [B, 1, d]

        cls = self.cls.expand(b, -1, -1)
        tokens = torch.cat([cls, visual_tokens, co, di, sem], dim=1)

        types = torch.cat([
            torch.zeros(1, device=device),
            torch.ones(visual_tokens.shape[1], device=device),
            torch.full((self.n_regions,), 2, device=device),
            torch.full((self.n_regions,), 3, device=device),
            torch.full((1,), 4, device=device),
        ]).long()
        tokens = tokens + self.type_embed(types).unsqueeze(0)
        tokens = tokens + self.pos[:, : tokens.shape[1]]

        n_vis = visual_tokens.shape[1]
        regions = torch.cat([
            torch.full((b, 1), -1, device=device, dtype=torch.long),   # cls
            visual_regions,                                            # [B, k]
            torch.arange(self.n_regions, device=device).unsqueeze(0).expand(b, -1),
            torch.arange(self.n_regions, device=device).unsqueeze(0).expand(b, -1),
            torch.full((b, 1), -1, device=device, dtype=torch.long),   # semantic
        ], dim=1)

        return {
            "tokens": tokens,
            "regions": regions,
            "budget_loss": budget,
            "n_visual": n_vis,
        }
