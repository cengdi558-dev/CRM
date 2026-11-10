"""The composite objective of Sect. III-E, eq. 21.

    L = L_cls + l1*L_align + l2*L_comp + l3*L_disp + l4*L_prior + l5*L_sparse

Every term is exposed separately so the ablations the paper reports can be run
by zeroing a single weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    align: float = 1.0
    comp: float = 0.5
    disp: float = 0.1
    prior: float = 0.1
    sparse: float = 0.01
    budget: float = 0.01
    margin: float = 0.2
    #: class-level discriminative AU pairs (u, v); empty = use all 28 pairs
    discriminative_pairs: tuple[tuple[int, int], ...] = field(default_factory=tuple)


class CRMLoss(nn.Module):
    """Compute every term and the weighted sum."""

    def __init__(self, weights: LossWeights | None = None, n_regions: int = 8) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        pairs = self.w.discriminative_pairs or tuple(
            (i, j) for i in range(n_regions) for j in range(i + 1, n_regions)
        )
        self.register_buffer("pairs", torch.tensor(pairs, dtype=torch.long))

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cls = F.cross_entropy(outputs["logits"], labels)

        # L_comp: the compensated response must stay close to the observed one
        # where the region is visible, and is free where it is occluded
        rho = outputs["rho"]
        comp = (((outputs["a_hat"] - outputs["a"]) ** 2) * rho).mean()

        # L_prior: anchor the learned symmetric part to the fixed concomitance C.
        # The graph carries one token per region per family (concomitance +
        # disparity), so C is tiled to the node count by region index.
        s = outputs["s"]
        if s.dim() == 3:
            ref = prior.to(s.device)
            if s.shape[1] != ref.shape[0]:
                factor = s.shape[1] // ref.shape[0]
                ref = ref.repeat(factor, factor)
            prior_term = ((s - ref.unsqueeze(0)) ** 2).mean()
        else:
            prior_term = s.new_zeros(())

        # L_disp: margin that separates discriminative from ordinary pairs
        a = outputs["graph_a"]
        if a.dim() == 3 and self.pairs.numel() > 0:
            p = self.pairs
            disc = (a[:, p[:, 0], p[:, 1]] * 2.0).abs().mean()
            mask = torch.ones_like(a, dtype=torch.bool)
            mask[:, p[:, 0], p[:, 1]] = False
            mask[:, p[:, 1], p[:, 0]] = False
            other = (a * 2.0).abs()[mask].mean() if mask.any() else disc.detach() * 0
            disp = F.relu(self.w.margin - (disc - other))
        else:
            disp = cls.new_zeros(())

        # L_sparse: L1 on the directed graph
        sparse = a.abs().mean() if a.dim() == 3 else a.abs()

        total = (
            cls
            + self.w.align * outputs["align_loss"]
            + self.w.comp * comp
            + self.w.disp * disp
            + self.w.prior * prior_term
            + self.w.sparse * sparse
            + self.w.budget * outputs["budget_loss"]
        )
        return {
            "loss": total,
            "cls": cls.detach(),
            "align": outputs["align_loss"].detach(),
            "comp": comp.detach(),
            "disp": disp.detach(),
            "prior": prior_term.detach() if torch.is_tensor(prior_term) else prior_term,
            "sparse": sparse.detach(),
            "budget": outputs["budget_loss"].detach(),
        }
