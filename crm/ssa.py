"""SSA — Synapse-Guided Semantic Alignment (Sect. III-C, eq. 9-12).

The three prompt levels (coarse / medium / fine) are generated offline by an
instruction LLM and cached; this module only needs their embeddings, so the
training loop never blocks on a text model.

Alignment is an entropic optimal-transport problem between visual tokens and
the semantic phrases, with a cosine cost. ``sinkhorn`` is exported separately so
the transport plan can be inspected (it doubles as the interpretability map of
Fig. 4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


def sinkhorn(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    iters: int = 30,
    row_mass: torch.Tensor | None = None,
    col_mass: torch.Tensor | None = None,
) -> torch.Tensor:
    """Entropic transport plan for a ``[B, N, M]`` cost matrix (eq. 10-11).

    Alternating row/column normalisations of the Gibbs kernel, which is the
    Sinkhorn iteration the paper refers to. Uniform marginals by default.
    """
    b, n, m = cost.shape
    kernel = torch.exp(-cost / epsilon)

    if row_mass is None:
        row_mass = torch.full((b, n), 1.0 / n, device=cost.device, dtype=cost.dtype)
    if col_mass is None:
        col_mass = torch.full((b, m), 1.0 / m, device=cost.device, dtype=cost.dtype)

    u = torch.ones_like(row_mass)
    v = torch.ones_like(col_mass)
    for _ in range(iters):
        u = row_mass / (kernel @ v.unsqueeze(-1)).squeeze(-1).clamp_min(_EPS)
        v = col_mass / (kernel.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(_EPS)
    return u.unsqueeze(-1) * kernel * v.unsqueeze(1)


class SSAModule(nn.Module):
    """Project visual and semantic tokens, align them, and score the objective.

    Args:
        dim: shared embedding width ``d``.
        semantic_dim: width of the cached LLM embeddings.
        epsilon: entropic regularisation.
        levels: number of prompt levels (3 in the paper).
        level_weights: ``omega = (0.25, 0.35, 0.40)`` in the paper.
        tau: temperature of the InfoNCE term.
    """

    def __init__(
        self,
        dim: int = 256,
        semantic_dim: int = 4096,
        epsilon: float = 0.05,
        levels: int = 3,
        level_weights: tuple[float, float, float] = (0.25, 0.35, 0.40),
        tau: float = 0.07,
        nce_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.epsilon = epsilon
        self.tau = tau
        self.nce_weight = nce_weight
        self.register_buffer("level_weights", torch.tensor(level_weights))

        self.visual_proj = nn.Linear(dim, dim)
        self.semantic_proj = nn.Linear(semantic_dim, dim)
        #: learned fallback when no LLM description is cached for a sample
        self.null_semantics = nn.Parameter(torch.zeros(levels, semantic_dim))

    def forward(
        self,
        visual: torch.Tensor,
        semantics: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """``visual``: [B, N, d]; ``semantics``: [B, L, semantic_dim] or None."""
        b, n, _ = visual.shape
        if semantics is None:
            semantics = self.null_semantics.unsqueeze(0).expand(b, -1, -1)

        v = F.normalize(self.visual_proj(visual), dim=-1)                 # [B, N, d]
        s = F.normalize(self.semantic_proj(semantics), dim=-1)            # [B, L, d]

        cost = 1.0 - torch.bmm(v, s.transpose(1, 2))                      # [B, N, L]
        plan = sinkhorn(cost, self.epsilon)                               # [B, N, L]

        weights = self.level_weights.to(semantics.device)[: s.shape[1]]
        weights = weights / weights.sum()
        semantic_fused = torch.einsum("bnl,bld->bd", plan, s)             # [B, d]
        semantic_fused = semantic_fused + (weights.view(1, -1, 1) * s).sum(dim=1)

        transport = (plan * cost).sum(dim=(1, 2)).mean()
        nce = self._nce(v.mean(dim=1), semantic_fused)
        align = transport + self.nce_weight * nce

        return {
            "plan": plan,
            "cost": cost,
            "semantic": semantic_fused,
            "align_loss": align,
            "transport": transport,
            "nce": nce,
        }

    def _nce(self, visual: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        """In-batch contrastive term that stops all tokens collapsing (eq. 12)."""
        logits = visual @ semantic.t() / self.tau                        # [B, B]
        target = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)
        )
