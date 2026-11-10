"""SIR — Synapse-Informed Interaction Reasoning (Sect. III-E, eq. 16-20).

Two departures from plain self-attention:

1. the logits carry an anatomical bias ``beta_b * B`` with
   ``B_ik = log(1 + C_{r(i),r(k)})`` and are sparsified by an *additive log-mask*
   **before** the softmax (thresholding after it kills the gradient);
2. a directed weight ``W_uv`` is learned from the ordered node pair, so
   ``A = (W - W^T)/2`` and the disparity ``Delta_uv = 2 A_uv`` exist at all.

Message passing weights each neighbour by ``S_uv + gamma * tanh(Delta_uv)``;
``gamma = 0`` recovers the pure co-activation model, which is the ablation the
paper reports.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIRLayer(nn.Module):
    """One synapse-aware interaction block.

    Args:
        dim: token width.
        heads: attention heads ``H``.
        n_regions: number of AU regions.
        gamma: trade-off between co-activation and direction.
        beta: strength of the anatomical bias.
        keep: fraction of edges kept after the per-head quantile mask.
    """

    def __init__(
        self,
        dim: int = 256,
        heads: int = 8,
        n_regions: int = 8,
        gamma: float = 0.5,
        beta: float = 1.0,
        keep: float = 0.8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.n_regions = n_regions
        self.gamma = gamma
        self.beta = beta
        self.keep = keep

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # directed weight W_uv from [u, v, relative region distance, prior coupling]
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 2, dim), nn.GELU(), nn.Linear(dim, 1)
        )
        self.graph_out = nn.Linear(dim * 2, dim)

    # ------------------------------------------------------------------ edge
    def _bias(self, regions: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """``[B, N, N]`` anatomical bias; non-AU tokens (region -1) get zero."""
        valid = regions >= 0
        onehot = F.one_hot(regions.clamp_min(0), num_classes=self.n_regions).to(prior.dtype)
        c = torch.einsum("bni,ij,bkj->bnk", onehot, prior.to(regions.device), onehot)
        bias = torch.log1p(c.clamp_min(0))
        gate = (valid[:, :, None] & valid[:, None, :]).float()
        return self.beta * bias * gate

    def _directed_graph(
        self,
        x: torch.Tensor,
        regions: torch.Tensor,
        prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Learn ``W_uv`` on the AU nodes and return ``S``, ``Delta`` and messages."""
        # the concomitance + disparity tokens form a fixed-size tail block:
        # [CLS][visual x k][co x 8][di x 8][semantic]
        n_tokens = x.shape[1]
        block = 2 * self.n_regions
        start = n_tokens - 1 - block
        if start < 1:
            empty = x.new_zeros(x.shape[0], x.shape[1], self.dim)
            return empty, x.new_zeros(()), empty
        idx = torch.arange(start, start + block, device=x.device)

        h = x[:, idx]                                             # [B, M, d]
        m = h.shape[1]
        r = regions[:, idx]                                       # [B, M]

        onehot = F.one_hot(r, num_classes=self.n_regions).to(x.dtype)
        coupling = torch.einsum("bni,ij,bmj->bnm", onehot, prior.to(x.device), onehot)
        rel = (r[:, :, None] - r[:, None, :]).abs().to(x.dtype) / self.n_regions

        u = h.unsqueeze(2).expand(-1, -1, m, -1)                  # [B, M, M, d]
        v = h.unsqueeze(1).expand(-1, m, -1, -1)
        edge_in = torch.cat([u, v, rel.unsqueeze(-1), coupling.unsqueeze(-1)], dim=-1)
        w = self.edge_mlp(edge_in).squeeze(-1)                    # [B, M, M]
        s = 0.5 * (w + w.transpose(1, 2))
        a = 0.5 * (w - w.transpose(1, 2))
        delta = 2.0 * a

        weight = s + self.gamma * torch.tanh(delta)
        eye = torch.eye(m, device=x.device, dtype=torch.bool).unsqueeze(0)
        weight = weight.masked_fill(eye, 0.0)
        norm = weight.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        agg = torch.bmm(weight / norm, h)                         # [B, M, d]
        msg = self.graph_out(torch.cat([h, agg], dim=-1))

        out = x.clone()
        out[:, idx] = out[:, idx] + msg
        return out, s, a

    # --------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        regions: torch.Tensor,
        prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, n, _ = x.shape
        h = self.norm1(x)

        qkv = self.qkv(h).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # [B, H, N, d_h]

        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)   # [B, H, N, N]
        logits = logits + self._bias(regions, prior).unsqueeze(1)

        # additive log-mask *before* softmax (eq. 17); a finite floor keeps rows
        # that would otherwise be fully masked from producing NaN
        if self.keep < 1.0:
            thr = torch.quantile(logits.flatten(2), self.keep, dim=2).view(b, self.heads, 1, 1)
            drop = logits < thr
            drop.scatter_(-1, logits.argmax(dim=-1, keepdim=True), False)
            logits = logits.masked_fill(drop, -1e4)

        attn = torch.softmax(logits, dim=-1)
        if getattr(self, "store_attention", False):
            self.last_attention = attn.detach()
        ctx = (attn @ v).transpose(1, 2).reshape(b, n, self.dim)
        x = x + self.proj(ctx)
        x = x + self.ffn(self.norm2(x))

        x, s, a = self._directed_graph(self.norm3(x), regions, prior)
        return {"tokens": x, "s": s, "a": a}


class SIRStack(nn.Module):
    """``T`` stacked SIR layers; returns the final tokens and graph statistics."""

    def __init__(self, dim: int = 256, depth: int = 6, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SIRLayer(dim=dim, **kwargs) for _ in range(depth)])

    def forward(self, tokens: torch.Tensor, regions: torch.Tensor, prior: torch.Tensor):
        s = a = tokens.new_zeros(())
        for layer in self.layers:
            out = layer(tokens, regions, prior)
            tokens, s, a = out["tokens"], out["s"], out["a"]
        return {"tokens": tokens, "s": s, "a": a}
