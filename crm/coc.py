"""COC 鈥?Concomitance-guided Occlusion Compensation (Sect. III-B, eq. 3-8).

The module turns an RGB face into

* eight compensated AU responses ``a_hat`` (plus the observed ones and the
  per-region visibility ratio, both needed by the loss), and
* the 256-channel compensated map ``V_comp`` = 8 single-AU maps
  + 28 pairwise co-activation maps + 220 texture channels (eq. 15).

Nothing here is learned except the texture projection: C is estimated offline
(``estimate_c.py``) from unoccluded training faces, exactly as the paper does.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regions import MIRROR, region_masks

_EPS = 1e-6


class COCModule(nn.Module):
    """Occlusion compensation with a fixed concomitance prior.

    Args:
        texture_dim: channels of the first Swin stage (220 of the 256 maps).
        patch: spatial size of that stage relative to the input (4 for Swin).
        n_regions: number of AU regions.
        dilate: radius ``delta`` of the mask dilation disc (eq. 5).
        dark_q / flat_q: quantiles used for the two occlusion cues (eq. 5).
        beta: decay of the bilateral reliability term (eq. 7).
    """

    def __init__(
        self,
        texture_dim: int = 128,
        n_regions: int = 8,
        dilate: int = 3,
        dark_q: float = 0.20,
        flat_q: float = 0.20,
        beta: float = 4.0,
    ) -> None:
        super().__init__()
        self.n_regions = n_regions
        self.dilate = dilate
        self.dark_q = dark_q
        self.flat_q = flat_q
        self.beta = beta
        self.texture_proj = nn.Conv2d(texture_dim, 220, kernel_size=1)
        self.register_buffer("response_mean", torch.zeros(n_regions))
        self.register_buffer("response_std", torch.ones(n_regions))
        self.register_buffer("prior", torch.eye(n_regions))

    # ------------------------------------------------------------------ mask
    @staticmethod
    def _gray(x: torch.Tensor) -> torch.Tensor:
        return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]

    def occlusion_mask(self, x: torch.Tensor, face: torch.Tensor | None = None) -> torch.Tensor:
        """Binary occlusion mask from two cues (eq. 5).

        ``Z < kappa_b`` (darker than the subject's own skin statistics) or
        ``r < kappa_g`` (gradient collapses relative to face-wide texture),
        union, restricted to the AU domains ``Omega``, then a disc dilation of
        radius ``delta``.
        """
        gray = self._gray(x)                                    # [B, H, W]
        b, h, w = gray.shape

        k_b = torch.quantile(gray.flatten(1), self.dark_q, dim=1).view(b, 1, 1)
        dark = gray < k_b

        gx = gray[:, :, 1:] - gray[:, :, :-1]
        gy = gray[:, 1:, :] - gray[:, :-1, :]
        gx = F.pad(gx, (1, 0, 0, 0))
        gy = F.pad(gy, (0, 0, 1, 0))
        r = torch.sqrt(gx * gx + gy * gy + _EPS)                # [B, H, W]
        k_g = torch.quantile(r.flatten(1), self.flat_q, dim=1).view(b, 1, 1)
        flat = r < k_g

        mask = (dark | flat).float().unsqueeze(1)               # [B, 1, H, W]
        if face is not None:
            # the criterion is only meaningful inside the AU domains; without
            # this the dark cue marks the whole background as occluded
            mask = mask * face.unsqueeze(0).unsqueeze(0).to(mask.dtype)
        if self.dilate > 0:
            k = 2 * self.dilate + 1
            mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=self.dilate)
        return mask.squeeze(1)                                  # [B, H, W]

    # -------------------------------------------------------------- response
    def au_responses(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
        occ: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raw AU responses ``a_j`` and visibility ratios ``rho_j`` (eq. 6)."""
        gray = self._gray(x)
        gx = F.pad(gray[:, :, 1:] - gray[:, :, :-1], (1, 0, 0, 0))
        gy = F.pad(gray[:, 1:, :] - gray[:, :-1, :], (0, 0, 1, 0))
        grad = torch.sqrt(gx * gx + gy * gy + _EPS)             # [B, H, W]

        m = masks.unsqueeze(0).to(grad.dtype)                   # [1, 8, H, W]
        visible = (1.0 - occ).unsqueeze(1) * m                  # [B, 8, H, W]
        denom = visible.sum(dim=(2, 3)).clamp_min(_EPS)
        raw = (grad.unsqueeze(1) * visible).sum(dim=(2, 3)) / denom     # [B, 8]
        area = m.sum(dim=(2, 3)).clamp_min(_EPS)                        # [1, 8]
        rho = visible.sum(dim=(2, 3)) / area                            # [B, 8]

        # The paper normalises with "dataset statistics". Fixed 0/1 buffers leave
        # tanh saturated for every region (all responses become +-1), so the
        # statistics are tracked as an EMA of the observed batch moments.
        if self.training:
            with torch.no_grad():
                self.response_mean.mul_(0.9).add_(0.1 * raw.mean(dim=0))
                self.response_std.mul_(0.9).add_(
                    0.1 * raw.std(dim=0, unbiased=False).clamp_min(1e-2)
                )
        norm = torch.tanh((raw - self.response_mean) / (self.response_std + _EPS))
        return norm, rho, occ.mean(dim=(1, 2))

    # ---------------------------------------------------------- compensation
    def compensate(
        self,
        a: torch.Tensor,
        rho: torch.Tensor,
        symmetry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convex combination over visible donors (eq. 7) and soft gate (eq. 8).

        ``symmetry`` is the scale-normalised symmetry violation in ``[0, 1]``:
        0 for a frontal face, growing with yaw. It only discounts the mirrored
        donor, so a frontal face trusts the contralateral region fully.
        """
        prior = self.prior.to(a.dtype)
        n = prior.shape[0]

        tau = torch.ones_like(prior).unsqueeze(0).repeat(a.shape[0], 1, 1)
        mirror_idx = torch.tensor([MIRROR[i] for i in range(n)], device=a.device)
        decay = torch.exp(-self.beta * symmetry).view(-1, 1, 1)          # [B,1,1]
        onehot = F.one_hot(mirror_idx, num_classes=n).to(a.dtype)        # [n, n]
        tau = tau - onehot.unsqueeze(0) * (1.0 - decay)                  # [B, n, n]

        omega = prior.unsqueeze(0) * tau                                 # [B, n, n]
        eye = torch.eye(n, device=a.device, dtype=a.dtype).unsqueeze(0)
        omega = omega * (1.0 - eye)                                      # no self-donation

        donors = a.unsqueeze(1).expand(-1, n, -1)                        # [B, n, n]
        num = (omega * donors).sum(dim=2)
        den = omega.sum(dim=2).clamp_min(_EPS)
        a_bar = num / den

        occlusion_ratio = (1.0 - rho).clamp(0.0, 1.0)
        a_hat = (1.0 - occlusion_ratio) * a + occlusion_ratio * a_bar
        return a_hat, omega

    # ------------------------------------------------------------------ splat
    def splat(
        self,
        a_hat: torch.Tensor,
        masks: torch.Tensor,
        texture: torch.Tensor,
    ) -> torch.Tensor:
        """Build ``V_comp`` with 8 + 28 + 220 = 256 channels (eq. 15).

        ``texture`` is the first Swin stage as ``[B, h, w, c]`` (timm returns
        NHWC) or ``[B, c, h, w]``; it is projected to 220 channels.
        """
        expect = self.texture_proj.in_channels
        if texture.shape[1] == expect:
            tex = texture
        elif texture.shape[-1] == expect:
            tex = texture.permute(0, 3, 1, 2).contiguous()
        else:
            tex = texture
        tex = self.texture_proj(tex)                                     # [B, 220, h, w]
        h, w = tex.shape[-2:]

        m = masks.unsqueeze(0).to(tex.dtype)                             # [1, 8, H, W]
        m = F.adaptive_avg_pool2d(m, (h, w))                             # [1, 8, h, w]

        single = a_hat.view(-1, self.n_regions, 1, 1) * m                # [B, 8, h, w]
        pairs = []
        for i in range(self.n_regions):
            for j in range(i + 1, self.n_regions):
                joint = torch.maximum(m[:, i], m[:, j]).unsqueeze(0)   # [1,1,h,w]
                pairs.append(
                    a_hat[:, i].view(-1, 1, 1, 1)
                    * a_hat[:, j].view(-1, 1, 1, 1)
                    * joint
                )
        pair_maps = torch.cat(pairs, dim=1)                              # [B, 28, h, w]

        return torch.cat([single, pair_maps, tex], dim=1)                # [B, 256, h, w]

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        texture: torch.Tensor,
        landmarks: torch.Tensor | None = None,
        symmetry: torch.Tensor | None = None,
        compensate: bool = True,
    ) -> dict[str, torch.Tensor]:
        b, _, h, w = x.shape
        masks = region_masks(h, w, device=x.device, landmarks=landmarks)  # [8, H, W]

        occ = self.occlusion_mask(x, face=masks.max(dim=0).values)        # [B, H, W]
        a, rho, occlusion_ratio = self.au_responses(x, masks, occ)

        if symmetry is None:
            # cheap proxy: how far the visible-mass centroid left the midline
            weight = rho * (1.0 - occlusion_ratio).unsqueeze(1)
            symmetry = (weight * 0.0).sum(dim=1)
        # `compensate=False` is the CRM w/o COR ablation column of Fig. 10
        a_hat, omega = self.compensate(a, rho, symmetry) if compensate else (a, torch.zeros_like(a))
        comp = self.splat(a_hat, masks, texture)

        return {
            "comp": comp,
            "a": a,
            "a_hat": a_hat,
            "rho": rho,
            "occ": occ,
            "omega": omega,
        }

