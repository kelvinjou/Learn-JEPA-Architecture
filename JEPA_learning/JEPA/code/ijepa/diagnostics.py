"""
Collapse diagnostics.  Run these every epoch; they are cheap and they are the
only thing standing between you and a week of training a constant function.

The JEPA loss is NOT a progress signal.  A collapsed model has a lower loss
than a healthy one, by orders of magnitude.  You need the four numbers below.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def embed_std(z: torch.Tensor) -> float:
    """Mean per-channel std of L2-normalised embeddings.  z: [N, D].

    SimSiam's monitor.  For embeddings scattered isotropically on the unit
    sphere the value is ~= 1/sqrt(D); for a constant embedding it is 0.
    Report it as a RATIO to 1/sqrt(D) so it is comparable across widths.
    Thresholds used by `verdict` below, on the RATIO to 1/sqrt(D):
        > 0.30   healthy enough to probe
        0.05-0.30 partial / dimensional collapse
        < 0.05   collapsed
    """
    zn = z / (z.norm(dim=1, keepdim=True) + 1e-12)
    return zn.std(dim=0).mean().item()


@torch.no_grad()
def embed_std_ratio(z: torch.Tensor) -> float:
    return embed_std(z) * math.sqrt(z.size(1))


@torch.no_grad()
def rankme(z: torch.Tensor, eps: float = 1e-7) -> float:
    """RankMe: exp(Shannon entropy of the L1-normalised singular spectrum).

    p_k = sigma_k / ||sigma||_1 + eps ;  RankMe = exp(-sum p_k log p_k).
    No centring, no mean subtraction -- that is deliberate in the original.
    High rank is NECESSARY but not sufficient for a good representation.

    Caveat worth knowing: RankMe can stay misleadingly high on a numerically
    collapsed model, because it is scale-free -- it sees the SHAPE of the
    spectrum, not its magnitude.  Always read it next to `embed_std_ratio`.
    """
    s = torch.linalg.svdvals(z.float())
    p = s / (s.abs().sum() + 1e-12) + eps
    return torch.exp(-(p * p.log()).sum()).item()


@torch.no_grad()
def spectrum(z: torch.Tensor) -> torch.Tensor:
    """Sorted singular values of the CENTRED embedding matrix.  Plot log-scale;
    collapsed directions show up as a cliff at the right-hand end."""
    zc = z.float() - z.float().mean(0, keepdim=True)
    return torch.linalg.svdvals(zc)


@torch.no_grad()
def effective_rank(z: torch.Tensor, thresh: float = 0.01, floor: float = 1e-4) -> int:
    """Count of singular values above `thresh` x the largest.  Blunter than
    RankMe but easier to interpret: 'how many directions actually carry signal'.

    A purely relative threshold is scale-blind, so a numerically collapsed
    encoder whose whole spectrum is float noise would score a high rank.  The
    `floor` (absolute, relative to the per-sample scale) guards against that."""
    s = spectrum(z)
    absolute = floor * math.sqrt(z.size(0))
    return int((s > max(thresh * s[0].item(), absolute)).sum().item())


@torch.no_grad()
def alignment(za: torch.Tensor, zb: torch.Tensor, alpha: float = 2.0) -> float:
    """E||f(x) - f(y)||^alpha over positive pairs (Wang & Isola).  Lower is
    more invariant.  A collapsed encoder scores a perfect 0 -- which is exactly
    why you must read it alongside `uniformity`."""
    a = za / (za.norm(dim=1, keepdim=True) + 1e-12)
    b = zb / (zb.norm(dim=1, keepdim=True) + 1e-12)
    return (a - b).norm(dim=1).pow(alpha).mean().item()


@torch.no_grad()
def uniformity(z: torch.Tensor, t: float = 2.0, n: int = 2048) -> float:
    """log E[exp(-t ||f(x)-f(y)||^2)] over independent pairs (Wang & Isola).
    Lower (more negative) = more spread out.  A collapsed encoder scores 0,
    the worst possible value."""
    z = z[:n]
    zn = z / (z.norm(dim=1, keepdim=True) + 1e-12)
    d2 = torch.cdist(zn, zn).pow(2)
    off = ~torch.eye(len(zn), dtype=torch.bool, device=z.device)
    return torch.log(torch.exp(-t * d2[off]).mean()).item()


@torch.no_grad()
def report(z: torch.Tensor) -> dict:
    """One call, all diagnostics.  z: [N, D] pooled embeddings."""
    s = spectrum(z)
    return dict(
        std_ratio=round(embed_std_ratio(z), 6),   # 4 dp rounds a collapse to 0.0
        rankme=round(rankme(z), 2),
        eff_rank=effective_rank(z),
        dim=z.size(1),
        sv_decay=round((s[0] / (s[min(7, len(s) - 1)] + 1e-12)).item(), 2),
        uniformity=round(uniformity(z), 4),
    )


def verdict(rep: dict) -> str:
    """Blunt human-readable call.  Not a substitute for a linear probe."""
    r = rep["std_ratio"]
    if r < 0.05:
        return "COLLAPSED - the encoder is close to a constant function"
    if r < 0.30 or rep["eff_rank"] < 0.15 * rep["dim"]:  # thresholds: see embed_std
        return "PARTIAL / DIMENSIONAL COLLAPSE - most directions are dead"
    if rep["eff_rank"] < 0.5 * rep["dim"]:
        return "usable, but the spectrum is concentrated"
    return "healthy"
