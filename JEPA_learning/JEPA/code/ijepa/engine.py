"""
The I-JEPA training step, EMA, schedules, and device handling.

The whole method is 20 lines (`jepa_step`).  Everything else in this file is
plumbing.  Read `jepa_step` first.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from .masks import apply_mask


# --------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------
def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device, enabled: bool = True):
    """bf16 autocast on CUDA only.  MPS autocast is still flaky for this
    workload as of torch 2.x, and on CPU it is slower than fp32."""
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


# --------------------------------------------------------------------------
# THE method
# --------------------------------------------------------------------------
def jepa_step(encoder, predictor, target_encoder, imgs, ctx_mask, tgt_masks,
              loss_type: str = "smooth_l1"):
    """One forward pass.  Returns (loss, extra) where extra carries the target
    tensor for diagnostics.

    imgs      [B, C, H, W]
    ctx_mask  [B, n_ctx]
    tgt_masks list of M tensors [B, n_tgt]
    """
    # ---- targets: full grid through the EMA encoder, THEN masked -----------
    # "the target blocks are obtained by masking the OUTPUT of the target-encoder,
    #  not the input.  This distinction is crucial."   Ablation (Table 11,
    #  ViT-H/16, 300ep): masking the output 67.3, masking the input 56.1.
    with torch.no_grad():
        h = target_encoder(imgs)                        # [B, N, D]
        # Stateless LayerNorm over the feature dim.  NOT in the paper; it is in
        # the reference code, and it is load-bearing: it puts the residuals in
        # the quadratic regime of smooth_l1 and keeps the target scale fixed as
        # the EMA encoder drifts.
        h = F.layer_norm(h, (h.size(-1),))
        targets = [apply_mask(h, m) for m in tgt_masks]  # M x [B, n_tgt, D]

    # ---- context: only the visible patches are ever encoded ---------------
    s_x = encoder(imgs, ctx_mask)                       # [B, n_ctx, D]

    # ---- predict each target block, conditioned on its positions ----------
    loss = imgs.new_zeros(())
    for m, t in zip(tgt_masks, targets):
        pred = predictor(s_x, ctx_mask, m)               # [B, n_tgt, D]
        if loss_type == "smooth_l1":
            loss = loss + F.smooth_l1_loss(pred, t)      # what the code does
        elif loss_type == "l2":
            loss = loss + (pred - t).pow(2).sum(-1).mean()   # what the paper says
        elif loss_type == "l1":
            loss = loss + (pred - t).abs().mean()            # what V-JEPA switched to
        else:
            raise ValueError(loss_type)
    loss = loss / len(tgt_masks)
    # detach the diagnostics payload so returning it cannot pin the forward
    # graph past backward()
    return loss, dict(target=h, context=s_x.detach())


# --------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------
def make_target_encoder(encoder):
    """Deep copy with gradients switched off.  The copy is what makes the two
    branches asymmetric, and the asymmetry is what prevents collapse."""
    te = copy.deepcopy(encoder)
    for p in te.parameters():
        p.requires_grad = False
    te.eval()
    return te


@torch.no_grad()
def ema_update(encoder, target_encoder, m: float):
    """theta_bar <- m * theta_bar + (1 - m) * theta.

    m = 1 freezes the target; m = 0 makes it a hard copy (== SimSiam, still
    protected by the stop-gradient).  I-JEPA ramps m linearly 0.996 -> 1.0.
    Only parameters are averaged, not buffers -- the reference does the same.
    """
    for pq, pk in zip(encoder.parameters(), target_encoder.parameters()):
        pk.mul_(m).add_(pq.detach(), alpha=1.0 - m)


# --------------------------------------------------------------------------
# schedules -- all stepped PER ITERATION, not per epoch
# --------------------------------------------------------------------------
def warmup_cosine(step, total, warmup, base, start=None, final=0.0):
    start = base * 0.1 if start is None else start
    if step < warmup:
        return start + (base - start) * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return final + (base - final) * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def linear_ramp(step, total, lo, hi):
    return lo + (hi - lo) * min(step / max(total, 1), 1.0)


def cosine_ramp(step, total, lo, hi):
    """The reference implementation's weight-decay schedule.  Note the paper
    says 'linearly increased from 0.04 to 0.4'; the code uses a cosine."""
    p = min(step / max(total, 1), 1.0)
    return hi + (lo - hi) * 0.5 * (1.0 + math.cos(math.pi * p))


def param_groups(encoder, predictor, weight_decay: float):
    """Exclude biases, all 1-D parameters (LayerNorm scales) and `mask_token`
    from weight decay -- standard for ViTs.

    Note on `mask_token`: it has shape [1, 1, d], so the usual `ndim <= 1` rule
    does NOT catch it, and the reference implementation therefore decays it.
    That is almost certainly unintentional: over a long run at wd 0.4 it drives
    the only parameter distinguishing a mask query from a bare positional
    embedding towards zero.  We exclude it by name.
    """
    decay, no_decay = [], []
    for mod in (encoder, predictor):
        for n, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            skip = p.ndim <= 1 or n.endswith("mask_token")
            (no_decay if skip else decay).append(p)
    return [dict(params=decay, weight_decay=weight_decay, _decay=True),
            dict(params=no_decay, weight_decay=0.0, _decay=False)]


def set_lr_wd(opt, lr: float, wd: float):
    for g in opt.param_groups:
        g["lr"] = lr
        if g.get("_decay", False):
            g["weight_decay"] = wd
