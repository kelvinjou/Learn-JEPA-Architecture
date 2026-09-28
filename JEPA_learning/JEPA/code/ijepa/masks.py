"""
Multi-block masking for I-JEPA.

This is the single most important design choice in the paper.  Ablation
(Table 6, ViT-B/16, 300 epochs, ImageNet-1% linear probe):

    multi-block  54.2      <- 4 target blocks, scale (0.15,0.2), context (0.85,1.0)
    block        20.2      <- 1 target block of scale 0.6, context = complement
    random       17.6      <- random patches, context = complement
    rasterized   15.5      <- 1 quadrant predicts the other 3

The recipe, verbatim from the paper (Sec. 3 + Fig. 4):
  1. Sample M = 4 (possibly overlapping) TARGET blocks, scale in (0.15, 0.20)
     of the patch grid, aspect ratio in (0.75, 1.50).
  2. Sample ONE CONTEXT block, scale in (0.85, 1.00), unit aspect ratio.
  3. Delete from the context every patch that belongs to any target block.

Two details that only exist in the reference implementation
(facebookresearch/ijepa, src/masks/multiblock.py), not in the paper:

  * Block DIMENSIONS are drawn once per mini-batch from a shared seeded
    generator, so every image in the batch gets identically-shaped blocks and
    the tensors stack cleanly.  Only block POSITIONS vary per image.
  * Because the context is a set difference, its cardinality still varies per
    image.  The reference collator fixes this by TRUNCATING every index list
    to the batch minimum, in raster order.  We keep that behaviour (and warn
    about it) because it is what the released checkpoints were trained with.

Masks are integer index tensors of shape [B, n_keep], not booleans -- the
model gathers with them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class MaskConfig:
    grid: int                          # patch-grid side, e.g. 224/16 = 14
    n_target: int = 4                  # M in the paper
    target_scale: tuple = (0.15, 0.20)
    target_ar: tuple = (0.75, 1.50)
    context_scale: tuple = (0.85, 1.00)
    context_ar: tuple = (1.0, 1.0)     # "unit aspect ratio"
    min_keep: int = 4                  # reference uses 10 at grid=16
    allow_overlap: bool = False        # if True, context may contain targets
    couple_scale_and_ar: bool = False  # True reproduces an upstream quirk (see below)


def _sample_block_size(cfg: MaskConfig, scale, ar, g: torch.Generator) -> tuple:
    """Draw (h, w) in patch units for one block.

    The reference implementation draws ONE uniform sample and uses it for both
    the area scale and the aspect ratio, which perfectly correlates them --
    bigger target blocks are always proportionally taller.  Almost certainly
    unintended.  `couple_scale_and_ar=True` reproduces it bit-for-bit; the
    default draws them independently, which is what the paper describes.
    """
    if cfg.couple_scale_and_ar:
        r = torch.rand(1, generator=g).item()
        r_scale = r_ar = r
    else:
        r_scale = torch.rand(1, generator=g).item()
        r_ar = torch.rand(1, generator=g).item()

    area = int(cfg.grid * cfg.grid * (scale[0] + r_scale * (scale[1] - scale[0])))
    aspect = ar[0] + r_ar * (ar[1] - ar[0])
    h = int(round(math.sqrt(area * aspect)))
    w = int(round(math.sqrt(area / aspect)))
    # Upstream shrinks with `while h >= self.height: h -= 1`, i.e. it enforces
    # h < grid strictly, so even a scale of 1.0 never covers the full grid.  We
    # match that, because it is what the reported context ratios were measured
    # with -- see `context_ratio` and Chapter 8 of the book.  (The strictness is
    # not actually required by `_place`, which uses randint(0, grid - h + 1).)
    h = max(1, min(h, cfg.grid - 1))
    w = max(1, min(w, cfg.grid - 1))
    return h, w


def _place(cfg: MaskConfig, hw, g: torch.Generator) -> torch.Tensor:
    """Random top-left corner -> boolean [grid, grid] occupancy map."""
    h, w = hw
    top = torch.randint(0, cfg.grid - h + 1, (1,), generator=g).item()
    left = torch.randint(0, cfg.grid - w + 1, (1,), generator=g).item()
    m = torch.zeros(cfg.grid, cfg.grid, dtype=torch.bool)
    m[top:top + h, left:left + w] = True
    return m


def sample_masks_for_image(cfg: MaskConfig, tgt_hw, ctx_hw, g: torch.Generator):
    """Masks for ONE image, given block shapes already fixed for the batch.

    Returns (targets, context) where `targets` is a list of M 1-D LongTensors
    of flat patch indices and `context` is one 1-D LongTensor.
    """
    for _ in range(100):                              # bounded retry, cf. min_keep
        occupied = torch.zeros(cfg.grid, cfg.grid, dtype=torch.bool)
        targets = []
        for _ in range(cfg.n_target):
            m = _place(cfg, tgt_hw, g)
            targets.append(m.flatten().nonzero(as_tuple=True)[0])
            occupied |= m
        ctx = _place(cfg, ctx_hw, g)
        if not cfg.allow_overlap:
            ctx = ctx & ~occupied                     # <- step 3: remove overlap
        ctx_idx = ctx.flatten().nonzero(as_tuple=True)[0]
        if len(ctx_idx) > cfg.min_keep and (not targets or min(len(t) for t in targets) > 0):
            return targets, ctx_idx
    # Degenerate grid (too small for these scales).  Fall back to the full
    # complement of the target blocks, which is the largest legal context.
    ctx_idx = (~occupied).flatten().nonzero(as_tuple=True)[0]
    return targets, ctx_idx


class MultiBlockMaskCollator:
    """DataLoader `collate_fn` that emits (images, context_mask, target_masks).

    Shapes returned:
        images        [B, C, H, W]
        context_mask  [B, n_ctx]                      LongTensor of patch indices
        target_masks  list of M tensors, each [B, n_tgt]

    n_ctx and n_tgt are the per-batch minima (see module docstring).
    """

    def __init__(self, cfg: MaskConfig, seed: int = 0):
        self.cfg = cfg
        self._step = 0
        self._seed = seed

    def __call__(self, batch):
        imgs = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch]) if len(batch[0]) > 1 else None

        # One generator per batch -> identical block SHAPES across the batch.
        # Each DataLoader worker holds its own copy of this object and its own
        # `_step`, so we fold the worker id into the seed; without it, W workers
        # would replay the same sequence of block shapes W times over.
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id
        nw = 1 if info is None else info.num_workers
        g = torch.Generator().manual_seed(self._seed + (self._step * nw + wid) * 9973)
        self._step += 1
        cfg = self.cfg
        tgt_hw = _sample_block_size(cfg, cfg.target_scale, cfg.target_ar, g)
        ctx_hw = _sample_block_size(cfg, cfg.context_scale, cfg.context_ar, g)

        per_img = [sample_masks_for_image(cfg, tgt_hw, ctx_hw, g) for _ in imgs]

        n_ctx = min(len(c) for _, c in per_img)
        n_tgt = min(min(len(t) for t in ts) for ts, _ in per_img)

        context = torch.stack([c[:n_ctx] for _, c in per_img])              # [B, n_ctx]
        targets = [torch.stack([ts[i][:n_tgt] for ts, _ in per_img])        # [B, n_tgt]
                   for i in range(cfg.n_target)]
        return (imgs, context, targets) if labels is None else (imgs, context, targets, labels)


def apply_mask(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather patch tokens.  x: [B, N, D], idx: [B, K] -> [B, K, D]."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def context_ratio(cfg: MaskConfig, n_trials: int = 2000, seed: int = 0) -> float:
    """Average |context| / N.  The paper reports 0.25 for its default config
    (Table 6, 'Avg. Ratio').  Useful sanity check when you change the grid."""
    g = torch.Generator().manual_seed(seed)
    tot = 0
    for _ in range(n_trials):
        tgt_hw = _sample_block_size(cfg, cfg.target_scale, cfg.target_ar, g)
        ctx_hw = _sample_block_size(cfg, cfg.context_scale, cfg.context_ar, g)
        _, c = sample_masks_for_image(cfg, tgt_hw, ctx_hw, g)
        tot += len(c)
    return tot / (n_trials * cfg.grid * cfg.grid)
