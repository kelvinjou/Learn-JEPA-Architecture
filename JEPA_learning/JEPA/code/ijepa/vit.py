"""
Vision Transformer encoder + narrow predictor, written from scratch.

Nothing here is imported from timm.  Every tensor shape is annotated because
shape bugs are the entire difficulty of implementing this.

Design decisions and where they come from:

  * Fixed (non-learned) 2-D sin-cos positional embeddings, `requires_grad=False`.
    The paper never says which kind; the reference implementation uses sin-cos,
    inherited from MAE.
  * No [CLS] token.  "I-JEPA is pretrained without a [cls] token.  We use the
    target-encoder for evaluation and average pool its output."  (Appendix A.1)
  * Pre-norm blocks, LayerNorm(eps=1e-6), GELU, mlp_ratio=4, qkv_bias=True,
    trunc_normal_(std=0.02) init, no dropout, no drop-path.  All from the
    reference config; the paper states none of it.
  * The encoder adds positional embeddings BEFORE masking, so a context token
    still knows its absolute position in the image.
  * The predictor keeps its own separate frozen sin-cos table at its own
    (narrower) width, and re-adds position to the context tokens.  Mask tokens
    are one shared learnable vector plus the sin-cos embedding of the position
    being predicted -- that positional add is the ONLY thing distinguishing one
    mask token from another.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# positional embeddings
# --------------------------------------------------------------------------
def sincos_2d(dim: int, grid: int) -> torch.Tensor:
    """Fixed 2-D sin-cos table.  Returns [grid*grid, dim].

    Half the channels encode the row coordinate, half the column.  Within each
    half, `omega_k = 1 / 10000^(2k/(dim/2))` and the channels are
    [sin(pos*omega) | cos(pos*omega)] -- the same construction as the original
    Transformer, applied twice.
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2-D sin-cos"
    d_half = dim // 2

    def _1d(d: int, pos: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(d // 2, dtype=torch.float64) / (d / 2.0)
        omega = 1.0 / (10000.0 ** omega)                       # [d/2]
        out = pos.reshape(-1).double()[:, None] * omega[None]  # [M, d/2]
        return torch.cat([out.sin(), out.cos()], dim=1)        # [M, d]

    r = torch.arange(grid, dtype=torch.float64)
    row, col = torch.meshgrid(r, r, indexing="ij")             # row-major
    emb = torch.cat([_1d(d_half, row), _1d(d_half, col)], dim=1)
    return emb.float()                                         # [grid^2, dim]


# --------------------------------------------------------------------------
# transformer building blocks
# --------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attn: bool = False):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # each [B, h, N, D/h]
        attn = (q @ k.transpose(-2, -1)) * self.scale           # [B, h, N, N]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return (out, attn) if return_attn else (out, None)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, return_attn: bool = False):
        a, attn = self.attn(self.norm1(x), return_attn)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return (x, attn) if return_attn else x


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02, a=-2.0, b=2.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02, a=-2.0, b=2.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------
class ViTEncoder(nn.Module):
    """Patchify -> +pos -> (optional mask) -> blocks -> final LayerNorm.

    Masking AFTER the positional add and BEFORE the blocks is the MAE trick:
    the encoder only ever pays attention cost on the visible tokens.
    """

    def __init__(self, img_size=32, patch_size=4, in_ch=3,
                 dim=192, depth=6, heads=3, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.n_patches = self.grid ** 2
        self.dim = dim
        self.heads = heads
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(in_ch, dim, patch_size, stride=patch_size)
        self.register_buffer("pos_embed", sincos_2d(dim, self.grid).unsqueeze(0),
                             persistent=False)                   # [1, N, D], frozen
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(_init)

    def forward(self, x, mask=None, return_attn=False):
        """x: [B, C, H, W];  mask: [B, K] patch indices or None.
        Returns [B, K, D] (or [B, N, D] when mask is None)."""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)       # [B, N, D]
        x = x + self.pos_embed
        if mask is not None:
            x = torch.gather(x, 1, mask.unsqueeze(-1).expand(-1, -1, self.dim))
        attns = []
        for blk in self.blocks:
            if return_attn:
                x, a = blk(x, return_attn=True)
                attns.append(a)
            else:
                x = blk(x)
        x = self.norm(x)
        return (x, attns) if return_attn else x


# --------------------------------------------------------------------------
# predictor
# --------------------------------------------------------------------------
class Predictor(nn.Module):
    """Narrow ViT: g_phi(s_x, {m_j}) -> predicted target representations.

    Forward pass, exactly as in the reference implementation:
        1. linear-project context tokens from encoder width to predictor width
        2. re-add the predictor's own sin-cos embedding at the context positions
        3. build one token per predicted position: shared learnable `mask_token`
           + sin-cos embedding of that position
        4. concatenate [context ; mask tokens] and run FULL self-attention
           (no causal mask, no cross-attention -- mask tokens see the context
           AND each other)
        5. slice off the mask-token half, LayerNorm, project back to encoder width

    Why a *narrow* predictor?  Ablation (Table 14, ViT-L/16, 600 epochs,
    ImageNet-1% fine-tune): width 384 -> 70.7, width 1024 (= encoder width)
    -> 68.4.  A bottleneck forces the predictor to be a coarse map and stops it
    from memorising the encoder's exact output.  Depth matters the other way
    (Table 12): depth 6 -> 64.0, depth 12 -> 66.9.
    """

    def __init__(self, enc_dim: int, grid: int, dim: int = 96, depth: int = 3, heads: int = 3):
        super().__init__()
        self.dim = dim
        self.embed = nn.Linear(enc_dim, dim, bias=True)
        self.register_buffer("pos_embed", sincos_2d(dim, grid).unsqueeze(0), persistent=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.proj = nn.Linear(dim, enc_dim, bias=True)
        self.apply(_init)
        nn.init.trunc_normal_(self.mask_token, std=0.02, a=-2.0, b=2.0)

    def forward(self, ctx, ctx_mask, tgt_mask):
        """ctx: [B, n_ctx, enc_dim]; ctx_mask: [B, n_ctx]; tgt_mask: [B, n_tgt]
        Returns [B, n_tgt, enc_dim]."""
        B, n_ctx, _ = ctx.shape
        n_tgt = tgt_mask.size(1)

        x = self.embed(ctx)                                              # [B, n_ctx, d]
        pe = self.pos_embed.expand(B, -1, -1)                            # [B, N, d]
        x = x + torch.gather(pe, 1, ctx_mask.unsqueeze(-1).expand(-1, -1, self.dim))

        tgt_pe = torch.gather(pe, 1, tgt_mask.unsqueeze(-1).expand(-1, -1, self.dim))
        tok = self.mask_token.expand(B, n_tgt, -1) + tgt_pe              # [B, n_tgt, d]

        x = torch.cat([x, tok], dim=1)                                   # [B, n_ctx+n_tgt, d]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, n_ctx:]                                      # keep mask half
        return self.proj(x)                                              # [B, n_tgt, enc_dim]


# --------------------------------------------------------------------------
# presets sized for a laptop GPU / M-series Mac
# --------------------------------------------------------------------------
PRESETS = {
    # name          img  patch  dim depth heads  pdim pdepth
    "nano-cifar":  dict(img_size=32, patch_size=4, dim=192, depth=6,  heads=3,
                        pred_dim=96,  pred_depth=3),
    "tiny-cifar":  dict(img_size=32, patch_size=4, dim=192, depth=12, heads=3,
                        pred_dim=96,  pred_depth=6),
    "tiny-tin64":  dict(img_size=64, patch_size=8, dim=192, depth=12, heads=3,
                        pred_dim=96,  pred_depth=6),
    "small-tin64": dict(img_size=64, patch_size=8, dim=384, depth=12, heads=6,
                        pred_dim=192, pred_depth=6),
}


def build(preset: str = "nano-cifar"):
    p = dict(PRESETS[preset])
    enc = ViTEncoder(img_size=p["img_size"], patch_size=p["patch_size"],
                     dim=p["dim"], depth=p["depth"], heads=p["heads"])
    pred = Predictor(enc_dim=p["dim"], grid=enc.grid,
                     dim=p["pred_dim"], depth=p["pred_depth"], heads=p["heads"])
    return enc, pred, p


def n_params(m: nn.Module) -> int:
    return sum(q.numel() for q in m.parameters() if q.requires_grad)
