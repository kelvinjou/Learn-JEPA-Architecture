#!/usr/bin/env python3
"""
Look inside the trained model (Lecture 6 territory: probing, not training).

  python visualize.py --ckpt runs/ijepa/last.pt --what masks
  python visualize.py --ckpt runs/ijepa/last.pt --what attention
  python visualize.py --ckpt runs/ijepa/last.pt --what neighbours

masks       what the context and target encoders actually see
attention   last-block attention, averaged over heads, from a chosen patch
neighbours  nearest neighbours in embedding space -- the honest qualitative
            check.  If the neighbours of a horse are other horses, you have
            learned something semantic; if they share only background colour,
            you have learned a colour histogram.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ijepa import data, engine
from ijepa.masks import MaskConfig, MultiBlockMaskCollator
from ijepa.vit import build

MEAN = np.array(data.CIFAR_MEAN); STD = np.array(data.CIFAR_STD)


def show(img_t):
    x = img_t.permute(1, 2, 0).cpu().numpy() * STD + MEAN
    return np.clip(x, 0, 1)


def get_dataset(ck, root, img_size):
    """Match the dataset the checkpoint was pretrained on.  Feeding CIFAR to a
    Tiny-ImageNet encoder would not crash -- Resize hides it -- it would just
    show you the wrong pictures with the wrong normalisation."""
    src = (ck.get("args") or {}).get("data")
    if src:
        global MEAN, STD
        MEAN, STD = np.array(data.IN_MEAN), np.array(data.IN_STD)
        return data.imagefolder(src + "/val", img_size, augment=False)
    return data.cifar10(root, img_size, train=False, augment=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--what", default="masks", choices=["masks", "attention", "neighbours"])
    p.add_argument("--root", default="./data")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    dev = engine.pick_device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    enc, _, cfgp = build(ck["preset"])
    enc.load_state_dict(ck["target_encoder"]); enc = enc.to(dev).eval()
    G, P = enc.grid, enc.patch_size

    if a.what == "masks":
        ds = get_dataset(ck, a.root, cfgp["img_size"])
        mcfg = MaskConfig(grid=G, min_keep=max(2, enc.n_patches // 16))
        col = MultiBlockMaskCollator(mcfg, seed=3)
        imgs, ctx, tgts = col([ds[i] for i in range(a.n)])[:3]
        fig, ax = plt.subplots(2, a.n, figsize=(1.5 * a.n, 3.2), squeeze=False)
        for i in range(a.n):
            ax[0, i].imshow(show(imgs[i])); ax[0, i].axis("off")
            m = np.zeros(enc.n_patches)
            m[ctx[i].numpy()] = 1.0
            vis = show(imgs[i]) * np.kron(m.reshape(G, G), np.ones((P, P)))[..., None]
            ax[1, i].imshow(vis)
            for t, c in zip(tgts, ["r", "orange", "m", "lime"]):
                for j in t[i].numpy():
                    ax[1, i].add_patch(plt.Rectangle((j % G * P - .5, j // G * P - .5),
                                                     P, P, fill=False, ec=c, lw=.7))
            ax[1, i].axis("off")
        ax[0, 0].set_title("image", fontsize=7, loc="left")
        ax[1, 0].set_title("context (lit) + target blocks (boxes)", fontsize=7, loc="left")
        fig.tight_layout(); fig.savefig("viz_masks.png", dpi=170)
        print("wrote viz_masks.png")

    elif a.what == "attention":
        ds = get_dataset(ck, a.root, cfgp["img_size"])
        xb = torch.stack([ds[i][0] for i in range(a.n)]).to(dev)
        with torch.no_grad():
            _, attns = enc(xb, return_attn=True)
        A = attns[-1].mean(1)                                   # [B, N, N] head-mean
        q = (G // 2) * G + G // 2                                 # the centre patch
        fig, ax = plt.subplots(2, a.n, figsize=(1.5 * a.n, 3.2), squeeze=False)
        for i in range(a.n):
            ax[0, i].imshow(show(xb[i]))
            ax[0, i].add_patch(plt.Rectangle((q % G * P - .5, q // G * P - .5), P, P,
                                             fill=False, ec="cyan", lw=1.2))
            ax[0, i].axis("off")
            m = A[i, q].reshape(G, G).cpu().numpy()
            ax[1, i].imshow(show(xb[i])); ax[1, i].imshow(
                np.kron(m, np.ones((P, P))), cmap="inferno", alpha=.65)
            ax[1, i].axis("off")
        ax[0, 0].set_title("query patch (cyan)", fontsize=7, loc="left")
        ax[1, 0].set_title("last-block attention from it", fontsize=7, loc="left")
        fig.tight_layout(); fig.savefig("viz_attention.png", dpi=170)
        print("wrote viz_attention.png")

    else:
        ds = get_dataset(ck, a.root, cfgp["img_size"])
        dl = data.loader(ds, 256, shuffle=False, workers=2, drop_last=False)
        Z, X, Y = [], [], []
        with torch.no_grad():
            for xb, yb in dl:
                Z.append(enc(xb.to(dev)).mean(1).float().cpu()); X.append(xb); Y.append(yb)
                if sum(len(z) for z in Z) >= 2000:
                    break
        Z = F.normalize(torch.cat(Z), dim=1); X = torch.cat(X); Y = torch.cat(Y)
        sim = Z @ Z.T; sim.fill_diagonal_(-2)
        _, nn = sim.topk(5, dim=1)
        fig, ax = plt.subplots(a.n, 6, figsize=(7.5, 1.25 * a.n), squeeze=False)
        hit = 0
        for r in range(a.n):
            i = (r * 37) % len(X)
            ax[r, 0].imshow(show(X[i])); ax[r, 0].axis("off")
            ax[r, 0].set_ylabel(ds.classes[Y[i]], fontsize=6)
            ax[r, 0].set_title(ds.classes[Y[i]], fontsize=6, loc="left")
            for c in range(5):
                j = nn[i, c]
                ok = Y[j] == Y[i]; hit += int(ok)
                ax[r, c + 1].imshow(show(X[j])); ax[r, c + 1].axis("off")
                ax[r, c + 1].set_title(ds.classes[Y[j]], fontsize=6,
                                       color="green" if ok else "red", loc="left")
        fig.suptitle(f"query | 5 nearest neighbours   "
                     f"(same-class rate on shown rows: {hit/(a.n*5):.0%})", fontsize=8)
        fig.tight_layout(); fig.savefig("viz_neighbours.png", dpi=170)
        print("wrote viz_neighbours.png")


if __name__ == "__main__":
    main()
