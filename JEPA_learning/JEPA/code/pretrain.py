#!/usr/bin/env python3
"""
Pretrain I-JEPA from scratch on CIFAR-10 (or any ImageFolder).

  python pretrain.py --epochs 60 --batch-size 256
  python pretrain.py --preset tiny-tin64 --data /path/to/tiny-imagenet
  python pretrain.py --smoke                 # 20 steps, verifies every shape

Sensible budgets:
  RTX 3060 (12 GB), nano-cifar, batch 256   ~ 25 s / epoch  -> 60 epochs in ~25 min
  M4 Pro (mps),     nano-cifar, batch 256   ~ 55 s / epoch  -> 60 epochs in ~55 min

Read the printed diagnostics, not the loss.  `std/(1/sqrt(D))` below ~0.3 means
you are collapsing and nothing downstream will work.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ijepa import data, diagnostics, engine
from ijepa.masks import MaskConfig, MultiBlockMaskCollator, apply_mask, context_ratio
from ijepa.vit import build, n_params


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="nano-cifar")
    p.add_argument("--data", default=None,
                   help="dataset root containing train/ and val/; omit for CIFAR-10")
    p.add_argument("--root", default="./data")
    p.add_argument("--out", default="./runs/ijepa")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--warmup-frac", type=float, default=0.15)
    p.add_argument("--wd", type=float, default=0.04)
    p.add_argument("--final-wd", type=float, default=0.4)
    p.add_argument("--ema-start", type=float, default=0.996)
    p.add_argument("--ema-end", type=float, default=1.0)
    p.add_argument("--n-target", type=int, default=4)
    p.add_argument("--target-scale", type=float, nargs=2, default=(0.15, 0.20))
    p.add_argument("--context-scale", type=float, nargs=2, default=(0.85, 1.00))
    p.add_argument("--loss", default="smooth_l1", choices=["smooth_l1", "l2", "l1"])
    p.add_argument("--clip", type=float, default=0.0, help="0 disables grad clipping")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed)
    dev = engine.pick_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    encoder, predictor, cfgp = build(a.preset)
    encoder, predictor = encoder.to(dev), predictor.to(dev)
    target_encoder = engine.make_target_encoder(encoder).to(dev)

    mcfg = MaskConfig(grid=encoder.grid, n_target=a.n_target,
                      target_scale=tuple(a.target_scale),
                      context_scale=tuple(a.context_scale),
                      min_keep=max(2, encoder.n_patches // 16))
    collator = MultiBlockMaskCollator(mcfg, seed=a.seed)

    if a.data:
        # --data points at the DATASET ROOT containing train/ and val/, the same
        # convention probe.py uses, so one path works for both scripts.
        ds = data.imagefolder(a.data + "/train", img_size=cfgp["img_size"], augment=True)
        eval_ds = data.imagefolder(a.data + "/val", img_size=cfgp["img_size"], augment=False)
    else:
        ds = data.cifar10(a.root, img_size=cfgp["img_size"], train=True, augment=True)
        eval_ds = data.cifar10(a.root, img_size=cfgp["img_size"], train=False, augment=False)

    dl = data.loader(ds, a.batch_size, collate_fn=collator, workers=a.workers)
    eval_dl = data.loader(eval_ds, 256, shuffle=False, workers=a.workers, drop_last=False)

    ipe = len(dl)
    total = ipe * a.epochs
    warmup = int(total * a.warmup_frac)
    opt = torch.optim.AdamW(engine.param_groups(encoder, predictor, a.wd), lr=a.lr)

    print(f"device        {dev}")
    print(f"preset        {a.preset}  grid {encoder.grid}x{encoder.grid} "
          f"= {encoder.n_patches} patches")
    print(f"params        encoder {n_params(encoder)/1e6:.2f}M  "
          f"predictor {n_params(predictor)/1e6:.2f}M")
    print(f"masking       M={mcfg.n_target} targets {mcfg.target_scale}, "
          f"context {mcfg.context_scale}, mean |ctx|/N = "
          f"{context_ratio(mcfg, 400):.3f}   (paper reports 0.25 at 14x14)")
    print(f"schedule      {a.epochs} epochs x {ipe} it = {total} it, warmup {warmup}")
    print()

    hist = []
    t0 = time.time()
    step = 0
    for epoch in range(a.epochs):
        encoder.train(); predictor.train()
        run = 0.0
        for batch in dl:
            imgs, ctx_mask, tgt_masks = batch[0], batch[1], batch[2]
            imgs = imgs.to(dev, non_blocking=True)
            ctx_mask = ctx_mask.to(dev)
            tgt_masks = [m.to(dev) for m in tgt_masks]

            engine.set_lr_wd(opt,
                             engine.warmup_cosine(step, total, warmup, a.lr,
                                                  start=a.lr * 0.1, final=a.lr * 1e-3),
                             engine.cosine_ramp(step, total, a.wd, a.final_wd))
            with engine.amp_context(dev):
                loss, _ = engine.jepa_step(encoder, predictor, target_encoder,
                                           imgs, ctx_mask, tgt_masks, a.loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()), a.clip)
            opt.step()
            engine.ema_update(encoder, target_encoder,
                              engine.linear_ramp(step, total, a.ema_start, a.ema_end))
            run += loss.item()
            step += 1
            if a.smoke and step >= 20:
                break
        if a.smoke:
            break

        # ---- diagnostics: pooled TARGET-encoder embeddings on the val split
        target_encoder.eval()
        zs = []
        with torch.no_grad():
            for xb, _ in eval_dl:
                zs.append(target_encoder(xb.to(dev)).mean(1).float().cpu())
                if sum(z.size(0) for z in zs) >= 4096:
                    break
        z = torch.cat(zs)
        rep = diagnostics.report(z)
        rep.update(epoch=epoch, loss=round(run / ipe, 5),
                   mins=round((time.time() - t0) / 60, 1))
        hist.append(rep)
        print(f"ep {epoch:3d}  loss {rep['loss']:9.5f}  "
              f"std/(1/sqrt D) {rep['std_ratio']:.3f}  rankme {rep['rankme']:6.2f}  "
              f"eff_rank {rep['eff_rank']:3d}/{rep['dim']}  "
              f"unif {rep['uniformity']:7.3f}  [{diagnostics.verdict(rep)}]")
        torch.save(dict(encoder=encoder.state_dict(),
                        target_encoder=target_encoder.state_dict(),
                        predictor=predictor.state_dict(),
                        preset=a.preset, args=vars(a), grid=encoder.grid),
                   out / "last.pt")
        json.dump(hist, open(out / "history.json", "w"), indent=1)

    if a.smoke:
        # exercise every shape once, ASSERT FIRST so a shape bug is loud
        assert step > 0, "--smoke needs >= 1 step: check --epochs and dataset size"
        with torch.no_grad():
            h = target_encoder(imgs)
            s_x = encoder(imgs, ctx_mask)
            pr = predictor(s_x, ctx_mask, tgt_masks[0])
        want = (imgs.size(0), tgt_masks[0].size(1), encoder.dim)
        assert tuple(pr.shape) == want, f"predictor gave {tuple(pr.shape)}, want {want}"
        assert pr.shape == apply_mask(h, tgt_masks[0]).shape, \
            "prediction shape != target block slice"
        torch.save(dict(encoder=encoder.state_dict(),
                        target_encoder=target_encoder.state_dict(),
                        predictor=predictor.state_dict(),
                        preset=a.preset, args=vars(a), grid=encoder.grid),
                   out / "smoke.pt")          # exercises the save/load round-trip
        print("SMOKE PASSED")
        print(f"  imgs        {tuple(imgs.shape)}")
        print(f"  ctx_mask    {tuple(ctx_mask.shape)}   (indices into {encoder.n_patches})")
        print(f"  tgt_masks   {len(tgt_masks)} x {tuple(tgt_masks[0].shape)}")
        print(f"  target h    {tuple(h.shape)}")
        print(f"  context s_x {tuple(s_x.shape)}")
        print(f"  prediction  {tuple(pr.shape)}  (must match target block slice)")
        print(f"  loss        {loss.item():.5f}")
        assert pr.shape == (imgs.size(0), tgt_masks[0].size(1), encoder.dim)
    else:
        print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {out/'last.pt'}")
        print("next:  python probe.py --ckpt", out / "last.pt")


if __name__ == "__main__":
    main()
