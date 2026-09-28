#!/usr/bin/env python3
"""
Break it on purpose.  Four variants of the same model, same data, same steps:

  shared    target encoder IS the online encoder, gradients flow through BOTH
            branches.  No stop-gradient.  This is the ablation the paper does
            not run, and it collapses.
  stopgrad  target weights are a hard copy of the online weights each step
            (m = 0), but the target branch is under no_grad.  This is the
            SimSiam configuration.
  ema       stop-gradient + EMA target encoder, m ramping 0.996 -> 1.0.
            The I-JEPA recipe.
  frozen    target encoder never updates (m = 1).  Predicting a random
            network's features.  A useful control: it does not collapse, but
            the ceiling is low.

What to look for.  `shared` reaches a loss two to three ORDERS OF MAGNITUDE
lower than the others and its representation is worthless.  This is the whole
lesson of the chapter: in a joint-embedding architecture the loss is not a
progress signal.

  python collapse_lab.py --steps 1200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ijepa import data, diagnostics, engine
from ijepa.masks import MaskConfig, MultiBlockMaskCollator, apply_mask
from ijepa.vit import build

VARIANTS = ["shared", "stopgrad", "ema", "frozen"]


def step_shared(encoder, predictor, imgs, ctx_mask, tgt_masks):
    """No stop-gradient: the SAME encoder computes both branches and gradients
    flow into the target.  The cheapest way to drive the loss down is now to
    make every embedding identical, and that is what happens."""
    h = encoder(imgs)                                  # <- NO torch.no_grad()
    h = F.layer_norm(h, (h.size(-1),))
    s_x = encoder(imgs, ctx_mask)
    loss = imgs.new_zeros(())
    for m in tgt_masks:
        loss = loss + F.smooth_l1_loss(predictor(s_x, ctx_mask, m), apply_mask(h, m))
    return loss / len(tgt_masks)


def run(variant, args, dev):
    torch.manual_seed(args.seed)
    encoder, predictor, cfgp = build(args.preset)
    encoder, predictor = encoder.to(dev), predictor.to(dev)
    target = engine.make_target_encoder(encoder).to(dev)

    mcfg = MaskConfig(grid=encoder.grid, min_keep=max(2, encoder.n_patches // 16))
    ds = data.cifar10(args.root, cfgp["img_size"], train=True, augment=True)
    ev = data.cifar10(args.root, cfgp["img_size"], train=False, augment=False)
    dl = data.loader(ds, args.batch_size, MultiBlockMaskCollator(mcfg, args.seed),
                     workers=args.workers)
    evl = data.loader(ev, 256, shuffle=False, workers=args.workers, drop_last=False)

    opt = torch.optim.AdamW(engine.param_groups(encoder, predictor, 0.04), lr=args.lr)
    curve, step, t0 = [], 0, time.time()
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            imgs, ctx_mask, tgt_masks = (batch[0].to(dev), batch[1].to(dev),
                                         [m.to(dev) for m in batch[2]])
            engine.set_lr_wd(opt, engine.warmup_cosine(step, args.steps,
                                                       int(0.1 * args.steps), args.lr),
                             0.04)
            if variant == "shared":
                loss = step_shared(encoder, predictor, imgs, ctx_mask, tgt_masks)
            else:
                loss, _ = engine.jepa_step(encoder, predictor, target,
                                           imgs, ctx_mask, tgt_masks)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            if variant == "ema":
                engine.ema_update(encoder, target,
                                  engine.linear_ramp(step, args.steps, 0.996, 1.0))
            elif variant == "stopgrad":
                engine.ema_update(encoder, target, 0.0)     # hard copy
            # 'frozen' -> never update;  'shared' -> no separate target

            if step % args.log_every == 0 or step == args.steps - 1:
                src = encoder if variant == "shared" else target
                was_training = src.training
                src.eval()
                with torch.no_grad():
                    zs, n = [], 0
                    for xb, _ in evl:
                        zs.append(src(xb.to(dev)).mean(1).float().cpu()); n += len(xb)
                        if n >= 2048:
                            break
                z = torch.cat(zs)
                src.train(was_training)     # target encoder must stay in eval()
                rep = diagnostics.report(z)
                curve.append(dict(step=step, loss=loss.item(), **rep))
                print(f"  {variant:9s} step {step:5d}  loss {loss.item():10.5f}  "
                      f"std/(1/sqrt D) {rep['std_ratio']:.4f}  rankme {rep['rankme']:6.2f}  "
                      f"eff_rank {rep['eff_rank']:3d}")
            step += 1
    if not curve:
        print(f"  {variant:9s} no steps run (--steps {args.steps})")
        return curve
    print(f"  {variant:9s} finished in {(time.time()-t0)/60:.1f} min  "
          f"[{diagnostics.verdict(curve[-1])}]")
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--preset", default="nano-cifar")
    p.add_argument("--root", default="./data")
    p.add_argument("--out", default="./runs/collapse")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    dev = engine.pick_device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for v in a.variants:
        print(f"\n=== {v} ===")
        res[v] = run(v, a, dev)
    json.dump(res, open(out / "curves.json", "w"), indent=1)

    print("\n" + "=" * 74)
    print(f"{'variant':10s}{'final loss':>13s}{'std ratio':>12s}{'rankme':>9s}"
          f"{'eff_rank':>10s}   verdict")
    print("=" * 74)
    for v, c in ((k, q) for k, q in res.items() if q):
        f = c[-1]
        print(f"{v:10s}{f['loss']:>13.5f}{f['std_ratio']:>12.4f}{f['rankme']:>9.2f}"
              f"{f['eff_rank']:>10d}   {diagnostics.verdict(f)}")
    print("=" * 74)
    print("Note which variant has the lowest loss.  Now note which one is usable.")
    print(f"curves -> {out/'curves.json'}   (plot with plot_collapse.py)")


if __name__ == "__main__":
    main()
