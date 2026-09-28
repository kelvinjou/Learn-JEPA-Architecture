#!/usr/bin/env python3
"""
Evaluate a pretrained encoder without fine-tuning it.

  python probe.py --ckpt runs/ijepa/last.pt
  python probe.py --ckpt runs/ijepa/last.pt --frac 0.01     # low-shot

Three evaluations, because any one of them can lie to you:

  linear probe   frozen encoder + one linear layer.  The standard number.
  kNN            no learned parameters at all, so it cannot paper over a bad
                 representation with a well-tuned head.
  random baseline the SAME architecture at initialisation.  If you do not beat
                 this, you have learned nothing, whatever the loss did.

Always compare against the random baseline.  A randomly initialised ViT is a
surprisingly strong feature extractor on CIFAR-10, and forgetting this is the
single most common way to fool yourself in self-supervised learning.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from ijepa import data, diagnostics, engine
from ijepa.vit import ViTEncoder, build


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--which", default="target_encoder",
                   choices=["target_encoder", "encoder"],
                   help="paper evaluates the TARGET encoder")
    p.add_argument("--data", default=None,
                   help="dataset root containing train/ and val/; omit for CIFAR-10")
    p.add_argument("--root", default="./data")
    p.add_argument("--frac", type=float, default=1.0, help="fraction of train labels")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def extract(enc, dl, dev):
    enc.eval()
    Z, Y = [], []
    for xb, yb in dl:
        z = enc(xb.to(dev)).mean(1)          # average-pool patch tokens
        Z.append(z.float().cpu()); Y.append(yb)
    return torch.cat(Z), torch.cat(Y)


def linear_probe(Ztr, Ytr, Zte, Yte, n_cls, epochs, lr, bs, dev):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    Ztr, Zte = ((Ztr - mu) / sd).to(dev), ((Zte - mu) / sd).to(dev)
    Ytr, Yte = Ytr.to(dev), Yte.to(dev)
    head = torch.nn.Linear(Ztr.size(1), n_cls).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        perm = torch.randperm(len(Ztr), device=dev)
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(head(Ztr[j]), Ytr[j])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
    with torch.no_grad():
        return (head(Zte).argmax(1) == Yte).float().mean().item()


def knn(Ztr, Ytr, Zte, Yte, k, n_cls, dev, T=0.07):
    """Cosine-similarity weighted kNN, as used in DINO/SimSiam monitors."""
    a = F.normalize(Ztr.to(dev), dim=1)
    b = F.normalize(Zte.to(dev), dim=1)
    Ytr, Yte = Ytr.to(dev), Yte.to(dev)
    correct = 0
    for i in range(0, len(b), 512):
        sim = b[i:i + 512] @ a.T                       # [n, Ntr]
        sv, si = sim.topk(min(k, sim.size(1)), dim=1)
        w = (sv / T).exp()
        oh = F.one_hot(Ytr[si], n_cls).float()         # [n, k, C]
        correct += ((oh * w.unsqueeze(-1)).sum(1).argmax(1) == Yte[i:i + 512]).sum().item()
    return correct / len(b)


def main():
    a = parse()
    torch.manual_seed(0)          # the random-init control must be reproducible
    dev = engine.pick_device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    enc, _, cfgp = build(ck["preset"])
    enc.load_state_dict(ck[a.which]); enc = enc.to(dev)

    if a.data:
        tr = data.imagefolder(a.data + "/train", cfgp["img_size"], augment=False)
        te = data.imagefolder(a.data + "/val", cfgp["img_size"], augment=False)
    else:
        tr = data.cifar10(a.root, cfgp["img_size"], train=True, augment=False)
        te = data.cifar10(a.root, cfgp["img_size"], train=False, augment=False)
    n_cls = len(tr.classes)
    dtr = data.loader(tr, 256, shuffle=False, workers=a.workers, drop_last=False)
    dte = data.loader(te, 256, shuffle=False, workers=a.workers, drop_last=False)

    Ztr, Ytr = extract(enc, dtr, dev)
    Zte, Yte = extract(enc, dte, dev)
    if a.frac < 1.0:
        g = torch.Generator().manual_seed(0)
        keep = torch.randperm(len(Ztr), generator=g)[:max(n_cls * 2, int(a.frac * len(Ztr)))]
        Ztr, Ytr = Ztr[keep], Ytr[keep]

    rep = diagnostics.report(Zte)
    lin = linear_probe(Ztr, Ytr, Zte, Yte, n_cls, a.epochs, a.lr, a.batch_size, dev)
    kn = knn(Ztr, Ytr, Zte, Yte, a.knn_k, n_cls, dev)

    # ---- the control: identical architecture, no training -----------------
    rnd = ViTEncoder(img_size=cfgp["img_size"], patch_size=cfgp["patch_size"],
                     dim=cfgp["dim"], depth=cfgp["depth"], heads=cfgp["heads"]).to(dev)
    Rtr, _ = extract(rnd, dtr, dev); Rte, _ = extract(rnd, dte, dev)
    if a.frac < 1.0:
        Rtr = Rtr[keep]
    rlin = linear_probe(Rtr, Ytr, Rte, Yte, n_cls, a.epochs, a.lr, a.batch_size, dev)
    rkn = knn(Rtr, Ytr, Rte, Yte, a.knn_k, n_cls, dev)

    print(f"checkpoint     {a.ckpt}  ({a.which})")
    print(f"labels used    {len(Ztr)} / {len(tr)}  ({len(Ztr)/len(tr):.1%})")
    print(f"diagnostics    {rep}")
    print(f"verdict        {diagnostics.verdict(rep)}")
    print()
    print(f"{'':16s}{'linear':>9s}{'kNN':>9s}")
    print(f"{'chance':16s}{1/n_cls:>9.2%}{1/n_cls:>9.2%}")
    print(f"{'random init':16s}{rlin:>9.2%}{rkn:>9.2%}")
    print(f"{'I-JEPA':16s}{lin:>9.2%}{kn:>9.2%}")
    print(f"{'delta':16s}{lin-rlin:>+9.2%}{kn-rkn:>+9.2%}")


if __name__ == "__main__":
    main()
