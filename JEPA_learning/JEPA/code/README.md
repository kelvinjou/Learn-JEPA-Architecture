# I-JEPA from scratch

A single-file-per-concept reimplementation of [I-JEPA](https://arxiv.org/abs/2301.08243)
(Assran et al., ICCV 2023), sized for one consumer GPU or an Apple M-series Mac.
No `timm`, no `transformers` — every layer is written out.

Companion to `JEPA_from_scratch.pdf` in the parent folder. Chapter numbers below
point at the relevant section of that book.

## Install and smoke test

```bash
pip install -r requirements.txt
python pretrain.py --smoke          # 20 steps, prints every tensor shape
```

The smoke test is the first thing to run and the first thing to re-run after any
edit. It exercises the collator, both encoders, the predictor and the loss, and
asserts that the prediction shape matches the target slice.

## The four commands

```bash
# 1. pretrain (Ch. 6).  ~25 min on an RTX 3060, ~55 min on an M4 Pro
python pretrain.py --epochs 60 --batch-size 256

# 2. evaluate without fine-tuning (Ch. 7)
python probe.py --ckpt runs/ijepa/last.pt
python probe.py --ckpt runs/ijepa/last.pt --frac 0.01     # low-shot

# 3. break it on purpose (Ch. 4)
python collapse_lab.py --steps 1200
python plot_collapse.py runs/collapse/curves.json

# 4. look inside (Ch. 8)
python visualize.py --ckpt runs/ijepa/last.pt --what masks
python visualize.py --ckpt runs/ijepa/last.pt --what attention
python visualize.py --ckpt runs/ijepa/last.pt --what neighbours
```

## Layout

| file | what it is |
|---|---|
| `ijepa/vit.py` | ViT encoder, narrow predictor, fixed 2-D sin-cos positions |
| `ijepa/masks.py` | multi-block mask sampler and DataLoader collator |
| `ijepa/engine.py` | `jepa_step` (the whole method, 20 lines), EMA, schedules |
| `ijepa/diagnostics.py` | collapse detection — read these, never the loss |
| `ijepa/data.py` | CIFAR-10 / ImageFolder with I-JEPA's minimal transforms |
| `pretrain.py` | training loop |
| `probe.py` | linear probe + kNN + random-init control |
| `collapse_lab.py` | four ablations of the anti-collapse mechanism |
| `visualize.py` | masks, attention maps, nearest neighbours |

## Presets

| preset | input | grid | encoder | predictor | fits in |
|---|---|---|---|---|---|
| `nano-cifar` | 32×32 /4 | 8×8 | 192d × 6 | 96d × 3 | ~2 GB |
| `tiny-cifar` | 32×32 /4 | 8×8 | 192d × 12 | 96d × 6 | ~3 GB |
| `tiny-tin64` | 64×64 /8 | 8×8 | 192d × 12 | 96d × 6 | ~4 GB |
| `small-tin64` | 64×64 /8 | 8×8 | 384d × 12 | 192d × 6 | ~8 GB |

`--device` is auto-detected (`cuda` → `mps` → `cpu`). bf16 autocast is enabled on
CUDA only; MPS autocast is still unreliable for this workload.

## Reading the output

The JEPA loss is **not** a progress signal. A collapsed model reaches a loss two
to three orders of magnitude below a healthy one. Watch instead:

- `std/(1/sqrt D)` — below 0.3 you are collapsing; healthy is 0.7–1.0
- `eff_rank` — how many embedding directions carry signal, out of `dim`
- `probe.py`'s **random-init row** — if you do not beat an untrained ViT of the
  same architecture, you have learned nothing, whatever the loss did

## Deliberate deviations from the released Meta code

Documented in the source at each site, and in Ch. 5 of the book:

- scale and aspect ratio are drawn independently (upstream reuses one sample for
  both, correlating them). Set `couple_scale_and_ar=True` in `MaskConfig` to
  reproduce upstream exactly.
- distributed plumbing (`AllReduce`, `repeat_interleave_batch`, DDP) is removed;
  the latter is a no-op at `num_enc_masks = 1`, which is every shipped config.
- the BEiT-style `fix_init_weight` rescale is dropped, because upstream's
  `init_model` silently overwrites it anyway.
- `--loss` selects `smooth_l1` (what the released code does), `l2` (what the
  paper's equation says) or `l1` (what V-JEPA switched to).
