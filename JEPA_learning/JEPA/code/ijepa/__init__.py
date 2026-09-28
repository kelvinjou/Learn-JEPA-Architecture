"""
I-JEPA from scratch.

A minimal, readable, single-GPU reimplementation of
  Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas,
  "Self-Supervised Learning from Images with a Joint-Embedding Predictive
  Architecture", ICCV 2023.  https://arxiv.org/abs/2301.08243

Module map:
  vit.py          ViT encoder, narrow predictor, fixed 2-D sin-cos positions
  masks.py        multi-block mask sampler + collator
  engine.py       the training step, EMA, schedules, device selection
  diagnostics.py  collapse detection -- read these, not the loss
  data.py         CIFAR-10 / ImageFolder with I-JEPA's minimal transforms
"""

__all__ = ["vit", "masks", "engine", "diagnostics", "data"]
__version__ = "1.0.0"
