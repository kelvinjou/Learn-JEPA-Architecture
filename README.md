# Learn-JEPA-Architecture
Claude Opus 4.8 generated cirriculum for learning JEPA from scratch

# JEPA from Scratch

A self-study curriculum on Joint-Embedding Predictive Architectures, built from the
primary literature. Start with the PDF.

| | |
|---|---|
| **`JEPA_from_scratch.pdf`** | The textbook. 37 pages, 15 chapters, ~40 footnote citations pointing at specific tables and sections. |
| **`interactive/index.html`** | Four interactive pages. Open this first; it links to the others. |
| **`code/`** | A from-scratch PyTorch I-JEPA sized for your RTX 3060 or M4 Pro. Start with `code/README.md`. |
| **`exp/`** | The numpy experiments behind Figures 5.1, 5.2 and Table 6.1, plus the mask-sampler test suite. |
| **`figures/`** | The figures from the book, as standalone PDFs. |

## Suggested path

1. **Chapters 1–3** — the argument for predicting in latent space, and a targeted
   transformer refresher. Chapter 3 is skippable if attention is already familiar.
2. **Run the collapse lab before reading Chapter 5.** Either
   `interactive/collapse-simulator.html` (ten seconds, in the browser) or
   `python code/collapse_lab.py --steps 1200` (about eight minutes on the 3060).
   The theory in Chapter 5 lands much harder once you have watched a loss go to
   zero while the representation died.
3. **Chapters 4–6** — the mathematical centre of the book. Collapse as a global
   minimum, the linear-model derivation of why stop-gradient and EMA work, and
   how to measure collapse.
4. **Chapters 7–11 alongside `code/`** — build it, train it, evaluate it honestly.
5. **Chapters 12–14** — V-JEPA, V-JEPA 2, latent-space planning, and five research
   projects that are actually open.

## What is verified, and how

Nothing in the book is asserted from memory. Specifically:

- **Every table of numbers** was checked cell-by-cell against the source paper, and
  every quotation against the source text. Where the paper and the released Meta code
  disagree — and they do, in four places that change results — both are stated and the
  difference is flagged (Chapter 7, Table 7.2).
- **Figures 5.1 and 5.2** are my own numerical integration of Tian, Chen & Ganguli's
  published equations (`exp/tian_dynamics.py`). The predicted collapse-basin edge
  p*₋ = 0.05445 is reproduced to four significant figures.
- **Table 6.1** is a real experiment: a minimal JEPA with hand-derived gradients,
  four target-branch variants, in `exp/mini_jepa_numpy.py`.
- **The mask-sampler measurements** in Chapter 8 come from `exp/test_masks_numpy.py`,
  which also asserts that context and target blocks never overlap and that all indices
  are in range. The browser version reproduces them to within 0.01.
- **The browser collapse simulator's claims** were verified headlessly: the
  no-stop-gradient variant reaches a loss ~500× lower than the others, collapses to a
  std ratio of 0.016 against a healthy 0.79, and probes no better than an untrained
  network of the same architecture.
- **The PyTorch code** passed a line-by-line shape and correctness audit. Where it
  deviates from the released Meta implementation, the deviation is documented at the
  site and listed in `code/README.md`.

Two things to be aware of. The PyTorch training runs have not been executed
end-to-end — the code is statically verified and the smoke test asserts every shape,
but the first real run is yours. And every claim carries a citation precisely so you
can check it; if you find something I got wrong, I would rather you found it than
believed it.

## Primary sources

- I-JEPA — <https://arxiv.org/abs/2301.08243> · code (archived) <https://github.com/facebookresearch/ijepa>
- V-JEPA — <https://arxiv.org/abs/2404.08471>
- V-JEPA 2 — <https://arxiv.org/abs/2506.09985>
- LeCun, *A Path Towards Autonomous Machine Intelligence* — <https://openreview.net/pdf?id=BZ5a1r-kVsf>
- Tian, Chen & Ganguli — <https://arxiv.org/abs/2102.06810>
- SimSiam — <https://arxiv.org/abs/2011.10566>

Full bibliography in Chapter 15 of the PDF.
