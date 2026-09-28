#!/usr/bin/env python3
"""Plot the output of collapse_lab.py.   python plot_collapse.py runs/collapse/curves.json"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"shared": "#c0392b", "stopgrad": "#2980b9", "ema": "#27ae60", "frozen": "#95a5a6"}
L = {"shared": "no stop-grad", "stopgrad": "stop-grad only",
     "ema": "stop-grad + EMA (I-JEPA)", "frozen": "frozen target"}

path = sys.argv[1] if len(sys.argv) > 1 else "runs/collapse/curves.json"
res = json.load(open(path))
D = res[list(res)[0]][0]["dim"]

fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
for v, c in res.items():
    s = [p["step"] for p in c]
    ax[0].plot(s, [p["loss"] for p in c], color=C[v], label=L[v], lw=1.4)
    ax[1].plot(s, [p["std_ratio"] for p in c], color=C[v], lw=1.4)
    ax[2].plot(s, [p["eff_rank"] for p in c], color=C[v], lw=1.4)
ax[0].set_yscale("log"); ax[0].set_ylabel("JEPA loss"); ax[0].set_title("(a) the loss lies")
ax[1].axhline(1.0, ls=":", c="k", lw=.8); ax[1].set_yscale("log")
ax[1].set_ylabel(r"std / $(1/\sqrt{D})$"); ax[1].set_title("(b) the std tells the truth")
ax[2].axhline(D, ls=":", c="k", lw=.8); ax[2].set_ylabel(f"effective rank (of {D})")
ax[2].set_title("(c) how many directions survive")
for a in ax:
    a.set_xlabel("step"); a.grid(alpha=.25)
ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("collapse.png", dpi=160)
print("wrote collapse.png")
