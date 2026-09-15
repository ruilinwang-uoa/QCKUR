# -*- coding: utf-8 -*-
"""Rebuild fig3_taskwise_error_heatmap.pdf without the in-figure title.

Cell values are the exact annotations extracted from the previous vector PDF
(verified identical to the numbers quoted in Section 5.2). Only the title is
removed; layout, colormap, and annotations are preserved.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = ["Binary classification", "Multi-class classification", "Regression"]
cols = ["DS-4-Pro\nQCKUR", "DS-4-Pro\nLangChain", "GPT-4\nQCKUR", "GPT-4\nDirect",
        "GPT-4\nLangChain", "GPT-5.5\nQCKUR", "GPT-5.5\nDirect", "GPT-5.5\nLangChain"]
vals = np.array([
    [1.0, 3.5, 1.2, 16.0, 4.0, 0.5, 10.5, 3.7],
    [0.6, 6.0, 0.7, 20.0, 5.0, 0.3, 14.0, 0.8],
    [3.4, 32.2, 3.4, 25.5, 27.1, 1.7, 18.5, 3.4],
])

fig, ax = plt.subplots(figsize=(10.0, 3.6))
im = ax.imshow(vals, cmap="YlOrRd", vmin=0, vmax=32, aspect="auto")

ax.set_xticks(range(len(cols)), cols, fontsize=9, rotation=30, ha="right")
ax.set_yticks(range(len(rows)), rows, fontsize=9)
ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.5)
ax.tick_params(which="minor", length=0)

for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=9,
                fontweight="bold", color="white" if v >= 15 else "black")

cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label("Answer error rate (%)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig.tight_layout()
out = Path(__file__).resolve().parents[2] / "figures" / "fig3_taskwise_error_heatmap.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
