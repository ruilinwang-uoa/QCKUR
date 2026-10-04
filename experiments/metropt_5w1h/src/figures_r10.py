"""
figures_r10.py -- publication figure utilities for the RESS manuscript.

  fig7_cost_transfer   : current generator for the main-text cost/transfer figure.

House style = make_figures.py / metropt_figures.py conventions
(ACCENT #2E86AB framework, GOLD #EAB23A template/knowledge, MUTED
#BFC4C9 baselines, INK #1F2933 text, font.size 9, vector PDF + PNG).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT.parent / "figures"
RUNS = ROOT / "runs"

ACCENT = "#2E86AB"
GOLD = "#EAB23A"
MUTED = "#BFC4C9"
INK = "#1F2933"
GOLD_BG = "#FDF3DC"
ACCENT_BG = "#E3EEF5"
MUTED_BG = "#F1F2F3"

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=220)
    print(f"-> {FIGS / name}.pdf/.png")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec, fs=8.5, weight="normal", tc=INK,
        rounding=0.02):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={rounding}",
        linewidth=1.0, facecolor=fc, edgecolor=ec, mutation_aspect=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.1, style="-|>", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, shrinkA=1.5, shrinkB=1.5))

# ------------------------------------------------------------------ Fig 7
def fig7_cost_transfer():
    """RSQ3 figure: instantiation cost and per-vehicle transfer.

    Panel (a) preserves the manuscript's logged/reasoned-estimate distinction.
    Panel (b) keeps the per-system range bands used in the caption, while
    pairing B1/B8 within each vehicle instead of implying a temporal trajectory
    across the three categorical vehicle-epochs.
    """
    # Use publication-friendly embedded TrueType fonts for this figure without
    # changing the drawing logic of Fig. 1 or Fig. 2.
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, (a1, a2) = plt.subplots(
            1, 2, figsize=(7.4, 3.0),
            gridspec_kw={"wspace": 0.34, "width_ratios": [0.95, 1.05]},
        )

        # (a) knowledge-engineering cost of the three instantiations
        labels = ["AI4I\n(controlled)", "CWRU\n(public rig)",
                  "MetroPT\n(in-service)"]
        hours = [34, 16, 19]
        estimated = [True, True, False]
        colors = [MUTED, MUTED, ACCENT]

        bars = a1.bar(
            labels, hours, color=colors, width=0.56,
            edgecolor=INK, linewidth=0.65, zorder=2,
        )
        for bar, is_estimate in zip(bars, estimated):
            if is_estimate:
                bar.set_hatch("///")
                bar.set_edgecolor(INK)

        for i, (h, is_estimate) in enumerate(zip(hours, estimated)):
            a1.text(
                i, h + 0.8, f"{h} h" + ("*" if is_estimate else ""),
                ha="center", va="bottom", fontsize=8.2,
            )

        a1.set_ylabel("knowledge-engineering cost (person-hours)", fontsize=8.5)
        a1.set_ylim(0, 40)
        a1.set_yticks(np.arange(0, 41, 10))
        a1.tick_params(axis="both", labelsize=8)
        a1.set_title("(a) instantiation cost", fontsize=8.8, loc="left", pad=6)
        a1.text(
            0.99, 0.97, "* reasoned estimate", transform=a1.transAxes,
            ha="right", va="top", fontsize=7.0, color=INK,
        )
        a1.spines[["top", "right"]].set_visible(False)
        a1.set_axisbelow(True)
        a1.grid(axis="y", linewidth=0.45, alpha=0.22)

        # (b) per-vehicle transfer: scores are categorical by vehicle-epoch.
        pv = pd.read_csv(RUNS / "metropt_per_vehicle_final.csv", index_col=0)
        order = ["metropt3", "metropt2022", "metropt2022B"]
        pv = pv.loc[[v for v in order if v in pv.index]]
        vehs = list(pv.index)
        vehicle_labels = {
            "metropt3": "2020 unit",
            "metropt2022": "train A",
            "metropt2022B": "train B",
        }
        x = np.arange(len(vehs), dtype=float)
        b1 = pv["B1"].to_numpy(dtype=float)
        b8 = pv["B8"].to_numpy(dtype=float)

        # Caption-consistent per-system score bands across vehicle-epochs.
        a2.axhspan(b1.min(), b1.max(), color=GOLD, alpha=0.12, zorder=0)
        a2.axhspan(b8.min(), b8.max(), color=ACCENT, alpha=0.12, zorder=0)

        # Pair B1 and B8 within each vehicle.  Do not connect successive
        # vehicles: they are categorical epochs, not a temporal trajectory.
        dx = 0.12
        for i in range(len(x)):
            a2.plot(
                [x[i] - dx, x[i] + dx], [b1[i], b8[i]],
                color=MUTED, lw=0.9, zorder=1,
            )

        a2.scatter(
            x - dx, b1, s=34, color=GOLD, edgecolor=INK,
            linewidth=0.45, zorder=3, label="B1",
        )
        a2.scatter(
            x + dx, b8, s=34, color=ACCENT, edgecolor=INK,
            linewidth=0.45, zorder=3, label="B8",
        )

        # Direct labels make the small between-vehicle ranges readable without
        # relying only on the truncated display range.
        for xi, y1, y8 in zip(x, b1, b8):
            a2.text(
                xi - dx, y1 + 0.28, f"{y1:.1f}",
                ha="center", va="bottom", fontsize=7.4, color=INK,
            )
            a2.text(
                xi + dx, y8 - 0.32, f"{y8:.1f}",
                ha="center", va="top", fontsize=7.4, color=INK,
            )

        a2.set_xticks(x)
        a2.set_xticklabels([vehicle_labels[v] for v in vehs], fontsize=8)
        a2.set_ylabel("Final score (0–100)", fontsize=8.5)
        a2.set_ylim(88, 100)
        a2.set_yticks(np.arange(88, 101, 2))
        a2.tick_params(axis="y", labelsize=8)
        a2.legend(frameon=False, fontsize=8, loc="lower right", ncol=2,
                  handletextpad=0.4, columnspacing=0.8)
        a2.set_title("(b) per-vehicle transfer (MetroPT)", fontsize=8.8,
                     loc="left", pad=6)
        a2.spines[["top", "right"]].set_visible(False)
        a2.set_axisbelow(True)
        a2.grid(axis="y", linewidth=0.45, alpha=0.22)

        fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.17)
        save(fig, "fig7_cost_transfer")


if __name__ == "__main__":
    fig7_cost_transfer()
