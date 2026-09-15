"""Publication figures for Section 5 (PDF, wired into main.tex).

  fig_main_results.pdf   - composite Final by system (primary), B8 highlighted
  fig_ablation.pdf       - component-contribution bar chart (B8 -> -T/-Q/-P/-M)
  fig_crossllm.pdf       - primary vs cross-LLM Final (grouped bars)

Restraint over chrome: one accent for the proposed method, muted gray for the
rest; direct labels; recessive axes; single axis per chart. No in-figure
titles (captions in the manuscript carry them); instance-level sd whiskers
where per-instance scores are available.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#2E86AB"   # proposed framework / focal
GOLD = "#EAB23A"     # template-only ceiling
MUTED = "#BFC4C9"    # baselines
INK = "#222831"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#888", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

NAMES = {"B1": "B1 Template-only", "B2": "B2 Trad. D2T", "B3": "B3 Direct LLM",
         "B4": "B4 Few-shot", "B5": "B5 Tool-using agent", "B6": "B6 RAG",
         "B7": "B7 RAG+CoVe", "B8": "B8 Framework"}
# mean and instance-level sd of composite Final (n=100; runs/n100_final_summary.csv)
PRIMARY = {"B1": (98.3, 9.7), "B2": (83.3, 9.5), "B3": (34.5, 10.9),
           "B4": (36.4, 9.9), "B5": (61.3, 25.4), "B6": (39.4, 15.2),
           "B7": (36.5, 11.3), "B8": (83.9, 12.8)}
CROSS = {"B1": 98.2, "B2": 84.0, "B3": 56.9, "B4": 66.6,
         "B5": 55.9, "B6": 61.1, "B7": 60.5, "B8": 85.1}
# third judge (GLM-5.3) re-scoring of the same DeepSeek-generated reports
THIRD = {"B1": 98.4, "B2": 88.2, "B3": 70.8, "B4": 77.8,
         "B5": 67.9, "B6": 71.9, "B7": 70.4, "B8": 86.8}
# descending by Final: B1, B8, B2, B5, B6, B7, B4, B3
ORDER = ["B1", "B8", "B2", "B5", "B6", "B7", "B4", "B3"]


def fig_main():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    vals = [PRIMARY[m][0] for m in ORDER]
    errs = [PRIMARY[m][1] for m in ORDER]
    colors = [ACCENT if m == "B8" else (GOLD if m == "B1" else MUTED) for m in ORDER]
    bars = ax.barh(range(len(ORDER)), vals, color=colors, height=0.68,
                   xerr=errs, error_kw=dict(ecolor=INK, elinewidth=0.9, capsize=2.5),
                   zorder=3)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([NAMES[m] for m in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("Composite Final score (0\N{EN DASH}100), mean $\\pm$ sd over $n{=}100$ instances")
    ax.grid(axis="x", color="#E4E7EB", zorder=0)
    for b, m in zip(bars, ORDER):
        ax.text(b.get_width() + PRIMARY[m][1] + 1.6, b.get_y() + b.get_height() / 2,
                f"{PRIMARY[m][0]:.1f}", va="center", fontsize=8.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT),
               plt.Rectangle((0, 0), 1, 1, color=GOLD),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    ax.legend(handles, ["Proposed framework (B8)", "Template-only ceiling (B1, no LLM)",
                        "Baselines"], frameon=False, fontsize=8,
              loc="lower right", bbox_to_anchor=(1.0, 0.02))
    fig.savefig(OUT / "fig_main_results.pdf")
    plt.close(fig)


def fig_ablation():
    # bar chart of Final after removing each component (mean +/- instance sd)
    labels = ["B8\nfull", "\u2212T\n(templates)", "\u2212Q\n(question\nmatching)",
              "\u2212P\n(physical\nrules)", "\u2212M\n(model\nhandlers)"]
    finals = [83.9, 81.4, 64.0, 62.6, 35.6]
    sds = [12.8, 13.8, 19.3, 18.3, 5.6]   # instance-level sd (n=100)
    drops = [None, -2.5, -19.9, -21.3, -48.3]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    colors = [ACCENT, "#9BB5C9", "#6F94AE", "#466E89", "#C0392B"]
    bars = ax.bar(range(len(labels)), finals, color=colors, width=0.62,
                  yerr=sds, error_kw=dict(ecolor=INK, elinewidth=0.9, capsize=2.5),
                  zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Composite Final (0\N{EN DASH}100), mean $\\pm$ sd")
    ax.grid(axis="y", color="#E4E7EB", zorder=0)
    for b, f, d in zip(bars, finals, drops):
        ax.text(b.get_x() + b.get_width() / 2, f + 3.2, f"{f:.1f}",
                ha="center", fontsize=8.5, color=INK)
        if d is not None:
            ax.text(b.get_x() + b.get_width() / 2, f - 6.0, f"{d:+.1f}",
                    ha="center", fontsize=8.5, color="white", fontweight="bold")
    fig.savefig(OUT / "fig_ablation.pdf")
    plt.close(fig)


def fig_crossllm():
    systems = ["B1", "B8", "B2", "B4", "B6", "B7", "B3", "B5"]
    short = {"B1": "B1\nTemplate", "B2": "B2\nD2T", "B3": "B3\nDirect",
             "B4": "B4\nFew-shot", "B5": "B5\nTool agent", "B6": "B6\nRAG",
             "B7": "B7\nRAG+CoVe", "B8": "B8\nFramework"}
    x = np.arange(len(systems))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    # series colors distinct from the B8 accent used in Figs. 6-9
    ax.bar(x - w, [PRIMARY[m][0] for m in systems], w,
           label="GLM-4-Flash gen., DeepSeek judge (primary)",
           color=MUTED, zorder=3)
    ax.bar(x, [CROSS[m] for m in systems], w,
           label="DeepSeek-V4-Flash gen., DeepSeek judge",
           color="#D55E00", zorder=3)
    ax.bar(x + w, [THIRD[m] for m in systems], w,
           label="DeepSeek-V4-Flash gen., GLM-5.3 judge",
           color="#009E73", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([short[m] for m in systems], fontsize=8)
    ax.set_ylabel("Composite Final (0\N{EN DASH}100)")
    ax.set_ylim(0, 118)
    ax.grid(axis="y", color="#E4E7EB", zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(0.99, 1.02))
    fig.savefig(OUT / "fig_crossllm.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_main()
    fig_ablation()
    fig_crossllm()
    print("Wrote:", *[p.name for p in OUT.glob("fig_*.pdf")])
