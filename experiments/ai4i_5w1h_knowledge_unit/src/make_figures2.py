# -*- coding: utf-8 -*-
"""Publication figures for Section 5, part 2 (no in-figure titles).

  fig_degradation.pdf - synthetic-anchor dose-response (B1/B8, D0-D3, mean +- sd)
  fig_radar.pdf       - Layer-3 rubric profiles (B1/B2/B5/B8, 5 dims)
  fig_pareto.pdf      - quality-cost trade-off (tokens vs Final)

House style matches make_figures.py (accent/gold/muted, no titles, captions
in the manuscript). Radar keeps the tab10 colors so the caption's
"B8 (red) ... B1 (blue)" stays valid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C  # noqa: E402

OUT = Path(__file__).resolve().parent.parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#2E86AB"
GOLD = "#EAB23A"
MUTED = "#BFC4C9"
INK = "#222831"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#888", "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

REPO_RUNS = C.ROOT.parent / "repo" / "experiments" / "ai4i_5w1h_knowledge_unit" / "runs"


def fig_degradation():
    l1 = pd.read_parquet(REPO_RUNS / "reports_degraded_l1.parquet")[["method", "instance_id", "L1"]]
    l2 = pd.read_parquet(REPO_RUNS / "reports_degraded_l2.parquet")[["method", "instance_id", "L2"]]
    l3 = pd.read_parquet(REPO_RUNS / "reports_degraded_l3.parquet")[["method", "instance_id", "L3"]]
    d = l1.merge(l2, on=["method", "instance_id"]).merge(l3, on=["method", "instance_id"])
    d["Final"] = 0.35 * 100 * d.L1 + 0.30 * 100 * d.L2 + 0.35 * 20 * d.L3
    d["sys"] = d["method"].str.split("_").str[0]
    d["level"] = d["method"].str.split("_").str[1]

    levels = ["D0", "D1", "D2", "D3"]
    stats = {s: d[d.sys == s].groupby("level")["Final"].agg(["mean", "std"]).reindex(levels)
             for s in ("B1", "B8")}
    x = np.arange(len(levels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for k, (s, color) in enumerate((("B1", GOLD), ("B8", ACCENT))):
        m = stats[s]["mean"].values
        sd = stats[s]["std"].values
        lo = np.maximum(m - sd, 0)          # error bars capped at the scale
        hi = np.minimum(m + sd, 100)        # bounds 0..100 (caption note)
        ax.bar(x + (k - 0.5) * w, m, w, color=color, zorder=3,
               label="B1 Template-only (no LLM)" if s == "B1" else "B8 Proposed framework")
        ax.errorbar(x + (k - 0.5) * w, m, yerr=[m - lo, hi - m], fmt="none",
                    ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=4)
        for xi, mi in zip(x + (k - 0.5) * w, m):
            ax.text(xi, mi + 1.2, f"{mi:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["D0\noriginal", "D1\n2 sections\ndeleted",
                        "D2\nnumbers\n$\times$1.3", "D3\nfault mode\ncorrupted"],
                       fontsize=8)
    ax.set_ylabel("Composite Final score (0\N{EN DASH}100)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#E4E7EB", zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(OUT / "fig_degradation.pdf")
    plt.close(fig)


def fig_radar():
    SRC = {"B1": "reports_main_glm_n100_B1B2B8_v3", "B2": "reports_main_glm_n100_B1B2B8_v3",
           "B5": "reports_main_glm_n100", "B8": "reports_main_glm_n100_B1B2B8_v3"}
    dims = ["faithfulness", "completeness", "coverage", "coherence", "actionability"]
    labels = ["Faithfulness", "Completeness", "5W1H coverage\n($s_{cov,5}$)", "Coherence", "Actionability"]
    prof = {}
    for m, stem in SRC.items():
        d = pd.read_parquet(C.RUNS_DIR / f"{stem}_l3.parquet")
        d = d[d.method == m]
        cov5 = d["coverage"].apply(
            lambda c: sum(1 for v in (c or {}).values() if v) / 6 * 5
            if isinstance(c, dict) else np.nan)
        prof[m] = [d["faithfulness"].mean(), d["completeness"].mean(), cov5.mean(),
                   d["coherence"].mean(), d["actionability"].mean()]

    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(5.4, 4.3), subplot_kw=dict(polar=True))
    colors = {"B1": "#1f77b4", "B2": "#ff7f0e", "B5": "#2ca02c", "B8": "#d62728"}
    names = {"B1": "B1 Template-only", "B2": "B2 Trad. D2T",
             "B5": "B5 Tool-using agent", "B8": "B8 Proposed framework"}
    for m in ("B2", "B5", "B1", "B8"):       # B8 last = on top
        v = prof[m] + prof[m][:1]
        ax.plot(ang, v, color=colors[m], linewidth=1.8 if m == "B8" else 1.3,
                label=names[m], zorder=3 if m == "B8" else 2)
        ax.fill(ang, v, color=colors[m], alpha=0.10 if m != "B8" else 0.16, zorder=1)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.tick_params(pad=12)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7, color="#888")
    ax.grid(color="#D8DBDE", linewidth=0.7)
    ax.spines["polar"].set_color("#AAA")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(-0.42, -0.10))
    fig.savefig(OUT / "fig_radar.pdf")
    plt.close(fig)


def fig_pareto():
    # tokens = prompt + completion per report (Table: efficiency); Final n=100
    pts = {  # system: (tokens, Final, is_framework, is_template)
        "B1": (0, 98.3, False, True),
        "B2": (0, 83.3, False, False),
        "B3 Direct LLM": (699, 34.5, False, False),
        "B8 Framework": (813, 83.9, True, False),
        "B6 RAG": (952, 39.4, False, False),
        "B4 Few-shot": (1301, 36.4, False, False),
        "B5 Tool agent": (1771, 61.3, False, False),
        "B7 RAG+CoVe": (2955, 36.5, False, False),
    }
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for name, (tok, fin, isfw, istpl) in pts.items():
        color = ACCENT if isfw else (GOLD if istpl else MUTED)
        marker = "D" if (isfw or istpl) else "o"
        size = 70 if (isfw or istpl) else 46
        x = 25 if tok == 0 else tok        # offset the no-LLM systems off the axis
        ax.scatter(x, fin, s=size, color=color, marker=marker, zorder=3,
                   edgecolor=INK, linewidth=0.5)
        dx, dy, ha = (12, -1.0, "left")
        if name == "B2":
            dx, dy, ha = (12, 2.0, "left")
        if name == "B6 RAG":
            dy = 2.4
        if name == "B4 Few-shot":
            dy = -3.6
        if name == "B7 RAG+CoVe":
            dy = 2.4
        if name == "B5 Tool agent":
            dy = 2.4
        ax.annotate(name, (x, fin), xytext=(x + dx, fin + dy), fontsize=8,
                    color=INK, ha=ha)
    # Pareto frontier among LLM-based systems: B3 (fewest tokens), B8 (highest Final)
    ax.plot([699, 813], [34.5, 83.9], color=ACCENT, linewidth=1.0,
            linestyle=":", zorder=2)
    ax.set_xlabel("Tokens per report (prompt + completion)")
    ax.set_ylabel("Composite Final score (0\N{EN DASH}100)")
    ax.set_xlim(-80, 3250)
    ax.set_ylim(20, 105)
    ax.grid(color="#E4E7EB", zorder=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="D", ls="", color=ACCENT, label="B8 Proposed framework"),
               Line2D([], [], marker="D", ls="", color=GOLD, label="B1 Template-only (no LLM)"),
               Line2D([], [], marker="o", ls="", color=MUTED, label="LLM-based baselines")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    fig.savefig(OUT / "fig_pareto.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_degradation()
    fig_radar()
    fig_pareto()
    print("saved fig_degradation/fig_radar/fig_pareto ->", OUT)
