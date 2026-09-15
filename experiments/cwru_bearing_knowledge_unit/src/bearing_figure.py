"""Composite figure for the bearing instantiation (Phase 3, fig_bearing.pdf).

Panel (a) mechanism: bandpass-envelope amplitude spectra for one rule-tier
outer-race record (130.mat) and one normal record (97.mat), with the first
five BPFO harmonic positions marked -- visualizes why the rule tier fires.

Panel (b) outcomes: per-instance composite Final for the six systems
(strip + mean marker), house palette (B8=ACCENT, B1=GOLD ceiling,
baselines=MUTED), descending mean order as in the AI4I figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, filtfilt, hilbert

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bearing_v1 import FS, band_edges  # noqa: E402  (same scan grid)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bearing_cwru"
RUNS = ROOT / "runs"
OUT = ROOT.parent / "figures"
OUT.mkdir(exist_ok=True)

ACCENT, GOLD, MUTED, INK = "#2E86AB", "#EAB23A", "#BFC4C9", "#222831"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#888", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
SPEC = json.load(open(ROOT / "knowledge" / "maps" / "physics_rules_spec_bearing.json",
                      encoding="utf-8"))
M_MULT = SPEC["defect_freq_multiples_x_fr"]


def band_env_spectrum(num: str, meta: dict) -> tuple[np.ndarray, np.ndarray, float]:
    m = scipy.io.loadmat(DATA / f"{num}.mat")
    x = m.get(f"X{num}_DE_time")
    if x is None:
        x = m.get(f"X{int(num):03d}_DE_time")
    x = x.ravel().astype(float)
    rpm = float(m[f"X{num}RPM"].ravel()[0]) if f"X{num}RPM" in m else 1797.0
    nyq = FS / 2
    n = len(x)
    win = np.hanning(n)
    best = (-np.inf, None)
    for lo, hi in band_edges():
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        xb = filtfilt(b, a, x)
        k = float(pd.Series(xb).kurt())
        if k > best[0]:
            best = (k, xb)
    xb = best[1]
    env = np.abs(hilbert(xb))
    env = env - env.mean()
    amp = np.abs(np.fft.rfft(env * win)) * 2 / np.sum(win)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    return freqs, amp, rpm / 60.0


def per_instance_finals() -> pd.DataFrame:
    l1 = pd.read_csv(RUNS / "bearing_l1.csv")
    l1["L1"] = (0.30 * l1.numeric.fillna(0) + 0.30 * l1.rule.fillna(0)
                + 0.40 * l1.classification.fillna(0))
    l2 = pd.read_parquet(RUNS / "bearing_l2.parquet")[["system", "instance_id", "L2"]]
    l3 = pd.read_parquet(RUNS / "bearing_l3.parquet")[["system", "instance_id", "L3"]]
    m = (l1.merge(l2, on=["system", "instance_id"])
           .merge(l3, on=["system", "instance_id"]))
    m["Final"] = 0.35 * 100 * m.L1 + 0.30 * 100 * m.L2 + 0.35 * 20 * m.L3
    return m


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                   gridspec_kw=dict(wspace=0.26,
                                                    width_ratios=[1, 1.12]))

    # ---- (a) mechanism: envelope spectra -------------------------------
    meta = {"load_hp": 0}
    f1, a1, fr = band_env_spectrum("130", meta)   # OR rule-tier, L0
    f2, a2, _ = band_env_spectrum("97", meta)     # normal, L0
    sel1 = f1 <= 620
    ax1.plot(f1[sel1], a1[sel1] / a1[sel1].max(), color=ACCENT, lw=1.2,
             label="outer-race fault (130.mat)", zorder=3)
    ax1.plot(f2[f2 <= 620], a2[f2 <= 620] / a2[:len(f2)][f2 <= 620].max(),
             color=MUTED, lw=1.0, label="normal (97.mat)", zorder=2)
    bpfo = M_MULT["BPFO"] * fr
    for k in range(1, 6):
        ax1.axvline(k * bpfo, color=INK, lw=0.7, ls=":", alpha=0.55, zorder=1)
    ax1.annotate("$f_{\\mathrm{BPFO}}$", xy=(bpfo + 6, 0.95), fontsize=8,
                 color=INK, ha="left")
    ax1.annotate("$5f_{\\mathrm{BPFO}}$", xy=(5 * bpfo + 6, 0.95), fontsize=8,
                 color=INK, ha="left")
    ax1.set_xlim(0, 620)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Envelope frequency (Hz)")
    ax1.set_ylabel("Normalized amplitude")
    ax1.text(-0.14, 1.06, "(a)", transform=ax1.transAxes, fontsize=11,
            fontweight="bold", color=INK)
    ax1.grid(color="#E4E7EB", lw=0.6, zorder=0)
    ax1.legend(fontsize=8, frameon=False, loc="center right",
               handlelength=1.6, borderaxespad=0.4)

    # ---- (b) outcomes: per-instance Finals ------------------------------
    m = per_instance_finals()
    means = m.groupby("system").Final.mean().sort_values(ascending=False)
    order = list(means.index)
    names = {"B1": "B1\nTemplate", "B2": "B2\nD2T", "B3": "B3\nDirect",
             "B4": "B4\nFew-shot", "B5": "B5\nAgent", "B8": "B8\nFramework"}
    rng = np.random.default_rng(42)
    for i, sys_id in enumerate(order):
        vals = m[m.system == sys_id].Final.values
        color = ACCENT if sys_id == "B8" else (GOLD if sys_id == "B1" else MUTED)
        x = i + rng.uniform(-0.13, 0.13, len(vals))
        ax2.scatter(x, vals, s=14, color=color, alpha=0.75, lw=0, zorder=3)
        ax2.scatter([i], [vals.mean()], marker="D", s=34, color=INK, zorder=4)
        ax2.annotate(f"{vals.mean():.1f}", xy=(i, vals.mean()),
                     xytext=(13, -3), textcoords="offset points",
                     fontsize=8.5, color=INK)
    ax2.axhline(100, color=GOLD, lw=0.8, ls="--", alpha=0.8, zorder=1)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([names[s] for s in order], fontsize=9)
    ax2.tick_params(axis="x", pad=3)
    ax2.set_ylim(0, 106)
    ax2.set_ylabel("Composite Final")
    ax2.text(-0.14, 1.06, "(b)", transform=ax2.transAxes, fontsize=11,
            fontweight="bold", color=INK)
    ax2.grid(axis="y", color="#E4E7EB", lw=0.6, zorder=0)

    fig.savefig(OUT / "fig_bearing.pdf")
    print("saved", OUT / "fig_bearing.pdf")

    # ---- programmatic collision check (legend vs lines/annotations) ------
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    leg = ax1.get_legend().get_window_extent(renderer=r)
    problems = []
    for ln in ax1.get_lines():
        xy = ln.get_xydata()
        pts = ax1.transData.transform(xy)
        inside = ((pts[:, 0] >= leg.x0) & (pts[:, 0] <= leg.x1)
                  & (pts[:, 1] >= leg.y0) & (pts[:, 1] <= leg.y1))
        if inside.any():
            problems.append(ln.get_label())
    for t in ax1.texts:
        if t.get_window_extent(renderer=r).overlaps(leg):
            problems.append(t.get_text())
    print("panel-a legend collisions:", problems or "NONE")


if __name__ == "__main__":
    main()
