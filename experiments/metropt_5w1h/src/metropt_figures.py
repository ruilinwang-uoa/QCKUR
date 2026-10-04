"""
metropt_figures.py -- the three new figures for main_ress.tex.

House style (make_figures.py conventions): ACCENT/GOLD/MUTED palette,
no in-figure titles, tight bbox, 300 dpi, PDF + PNG twins, small fonts.

  fig_metropt_timeline    -- 3 vehicle-epochs, 9 documented events,
                             LPS triggers, ghost windows, data spans
  fig_metropt_risk        -- selective risk-coverage family with the
                             frozen operating point
  fig_metropt_leadtime    -- detection over lead time per system

Usage: python metropt_figures.py
Output: figures/fig_metropt_*.pdf (+ .png twins)
"""
from __future__ import annotations

from pathlib import Path

NL = chr(10)

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
if not RUNS.exists():
    RUNS = ROOT / "runs"
OUT = ROOT / ".." / "figures"
OUT.mkdir(exist_ok=True)

ACCENT = "#2E86AB"   # framework / focal
GOLD = "#EAB23A"     # template ceiling
MUTED = "#BFC4C9"    # baselines
INK = "#1F2933"
RED = "#C0392B"      # oil leak / risk accents

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    print(f"-> {OUT}/{name}.pdf/.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig 1
def timeline():
    import datetime as dt

    st = pd.read_csv(RUNS / "metropt_positive_strata.csv",
                     parse_dates=["t_end"])

    # Each vehicle-epoch uses its own calendar range.  This removes the
    # empty 2021 span while preserving every event/window timestamp and the
    # original visual encoding used in the manuscript figure.
    panels = [
        {
            "title": "2020 epoch (10 s)",
            "cal": "cal. pool: 112 win.",
            "start": dt.datetime(2020, 2, 1),
            "end": dt.datetime(2020, 9, 1),
            "ticks": [dt.datetime(2020, m, 1) for m in (2, 4, 6, 8)],
            "vehicle": "metropt3",
            "events": [
                ("1", dt.datetime(2020, 4, 18, 0, 0),
                 dt.datetime(2020, 4, 18, 23, 59), "air", True, None),
                ("2", dt.datetime(2020, 5, 29, 23, 30),
                 dt.datetime(2020, 5, 30, 6, 0), "air", False, None),
                ("3", dt.datetime(2020, 6, 5, 10, 0),
                 dt.datetime(2020, 6, 7, 14, 30), "air", True, None),
                ("4", dt.datetime(2020, 7, 15, 14, 30),
                 dt.datetime(2020, 7, 15, 19, 0), "air", True, None),
            ],
            "ghosts": [dt.datetime(2020, 3, 29), dt.datetime(2020, 6, 25)],
        },
        {
            "title": "Train A (2022, 5/6 Hz)",
            "cal": "cal. pool: 100 win.",
            "start": dt.datetime(2022, 1, 1),
            "end": dt.datetime(2022, 6, 2),
            "ticks": [dt.datetime(2022, m, 1) for m in (1, 2, 3, 4, 5, 6)],
            "vehicle": "metropt2022",
            "events": [
                ("A-1", dt.datetime(2022, 2, 28, 21, 53),
                 dt.datetime(2022, 3, 1, 2, 0), "air", True,
                 dt.datetime(2022, 2, 28, 22, 50)),
                ("A-2", dt.datetime(2022, 3, 23, 14, 54),
                 dt.datetime(2022, 3, 23, 15, 24), "air", False, None),
                ("A-3", dt.datetime(2022, 5, 30, 12, 0),
                 dt.datetime(2022, 6, 2, 6, 18), "oil", True,
                 dt.datetime(2022, 6, 2, 6, 18)),
            ],
            "ghosts": [dt.datetime(2022, 2, 12)],
        },
        {
            "title": "Train B (2022, 1 Hz)",
            "cal": "cal. pool: 53 win.",
            # Keep the true data span (late April to late July 2022), but
            # show April explicitly on the axis so the panel matches the
            # manuscript description “April–July 2022”.
            "start": dt.datetime(2022, 4, 1),
            "end": dt.datetime(2022, 7, 28),
            "ticks": [dt.datetime(2022, m, 1) for m in (4, 5, 6, 7)],
            "vehicle": "metropt2022B",
            "events": [
                ("B-1", dt.datetime(2022, 6, 4, 10, 19),
                 dt.datetime(2022, 6, 4, 14, 22), "air", True,
                 dt.datetime(2022, 6, 4, 11, 26)),
                ("B-2", dt.datetime(2022, 7, 11, 10, 10),
                 dt.datetime(2022, 7, 14, 10, 22), "oil", True,
                 dt.datetime(2022, 7, 13, 19, 43)),
            ],
            "ghosts": [],
        },
    ]

    strat_style = {
        "signature": (ACCENT, 3.2, 1.0),
        "subthreshold": (GOLD, 2.6, 0.9),
        "presymptomatic": (MUTED, 2.2, 0.75),
    }

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 4.65))

    for ax, panel in zip(axes, panels):
        y = 0.0
        s0, s1 = panel["start"], panel["end"]

        # Vehicle data span.
        ax.hlines(y, s0, s1, color=MUTED, lw=7, alpha=0.55, zorder=2)

        # Positive-window rug, kept above the event/data line and coloured by
        # the manuscript's three ground-truth visibility strata.
        veh_rows = st[st.vehicle == panel["vehicle"]]
        for _, w in veh_rows.iterrows():
            c, lw_, al = strat_style[w.stratum]
            ax.vlines(w.t_end, 0.26, 0.42, color=c, lw=lw_, alpha=al,
                      zorder=4)

        last_lbl_x = None
        for (eid, a, b, mode, removed, lps) in panel["events"]:
            c = ACCENT if mode == "air" else RED
            ax.hlines(0.07, a, b, color=c, lw=2.4, alpha=0.88, zorder=4)

            marker = "v" if removed else "o"
            ax.scatter([a], [0], marker=marker, s=54, color=c, zorder=5,
                       edgecolors=INK, linewidths=0.55)

            # Event identifiers stay close to their markers; only genuinely
            # adjacent events are staggered to avoid text collision.
            ylbl = -0.23
            if last_lbl_x is not None and (a - last_lbl_x).days < 25:
                ylbl = -0.36
            ax.text(a, ylbl, eid, fontsize=8.0, color=INK,
                    ha="center", va="center")
            last_lbl_x = a

            if lps is not None:
                ax.vlines(lps, -0.10, 0.17, color=INK, lw=1.15, zorder=5)

        # Undocumented episodes excluded from calibration.
        for g in panel["ghosts"]:
            ax.scatter([g], [-0.34], marker="x", s=52, color=GOLD,
                       zorder=5, linewidths=1.8)

        # Panel labels use only wording already present in the original figure.
        ax.text(0.0, 1.10, panel["title"], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.8, color=INK,
                fontweight="semibold")
        ax.text(1.0, 1.10, panel["cal"], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.0, color=INK)

        ax.set_xlim(s0, s1)
        ax.set_ylim(-0.49, 0.50)
        ax.set_yticks([])
        ax.set_xticks(panel["ticks"])
        tick_labels = []
        for j, t in enumerate(panel["ticks"]):
            tick_labels.append(t.strftime("%b\n%Y") if j == 0 else t.strftime("%b"))
        ax.set_xticklabels(tick_labels, fontsize=7.8)
        ax.tick_params(axis="x", length=3.5, width=0.8, pad=3)

        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["bottom"].set_color(INK)

    # Each panel now uses its own local calendar axis, so label the x-axis
    # on every panel rather than implying a single shared axis at the bottom.
    for ax in axes:
        ax.set_xlabel("Calendar time", fontsize=8.1, labelpad=4)

    handles = [
        plt.Line2D([], [], color=MUTED, lw=3, alpha=0.55, label="data span"),
        plt.Line2D([], [], color=ACCENT, lw=3, label="air leak (bar = span)"),
        plt.Line2D([], [], color=RED, lw=3, label="oil leak (bar = span)"),
        plt.Line2D([], [], marker="v", ls="", color=INK,
                   label="removed / repaired"),
        plt.Line2D([], [], marker="o", ls="", color=INK,
                   label="continued / self-resolved"),
        plt.Line2D([], [], color=INK, lw=1.6, label="LPS trigger (verified)"),
        plt.Line2D([], [], marker="x", ls="", color=GOLD,
                   label="undocumented episode (excl.)"),
        plt.Line2D([], [], color=ACCENT, lw=2.6,
                   label="positive window: signature"),
        plt.Line2D([], [], color=GOLD, lw=2.6,
                   label="positive window: sub-threshold"),
        plt.Line2D([], [], color=MUTED, lw=2.6,
                   label="positive window: pre-symptomatic"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=7.0,
               loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5,
               columnspacing=1.0, handletextpad=0.45, labelspacing=0.8)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.23,
                        hspace=0.76)
    save(fig, "fig_metropt_timeline_v2")


# ------------------------------------------------------------------ fig 2
def risk_coverage():
    H = pd.read_csv(RUNS / "metropt_risk_coverage.csv")
    R = pd.read_csv(RUNS / "metropt_risk_coverage_reports.csv")

    # Preserve the original four risk series and frozen operating point; the
    # changes below are publication-layout changes only (font size, spacing,
    # line/marker legibility, and annotation placement).
    fig, ax = plt.subplots(figsize=(6.35, 3.75))
    series = [
        (H.coverage * 100, H.risk * 100, INK, "-", "o",
         "Handler verdict errors"),
        (R.coverage * 100, R.B8_risk_full * 100, ACCENT, "-", "o",
         "B8 composite factual errors"),
        (R.coverage * 100, R.B8_risk_num * 100, "#9BB5C9", "--", "s",
         "B8 numeric errors"),
        (R.coverage * 100, R.B8_action_risk * 100, RED, ":", "D",
         "B8 action mismatches"),
    ]
    for x, y, c, ls, mk, lab in series:
        ax.plot(x, y, color=c, ls=ls, lw=2.0, marker=mk, ms=4.4,
                markeredgewidth=0.45, markeredgecolor=INK,
                label=lab, zorder=3)

    op = R.loc[R.threshold == 6.0].iloc[0]
    ox, oy = op.coverage * 100, op.B8_risk_full * 100
    # Display from the error count (4/104 = 3.846 -> 3.8), not the rounded CSV.
    oy_disp = int(round(op.B8_risk_full * op.B8_n)) / op.B8_n * 100
    ax.scatter([ox], [oy], s=82, facecolors="none", edgecolors=ACCENT,
               linewidths=1.7, zorder=6)
    ax.annotate(f"operating point $t{{=}}6$\n({ox:.0f}% cov., {oy_disp:.1f}%)",
                xy=(ox, oy), xytext=(82.5, 11.2), fontsize=8.2,
                color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.0,
                                shrinkA=2, shrinkB=4))

    ax.set_xlabel("Coverage (% windows accepted)", fontsize=9.2)
    ax.set_ylabel("Risk on accepted windows (%)", fontsize=9.2)
    ax.set_xlim(30, 102)
    ax.set_ylim(-1.5, 32)
    ax.tick_params(axis="both", labelsize=8.3, width=0.8, length=3.5)
    ax.grid(axis="y", color=MUTED, alpha=0.28, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8.1, loc="upper left",
              handlelength=2.5, borderaxespad=0.3, labelspacing=0.55)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.97, bottom=0.18)
    save(fig, "fig_metropt_risk")


# ------------------------------------------------------------------ fig 3
def leadtime():
    cv = pd.read_csv(RUNS / "metropt_leadtime_curve.csv")
    dm = pd.read_csv(RUNS / "metropt_detection_matrix.csv", index_col=0)

    # Keep the original two-panel evidence structure. The revision enlarges
    # the final-size text, reduces lead-time jitter, and makes the matrix
    # labels readable without changing any detection values.
    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(7.55, 3.35),
        gridspec_kw={"wspace": 0.30, "width_ratios": [1.04, 1.10]})

    piv = cv.pivot(index="offset_h", columns="system", values="rate")
    # Small symmetric offsets reveal coincident series while keeping markers
    # visually tied to the actual 6/12/24/48-h lead-time positions.
    style = {
        "B1": (GOLD, "-", "o", -0.55),
        "B2": ("#D9C08A", "-", "s", 0.55),
        "B8": (ACCENT, "-", "o", 0.0),
        "B3R": ("#7A8B99", "--", "x", 0.85),
        "B3": ("#9CA3AB", "--", "^", -0.40),
        "B5": ("#6B7280", ":", "v", 0.40),
    }
    for s_, (c, ls, mk, dx) in style.items():
        if s_ in piv.columns:
            ax.plot(piv.index + dx, piv[s_], color=c, ls=ls, lw=2.0,
                    marker=mk, ms=4.8, label=s_, zorder=3,
                    markeredgecolor=INK, markeredgewidth=0.35)

    ax.annotate("B8: one verdict softened" + NL + "in verbalization",
                xy=(24, 0.5), xytext=(14.2, 0.63), fontsize=7.8,
                color=ACCENT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.9,
                                shrinkA=2, shrinkB=3))
    ax.set_xlabel("Lead time before failure end (h)", fontsize=8.9)
    ax.set_ylabel("Detection rate (signature windows)", fontsize=8.9)
    ax.set_xticks([6, 12, 24, 48])
    ax.set_xlim(3, 54)
    ax.set_ylim(-0.08, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax.tick_params(axis="both", labelsize=8.0, width=0.8, length=3.5)
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.7)

    sec = ax.secondary_xaxis("top")
    sec.set_xticks([6, 12, 24, 48])
    n_per = {"6": "n=2", "12": "n=3", "24": "n=2", "48": "n=1"}
    sec.set_xticklabels([n_per[str(t)] for t in [6, 12, 24, 48]],
                        fontsize=7.5, color=INK)
    sec.tick_params(length=0, pad=2.0)
    sec.spines["top"].set_visible(False)

    ax.legend(frameon=False, fontsize=7.9, loc="center right", ncol=1,
              title=None, labelspacing=0.65, handlelength=2.0,
              borderaxespad=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_title("(a) detection vs. lead time", fontsize=8.7, loc="left",
                 pad=15)

    cols = list(dm.columns)
    # Same identifiers as the original figure; line breaks replace rotation.
    col_labels = ["20-1\n6 h", "20-1\n12 h", "20-2\n48 h",
                  "20-3\n6 h", "20-3\n12 h", "20-3\n24 h",
                  "OOC\nctl", "B-2\n12 h", "B-2\n24 h"]
    M = dm.values
    axm.imshow(M, cmap=matplotlib.colors.ListedColormap(
        ["#EDF0F2", ACCENT]), aspect="auto", vmin=0, vmax=1)
    axm.set_xticks(range(len(cols)))
    axm.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=7.0)
    axm.set_yticks(range(len(dm.index)))
    axm.set_yticklabels(dm.index, fontsize=8.0)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axm.text(j, i, str(int(M[i, j])), ha="center", va="center",
                     fontsize=7.2,
                     color="#FFFFFF" if M[i, j] else "#7A848D")
        axm.text(len(cols) + 0.12, i, f"{int(M[i].sum())}/9",
                 fontsize=7.6, va="center", ha="left", color=INK)

    axm.set_xlim(-0.5, len(cols) + 0.72)
    axm.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    axm.set_yticks([y - 0.5 for y in range(1, len(dm.index))], minor=True)
    axm.grid(which="minor", color="white", linewidth=1.15)
    axm.tick_params(which="both", length=0)
    axm.tick_params(axis="x", pad=4)
    for s in axm.spines.values():
        s.set_visible(False)
    axm.set_title("(b) per-window detection matrix", fontsize=8.7,
                  loc="left", pad=8)

    fig.subplots_adjust(left=0.085, right=0.975, top=0.84, bottom=0.24)
    save(fig, "fig_metropt_leadtime")


if __name__ == "__main__":
    timeline()
    risk_coverage()
    leadtime()
