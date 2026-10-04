"""Statistical analysis on the n=100 main results (paper Section 4.3.4).

Uses the per-instance composite Final (paired by instance across systems) to
compute, for B8 vs each baseline:
  * mean paired Final difference
  * Wilcoxon signed-rank p-value
  * Cliff's delta effect size
plus a Kruskal-Wallis omnibus across all eight systems. No new runs required:
all eight systems were evaluated on the SAME 100 instances, so pairing is exact.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import config as C
import composite_final as cf

ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a) * len(b)
    if n == 0:
        return float("nan")
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / n


def instance_final_table():
    """Per-instance Final for every system, aligned on instance_id."""
    inst = None
    for key, col in cf.LAYERS:
        d = cf._layer_df(col).rename(columns={col: key})
        inst = d if inst is None else inst.merge(d, on=["method", "instance_id"], how="outer")
    inst["Final"] = (0.35 * inst["L1"] * 100 + 0.30 * inst["L2"] * 100
                     + 0.35 * inst["L3"] * 20)
    return inst.pivot_table(index="instance_id", columns="method", values="Final")


def main():
    F = instance_final_table()
    F = F[[m for m in ORDER if m in F.columns]]
    print("Per-instance Final matrix:", F.shape, "| systems:", list(F.columns))

    # omnibus
    groups = [F[m].dropna().values for m in F.columns]
    H, p_omni = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis omnibus: H={H:.2f}, p={p_omni:.3e}")

    b8 = F["B8"].dropna()
    rows = []
    for m in F.columns:
        if m == "B8":
            continue
        paired = F[[m, "B8"]].dropna()
        diff = paired["B8"] - paired[m]
        try:
            w, p = stats.wilcoxon(paired["B8"], paired[m], zero_method="wilcox")
        except ValueError:  # all differences zero (deterministic tie)
            w, p = float("nan"), 1.0
        d = cliffs_delta(paired["B8"].values, paired[m].values)
        marker = "\u2021" if p < 0.01 else ("\u2020" if p < 0.05 else "")
        rows.append({"baseline": m, "delta_Final": round(diff.mean(), 2),
                     "wilcoxon_p": p, "cliffs_delta": round(d, 3), "sig": marker,
                     "n_pairs": len(paired)})

    df = pd.DataFrame(rows)
    # save
    out = C.RUNS_DIR / "n100_significance.csv"
    df.to_csv(out, index=False)

    # LaTeX snippet for Section 5
    L = ["% Auto-generated: B8 vs each baseline, paired by instance (n=100)\n",
         "% \\ddagger p<0.01, \\dagger p<0.05 (Wilcoxon signed-rank); Cliff's $\\delta$\n",
         "\\begin{tabular}{lrrrl}\n\\toprule\n",
         "Baseline & $\\Delta$Final & Wilcoxon $p$ & Cliff's $\\delta$ & sig \\\\\n\\midrule\n"]
    for _, r in df.iterrows():
        L.append(f"{r['baseline']} & {r['delta_Final']:+.1f} & {r['wilcoxon_p']:.1e} "
                 f"& {r['cliffs_delta']:+.2f} & {r['sig']} \\\\\n")
    L.append(f"\\midrule\n\\multicolumn{{5}}{{l}}{{Kruskal-Wallis omnibus: $H={H:.1f}$, "
             f"$p={p_omni:.1e}$}} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    (C.ANALYSIS_DIR / "n100_significance.tex").write_text("".join(L), encoding="utf-8")

    print("\n=== B8 vs each baseline (paired, n=100) ===")
    show = df.copy()
    show["sig"] = show["sig"].replace({"‡": "p<.01", "†": "p<.05", "": "n.s."})
    print(show.to_string(index=False))
    print(f"\nSaved {out} and analysis/n100_significance.tex")


if __name__ == "__main__":
    main()
