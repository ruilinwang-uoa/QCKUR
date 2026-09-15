"""Design-matched omnibus test for the eight-system AI4I comparison.

The experiment plan pre-registered a Kruskal-Wallis omnibus, which assumes
independent samples; the eight systems are scored on the same n=100
instances (complete-block design), so the Friedman test is the
design-matched omnibus.  This script recomputes per-instance Final exactly
as make_figures2.final_per_inst does and reports Friedman chi-squared,
Kendall's W, and a LaTeX-ready sentence for Section 5.3.

Run from the machine holding the full runs/ parquets:
    python src/friedman_omnibus.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

MAIN = "reports_main_glm_n100"
SRC = {s: MAIN for s in ["B3", "B4", "B5", "B6"]}
SRC["B7"] = "reports_main_glm_n100_B7_v2"
for s in ["B1", "B2", "B8"]:
    SRC[s] = "reports_main_glm_n100_B1B2B8_v3"
ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]


def final_per_inst(stem: str) -> pd.DataFrame:
    l1 = pd.read_parquet(RUNS / f"{stem}_l1.parquet")
    l2 = pd.read_parquet(RUNS / f"{stem}_l2.parquet")[["method", "instance_id", "L2"]]
    l3 = pd.read_parquet(RUNS / f"{stem}_l3.parquet")[["method", "instance_id", "L3"]]
    d = l1.merge(l2, on=["method", "instance_id"]).merge(l3, on=["method", "instance_id"])
    d["Final"] = 0.35 * 100 * d["L1"] + 0.30 * 100 * d["L2"] + 0.35 * 20 * d["L3"]
    return d


def main() -> None:
    frames = [final_per_inst(SRC[s]) for s in ORDER]
    sel = pd.concat(
        [f[f["method"] == s] for s, f in zip(ORDER, frames)], ignore_index=True
    )
    wide = (
        sel.pivot(index="instance_id", columns="method", values="Final")[ORDER]
        .dropna()
    )
    n, k = wide.shape
    stat, p = friedmanchisquare(*[wide[s] for s in ORDER])
    W = stat / (n * (k - 1))
    print(f"blocks n={n}, systems k={k}")
    print(f"Friedman chi2({k - 1}) = {stat:.1f}, p = {p:.2e}, Kendall's W = {W:.2f}")
    print()
    print("LaTeX sentence for Section 5.3:")
    print(
        f"A Friedman omnibus on the complete-block design ($n={n}$ instances $\\times$ "
        f"$k={k}$ systems) likewise rejects the null of equal systems "
        f"($\\chi^2({k - 1}) = {stat:.1f}$, $p {'< 0.001' if p < 0.001 else f'= {p:.3g}'}$, "
        f"Kendall's $W = {W:.2f}$); the paired Wilcoxon tests above are the "
        f"authoritative comparisons."
    )


if __name__ == "__main__":
    main()
