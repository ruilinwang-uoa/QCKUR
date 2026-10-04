"""Analyze the synthetic-anchor degradation set (protocol validity, no humans).

Expectations if the protocol measures its constructs:
  D1 (delete 2 sections) : judge coverage /6 drops by ~2; L3 falls modestly
  D2 (numbers x1.3)      : numeric tolerance collapses; L3 faithfulness falls
  D3 (wrong fault mode)  : classification collapses; L2 falls sharply
Overall: monotone dose-response of Final across D0 > D1 > D2 > D3.

Reports: per-level layer means, Spearman(level, score) per layer per system,
paired Wilcoxon Dk-vs-D0, and the judge-coverage-vs-deleted-sections check.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
STEM = "reports_degraded"
SYSTEMS = ["B1", "B8"]
LEVELS = ["D0", "D1", "D2", "D3"]


def main() -> None:
    l1 = pd.read_parquet(RUNS / f"{STEM}_l1.parquet")
    l2 = pd.read_parquet(RUNS / f"{STEM}_l2.parquet")[["method", "instance_id", "L2"]]
    l3 = pd.read_parquet(RUNS / f"{STEM}_l3.parquet")
    l3["cov6"] = l3["coverage"].apply(lambda d: sum(d.values()) if isinstance(d, dict) else 0)
    l3 = l3[["method", "instance_id", "L3", "faithfulness", "cov6"]]
    d = (l1.merge(l2, on=["method", "instance_id"])
           .merge(l3, on=["method", "instance_id"]))
    d["sys"] = d["method"].str.split("_").str[0]
    d["level"] = d["method"].str.split("_").str[1]
    d["lvl"] = d["level"].map({"D0": 0, "D1": 1, "D2": 2, "D3": 3})
    d["Final"] = 0.35 * 100 * d["L1"] + 0.30 * 100 * d["L2"] + 0.35 * 20 * d["L3"]

    md = ["# Synthetic-anchor degradation results (protocol validity)", ""]
    cols = ["L1", "classification", "numerical", "L2", "L3", "faithfulness",
            "cov6", "Final"]
    for s in SYSTEMS:
        md += [f"## {s}: layer means by degradation level (n=30)", ""]
        t = (d[d["sys"] == s].groupby("level")[cols].mean()
             .reindex(LEVELS).round(3))
        md += ["```", t.to_string(), "```", ""]
        md += [f"Spearman(level, score) per layer, {s}:", "```"]
        rows = []
        for c in cols:
            rho, p = stats.spearmanr(d[d["sys"] == s]["lvl"],
                                     d[d["sys"] == s][c])
            rows.append({"layer": c, "rho": round(rho, 3), "p": f"{p:.1e}"})
        md += [pd.DataFrame(rows).to_string(index=False), "```", ""]
        # paired tests vs D0
        base = d[(d["sys"] == s) & (d["level"] == "D0")].set_index("instance_id")
        md += ["Paired Wilcoxon Dk vs D0 (Final):"]
        for lv in ["D1", "D2", "D3"]:
            dk = d[(d["sys"] == s) & (d["level"] == lv)].set_index("instance_id")
            j = base[["Final"]].join(dk[["Final"]], lsuffix="_0", rsuffix="_k").dropna()
            try:
                w = stats.wilcoxon(j["Final_0"], j["Final_k"])
                md.append(f"  {lv}: drop {j['Final_0'].mean()-j['Final_k'].mean():+.1f} "
                          f"(p={w.pvalue:.1e})")
            except ValueError:
                md.append(f"  {lv}: no variance")
        md.append("")

    # judge-coverage anchor: D1 deleted two sections -> cov6 should fall ~2
    md += ["## Judge-coverage anchor (D1 deleted two 5W1H sections)", ""]
    for s in SYSTEMS:
        c0 = d[(d["sys"] == s) & (d["level"] == "D0")]["cov6"].mean()
        c1 = d[(d["sys"] == s) & (d["level"] == "D1")]["cov6"].mean()
        md.append(f"{s}: judge coverage {c0:.2f}/6 (D0) -> {c1:.2f}/6 (D1) "
                  f"(expected drop ~2)")
    md.append("")

    out = ROOT / "analysis" / "degradation_summary.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
