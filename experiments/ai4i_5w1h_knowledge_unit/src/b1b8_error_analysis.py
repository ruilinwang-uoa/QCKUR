"""B1-vs-B8 per-instance error analysis (revision Phase 1.1).

Decomposes the 14.4-point Final gap between the template-only system (B1,
98.3) and the full framework (B8, 83.9) on the n=100 AI4I primary sample:

  1. sanity check: reproduce tab:main layer/Final means for B1/B2/B8;
  2. layer-term decomposition of the mean gap (paper weights, Eq. finalscore);
  3. per-instance gap distribution (faulted vs healthy);
  4. Layer-1 sub-component comparison (numeric artifact vs TWF classification misses);
  5. Layer-2 component comparison (consistency / contradiction / coverage);
  6. Layer-3 per-dimension comparison (quantify where the rubric favors B1);
  7. repaired-protocol Final recompute (repaired numeric + coverage-free L3)
     for B1/B2/B8 -- fills the B1 number missing from the paper (B8 93.2,
     B2 86.5 are quoted in Section res-pdm);
  8. text-level B1 vs B8 statistics (length / lexical diversity proxies for
     the fluency the rubric cannot perceive);
  9. dump of every B8 instance flagged by the Layer-2 judge (contradiction>0
     or consistency<1) with report text + facts, for manual distortion
     taxonomy (see b1b8_error_analysis.md).

No LLM calls; reads only the canonical parquets in code/runs.
Outputs: code/analysis/b1b8_*.csv, code/analysis/b1b8_contradiction_dump.md,
and a console summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

# Composite weights (experiment plan v6.0; Eqs. layer1-layer3, finalscore)
W1, W2, W3 = 0.35, 0.30, 0.35
L1W = {"numerical": 0.30, "rule": 0.30, "classification": 0.40}
L2W = {"consistency": 0.50, "coverage": 0.25, "contradiction": 0.25}
L3W = {"faithfulness": 0.30, "completeness": 0.15, "coverage": 0.25,
       "coherence": 0.15, "actionability": 0.15}

# Repaired-scorer numeric component (tab:audit): ties at 1.000 for B1/B2/B8.
S_NUM_REPAIRED = {"B1": 1.000, "B2": 1.000, "B8": 1.000}


def load() -> pd.DataFrame:
    base = pd.read_parquet(RUNS / "reports_main_glm_n100_B1B2B8_v3.parquet")[
        ["method", "instance_id", "udi", "machine_failure", "active_modes",
         "parsed_answer"]].rename(columns={"parsed_answer": "report"})
    l1 = pd.read_parquet(RUNS / "reports_main_glm_n100_B1B2B8_v3_l1.parquet")
    l2 = pd.read_parquet(RUNS / "reports_main_glm_n100_B1B2B8_v3_l2.parquet")\
        .rename(columns={"coverage": "cov2"})
    l3 = pd.read_parquet(RUNS / "reports_main_glm_n100_B1B2B8_v3_l3.parquet")
    df = (base.merge(l1, on=["method", "instance_id"])
              .merge(l2, on=["method", "instance_id"])
              .merge(l3, on=["method", "instance_id"]))
    df["Final"] = W1 * 100 * df.L1 + W2 * 100 * df.L2 + W3 * 20 * df.L3
    # Layer-3 coverage is a per-aspect 0/1 dict -> aspect count (0-6)
    df["cov3_count"] = df["coverage"].map(
        lambda d: float(sum(d.values())) if isinstance(d, dict) else np.nan)
    return df


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def main() -> None:
    df = load()

    # ---- 1. sanity: reproduce tab:main -------------------------------------
    print("== 1. Layer/Final means (n=100; tab:main reference in brackets) ==")
    ref = {"B1": (0.987, 0.981, 4.899, 98.3), "B2": (0.988, 0.700, 3.958, 83.3),
           "B8": (0.715, 0.841, 4.807, 83.9)}
    for m, (r1, r2, r3, rf) in ref.items():
        d = df[df.method == m]
        print(f"  {m}: L1={d.L1.mean():.3f}[{r1}] L2={d.L2.mean():.3f}[{r2}] "
              f"L3={d.L3.mean():.3f}[{r3}] Final={d.Final.mean():.1f}[{rf}]")

    # ---- 2. layer-term decomposition of the mean gap -----------------------
    print("\n== 2. B1-B8 mean gap decomposition (points of Final) ==")
    means = df.groupby("method")[["L1", "L2", "L3"]].mean()
    d1 = 100 * W1 * (means.L1.B1 - means.L1.B8)
    d2 = 100 * W2 * (means.L2.B1 - means.L2.B8)
    d3 = 20 * W3 * (means.L3.B1 - means.L3.B8)
    total = d1 + d2 + d3
    for name, val in [("L1 term", d1), ("L2 term", d2), ("L3 term", d3)]:
        print(f"  {name}: {val:+.2f} pts ({val/total*100:.0f}% of gap)")
    print(f"  total: {total:+.2f} pts (table: 98.3-83.9 = 14.4)")

    # ---- 3. per-instance gap distribution ----------------------------------
    print("\n== 3. Per-instance Final gap (B1 - B8) ==")
    piv = df.pivot(index="instance_id", columns="method", values="Final")
    meta = df[df.method == "B1"].set_index("instance_id")[["machine_failure"]]
    gap = (piv.B1 - piv.B8).to_frame("gap").join(meta)
    for lo, hi in [(-99, 0), (0, 10), (10, 20), (20, 99)]:
        sel = gap[(gap.gap >= lo) & (gap.gap < hi)]
        print(f"  gap in [{lo},{hi}): n={len(sel)} "
              f"(faulted={sel.machine_failure.sum()})")
    print(f"  median gap={gap.gap.median():.1f}, mean={gap.gap.mean():.1f}")
    w = stats.wilcoxon(piv.B1, piv.B8, zero_method="wilcox")
    print(f"  Wilcoxon paired B1 vs B8: p={w.pvalue:.2e}")
    gap.reset_index().to_csv(OUT / "b1b8_per_instance_gap.csv", index=False)

    # ---- 4. Layer-1 sub-components ------------------------------------------
    print("\n== 4. Layer-1 sub-components (mean) ==")
    subs = df.groupby("method")[["numerical", "rule", "classification"]].mean()
    print(subs.loc[["B1", "B2", "B8"]].round(3).to_string())
    b8 = df[df.method == "B8"]
    cls_miss = b8[b8.classification < 1]
    print("  B8 classification<1 instances:")
    for _, r in cls_miss.iterrows():
        print(f"    {r.instance_id} faulted={r.machine_failure} "
              f"true={r.true_modes} cls={r.classification}")

    # ---- 5. Layer-2 components ----------------------------------------------
    print("\n== 5. Layer-2 components (mean) ==")
    l2c = df.groupby("method")[["consistency", "contradiction", "cov2",
                                "n_claims"]].mean()
    print(l2c.loc[["B1", "B2", "B8"]].round(3).to_string())
    n_contra = (b8.contradiction > 0).sum()
    n_incons = (b8.consistency < 1).sum()
    print(f"  B8 instances with contradiction>0: {n_contra}/100; "
          f"consistency<1: {n_incons}/100")
    print(f"  B8 claim-weighted contradiction rate: "
          f"{(b8.contradiction * b8.n_claims).sum() / b8.n_claims.sum():.3f}")
    for m in ["B1", "B2"]:
        d = df[df.method == m]
        print(f"  {m} claim-weighted contradiction rate: "
              f"{(d.contradiction * d.n_claims).sum() / d.n_claims.sum():.3f} "
              f"(noise floor reference)")

    # ---- 6. Layer-3 per-dimension -------------------------------------------
    print("\n== 6. Layer-3 dimensions (mean; cov as count/6 scaled to /5) ==")
    dim_cols = ["faithfulness", "completeness", "coherence", "actionability",
                "cov3_count"]
    dims = df.groupby("method")[dim_cols].mean()
    dims["cov_scale5"] = dims["cov3_count"] / 6.0 * 5.0
    print(dims.loc[["B1", "B2", "B8"]].round(2).to_string())
    for dcol in dim_cols + ["cov_scale5"]:
        col = df if dcol != "cov_scale5" else df.assign(
            cov_scale5=df.cov3_count / 6.0 * 5.0)
        x = col[col.method == "B8"].set_index("instance_id")[dcol]
        y = col[col.method == "B1"].set_index("instance_id")[dcol]
        p = stats.wilcoxon(x, y, zero_method="wilcox").pvalue
        print(f"  B8 vs B1 {dcol}: {x.mean():.2f} vs {y.mean():.2f} "
              f"(paired Wilcoxon p={p:.3g})")

    # ---- 7. repaired-protocol Finals ----------------------------------------
    print("\n== 7. Repaired protocol (repaired numeric + coverage-free L3) ==")
    for m in ["B1", "B2", "B8"]:
        d = df[df.method == m]
        l1p = (L1W["numerical"] * S_NUM_REPAIRED[m]
               + L1W["rule"] * d.rule + L1W["classification"] * d.classification)
        l3cf = ((L3W["faithfulness"] * d.faithfulness
                 + L3W["completeness"] * d.completeness
                 + L3W["coherence"] * d.coherence
                 + L3W["actionability"] * d.actionability)
                / (1 - L3W["coverage"]))
        finalp = W1 * 100 * l1p + W2 * 100 * d.L2 + W3 * 20 * l3cf
        print(f"  {m}: L1'={l1p.mean():.3f} L3_cf={l3cf.mean():.3f} "
              f"Final'={finalp.mean():.1f} (per-instance mean)")
    print("  reference: B8 93.2, B2 86.5 quoted in paper (layer-mean method)")

    # ---- 8. text-level statistics -------------------------------------------
    print("\n== 8. Text statistics (B1 vs B8) ==")
    for m in ["B1", "B8"]:
        d = df[df.method == m]
        wc = d.report.map(word_count)
        ttr = d.report.map(lambda t: len(set(re.findall(r"[a-z]+", t.lower())))
                           / max(1, len(re.findall(r"[a-z]+", t.lower()))))
        print(f"  {m}: words mean={wc.mean():.0f} sd={wc.std():.0f}; "
              f"type-token ratio={ttr.mean():.3f}")
    b8w = df[df.method == "B8"].set_index("instance_id").report.map(word_count)
    b1w = df[df.method == "B1"].set_index("instance_id").report.map(word_count)
    print(f"  length ratio B8/B1 mean={(b8w/b1w).mean():.2f}")

    # ---- 9. contradiction dump for manual taxonomy ---------------------------
    flagged = b8[(b8.contradiction > 0) | (b8.consistency < 1)].sort_values(
        ["contradiction", "consistency"], ascending=[False, True])
    lines = ["# B8 instances flagged by the Layer-2 judge",
             "",
             f"n={len(flagged)} of 100 (contradiction>0 or consistency<1).",
             "Sorted by contradiction desc, consistency asc.",
             ""]
    for i, (_, r) in enumerate(flagged.iterrows(), 1):
        lines += [f"## {i}. {r.instance_id} (udi={r.udi}, "
                  f"faulted={r.machine_failure}, true_modes={r.true_modes})",
                  f"- L2: consistency={r.consistency:.2f}, "
                  f"contradiction={r.contradiction:.2f}, "
                  f"L2-coverage={r.cov2:.2f}, n_claims={r.n_claims}",
                  "", "```text", r.report, "```", ""]
    (OUT / "b1b8_contradiction_dump.md").write_text("\n".join(lines),
                                                    encoding="utf-8")
    print(f"\nDumped {len(flagged)} flagged instances -> "
          f"analysis/b1b8_contradiction_dump.md")


if __name__ == "__main__":
    main()
