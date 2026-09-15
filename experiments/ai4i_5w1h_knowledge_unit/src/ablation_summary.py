"""Ablation analysis (Section 5.4 / RQ5): each component's contribution.

Computes per-instance Final for B8 and each ablation (AblT/AblM/AblQ/AblP),
then the paired Final drop when that component is removed (B8 - ablation) with a
Wilcoxon signed-rank test. A large, significant drop => the component is
necessary.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

import config as C

B8_SRC = "reports_main_glm_n100_B1B2B8_v3"   # B8 rule-engine run
ABL_SRC = "reports_ablations_glm_n100"
ABLATIONS = ["AblT", "AblM", "AblQ", "AblP"]
DESC = {"AblT": "drop answer templates", "AblM": "drop model handler",
        "AblQ": "drop question matching", "AblP": "drop physical rules"}


def per_instance_final(src, method):
    l1 = pd.read_parquet(C.RUNS_DIR / f"{src}_l1.parquet")
    l2 = pd.read_parquet(C.RUNS_DIR / f"{src}_l2.parquet")
    l3 = pd.read_parquet(C.RUNS_DIR / f"{src}_l3.parquet")
    l1 = l1[l1["method"] == method][["instance_id", "L1"]]
    l2 = l2[l2["method"] == method][["instance_id", "L2"]]
    l3 = l3[l3["method"] == method][["instance_id", "L3"]]
    m = l1.merge(l2, on="instance_id").merge(l3, on="instance_id")
    m["Final"] = 0.35 * (m["L1"] * 100) + 0.30 * (m["L2"] * 100) + 0.35 * (m["L3"] * 20)
    return m.set_index("instance_id")["Final"]


def main():
    b8 = per_instance_final(B8_SRC, "B8")
    rows = [{"config": "B8 (full)", "Final": round(b8.mean(), 1), "drop": 0.0,
             "p": "", "n": b8.shape[0]}]
    for ab in ABLATIONS:
        try:
            a = per_instance_final(ABL_SRC, ab)
        except FileNotFoundError as e:
            print("skip", ab, e); continue
        paired = pd.concat({"B8": b8, ab: a}, axis=1).dropna()
        drop = (paired["B8"] - paired[ab]).mean()
        try:
            _, p = stats.wilcoxon(paired["B8"], paired[ab])
        except ValueError:
            p = 1.0
        rows.append({"config": f"{ab} ({DESC[ab]})", "Final": round(a.mean(), 1),
                     "drop": round(drop, 1), "p": f"{p:.1e}", "n": paired.shape[0]})

    df = pd.DataFrame(rows)
    df.to_csv(C.RUNS_DIR / "ablation_summary.csv", index=False)
    print(df.to_string(index=False))

    L = ["% Auto-generated: ablation contributions (n=100, paired vs B8)\n",
         "\\begin{tabular}{lrrrl}\n\\toprule\n",
         "Configuration & Final & $\\Delta$ & Wilcoxon $p$ & \\\\\n\\midrule\n"]
    for _, r in df.iterrows():
        L.append(f"{r['config']} & {r['Final']} & {r['drop']:+.1f} & {r['p']} & \\\\\n")
    L.append("\\bottomrule\n\\end{tabular}\n")
    (C.ANALYSIS_DIR / "ablation_table.tex").write_text("".join(L), encoding="utf-8")
    print("\nSaved runs/ablation_summary.csv and analysis/ablation_table.tex")


if __name__ == "__main__":
    main()
