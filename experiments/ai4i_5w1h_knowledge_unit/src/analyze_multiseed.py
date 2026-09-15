"""Multi-seed baseline analysis (round-5 review, Required Analysis 3).

Question: B8 was re-generated under seeds 1 and 7 while the baselines ran under
seed 42 only. Does B8's lead over each LLM-based baseline survive when the
BASELINE side also gets seed variance?

Inputs (after run_multiseed_baselines.bat completes):
  code/runs/reports_main_glm_n100_B3toB7_seed{1,7}{,_l1,_l2,_l3}.parquet
plus the existing seed-42 primary scores and the B8 seed replicates:
  B8 s42 : code/runs/reports_main_glm_n100_B1B2B8_v3_{l1,l2,l3}.parquet
  B8 s1/s7: repo/experiments/ai4i_5w1h_knowledge_unit/runs/reports_main_glm_n100_B8_seed{1,7}_{l1,l2,l3}.parquet
  B3-B6 s42: code/runs/reports_main_glm_n100_{l1,l2,l3}.parquet
  B7 s42 : code/runs/reports_main_glm_n100_B7_v2_{l1,l2,l3}.parquet

Composite: Final = 0.35*(L1*100) + 0.30*(L2*100) + 0.35*(L3*20).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPO_RUNS = RUNS
OUT = ROOT / "analysis" / "multiseed_baselines.csv"

SYSTEMS = ["B3", "B4", "B5", "B6", "B7"]
SEEDS = [42, 1, 7]


def _layer(run_dir: Path, stem: str, layer: str, systems) -> pd.DataFrame:
    path = run_dir / f"{stem}_{layer}.parquet"
    d = pd.read_parquet(path)
    d = d[d["method"].isin(systems)][["method", "instance_id", layer.upper()]]
    return d


def final_table(seed: int) -> pd.DataFrame:
    """Per-instance Final for B3-B7 (+B8) at one seed, from honest sources."""
    frames = []
    if seed == 42:
        for layer in ["l1", "l2", "l3"]:
            parts = [_layer(RUNS, "reports_main_glm_n100", layer, ["B3", "B4", "B5", "B6"]),
                     _layer(RUNS, "reports_main_glm_n100_B7_v2", layer, ["B7"]),
                     _layer(RUNS, "reports_main_glm_n100_B1B2B8_v3", layer, ["B8"])]
            frames.append(pd.concat(parts))
    else:
        for layer in ["l1", "l2", "l3"]:
            parts = [_layer(RUNS, f"reports_main_glm_n100_B3toB7_seed{seed}", layer, SYSTEMS),
                     _layer(REPO_RUNS, f"reports_main_glm_n100_B8_seed{seed}", layer, ["B8"])]
            frames.append(pd.concat(parts))
    inst = frames[0]
    for f in frames[1:]:
        inst = inst.merge(f, on=["method", "instance_id"], how="outer")
    inst["Final"] = 0.35 * inst["L1"] * 100 + 0.30 * inst["L2"] * 100 + 0.35 * inst["L3"] * 20
    return inst.pivot_table(index="instance_id", columns="method", values="Final")


def main() -> None:
    tables = {}
    for s in SEEDS:
        try:
            tables[s] = final_table(s)
            print(f"seed {s}: {tables[s].shape[0]} instances x {tables[s].shape[1]} systems")
        except FileNotFoundError as e:
            print(f"seed {s}: MISSING INPUT — {e.filename}")
    if len(tables) < 2:
        print("Need at least two seeds; run code/run_multiseed_baselines.bat first.")
        return

    # per-seed system summary
    rows = []
    for s in SEEDS:
        if s not in tables:
            continue
        for m in ["B8"] + SYSTEMS:
            if m in tables[s].columns:
                v = tables[s][m].dropna()
                rows.append({"seed": s, "system": m, "n": len(v),
                             "Final_mean": round(v.mean(), 2), "Final_sd": round(v.std(), 2)})
    summ = pd.DataFrame(rows)
    print("\n=== Per-seed Final (mean +/- sd) ===")
    print(summ.pivot_table(index="system", columns="seed", values="Final_mean").round(2).to_string())

    # B8 margin over each baseline, per seed + pooled
    lines = ["\n=== B8 margin over each baseline (mean paired diff; Wilcoxon p) ===",
             f"{'baseline':8s} " + " ".join(f"{'s'+str(s):>16s}" for s in SEEDS if s in tables)]
    pool_rows = []
    for m in SYSTEMS:
        cells = []
        for s in SEEDS:
            if s not in tables or m not in tables[s].columns:
                cells.append(f"{'—':>16s}")
                continue
            t = tables[s][["B8", m]].dropna()
            diff = t["B8"] - t[m]
            try:
                _, p = stats.wilcoxon(t["B8"], t[m], zero_method="wilcox")
            except ValueError:
                p = float("nan")
            cells.append(f"{diff.mean():+.1f} (p={p:.0e})".rjust(16))
            pool_rows.append({"baseline": m, "seed": s, "margin": round(diff.mean(), 2),
                              "wilcoxon_p": p})
        lines.append(f"{m:8s} " + " ".join(cells))
    print("\n".join(lines))

    pd.DataFrame(pool_rows).to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")
    print("\nHeadline check: does the B8-over-B5 margin (seed-42: +22.6) hold at seeds 1 and 7,")
    print("and is any baseline seed draw within seed noise of B8 (B8 across-seed sd = 2.1)?")


if __name__ == "__main__":
    main()
