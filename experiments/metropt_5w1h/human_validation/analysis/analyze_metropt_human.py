"""
analyze_metropt_human.py -- analyze the MetroPT human ratings once the
raters have dropped their JSONs into ../ratings_in/.

Run from anywhere with pandas/numpy/scipy available:
    python analyze_metropt_human.py

Outputs (printed + ../analysis/human_metropt_summary.json):
  * per-system means on every dimension + overall (L3-like mean)
  * readability rank distribution + most-readable counts
  * inter-rater agreement (mean pairwise Spearman on system means;
    Kendall W per dimension)
  * human vs automatic L3 judge agreement (if metropt_l3.parquet is
    reachable at ../../runs/metropt3/)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RATINGS = PKG / "ratings_in"
RUNS = PKG.parent / "runs" / "metropt3"
SYSTEMS = ["B1", "B2", "B5", "B8"]
DIMS = ["faith", "compl", "coh", "act", "state", "actionok"]


def load() -> pd.DataFrame:
    rows = []
    for p in sorted(RATINGS.glob("human_validation_metropt[-_]*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rater = d.get("rater") or p.stem.split("-")[-1]
        for rec in d["ratings"]:
            for sys_id, o in rec["systems"].items():
                row = {"rater": rater, "window_id": rec["window_id"],
                       "vehicle": rec["vehicle"], "system": sys_id}
                row.update({k: o.get(k) for k in DIMS})
                row["readability_rank"] = o.get("readability_rank")
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    if df.empty:
        print(f"no ratings found in {RATINGS} -- waiting for raters")
        return
    df["inst"] = df.vehicle + "|" + df.window_id
    print(f"raters: {df.rater.nunique()} | windows: {df.inst.nunique()} "
          f"| rows: {len(df)}")
    dims = DIMS + ["readability_rank"]
    m = df.groupby("system")[dims].mean().round(3)
    m["overall"] = df.groupby("system")[DIMS].mean().mean(axis=1).round(3)
    m["n"] = df.groupby("system").size()
    print("\n=== per-system human means ===")
    print(m.reindex(SYSTEMS).to_string())

    most_read = (df.dropna(subset=["readability_rank"])
                 .groupby(["inst", "rater"], group_keys=False)
                 .apply(lambda d: d.loc[d.readability_rank.idxmin(),
                                        "system"]))
    print("\n=== most-readable picks ===")
    print(most_read.value_counts().reindex(SYSTEMS).to_string())

    # inter-rater: pairwise Spearman between raters on system means
    sm = (df.groupby(["rater", "system"])[DIMS].mean()
          .reset_index().pivot(index="rater", columns="system",
                               values="faith"))
    if sm.shape[0] >= 2:
        cors = []
        rs = list(sm.index)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                c = sm.loc[rs[i]].corr(sm.loc[rs[j]], method="spearman")
                cors.append(c)
        print(f"\ninter-rater mean pairwise Spearman (faith, system-level, "
              f"n_systems={sm.shape[1]}): {np.mean(cors):.3f}")

    out = {"n_raters": int(df.rater.nunique()),
           "n_windows": int(df.window_id.nunique()),
           "system_means": m.reindex(SYSTEMS).to_dict("index"),
           "most_readable": most_read.value_counts().to_dict()}
    dest = HERE / "human_metropt_summary.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nsummary -> {dest}")

    l3p = RUNS / "metropt_l3.parquet"
    if l3p.exists():
        l3 = pd.read_parquet(l3p)
        l3 = l3[l3.ok.astype(str) == "True"]
        hm = df.groupby("system")[DIMS].mean().mean(axis=1)
        jm = l3.groupby("system").L3.mean()
        common = [s for s in SYSTEMS if s in hm.index and s in jm.index]
        c = pd.Series(hm[common]).corr(pd.Series(jm[common]),
                                       method="spearman")
        print(f"human-vs-judge system-level Spearman (n={len(common)}): "
              f"{c:.3f}")


if __name__ == "__main__":
    main()
