"""
metropt_human_analysis.py -- paper-grade analysis of the MetroPT human
validation ratings (human_validation/ratings_in/*.json).

Complements the package's minimal analyze_metropt_human.py with the stats
the paper's human-validation section needs:

  A. per-system per-dimension means + 6-dim overall            (CSV)
  B. stratum x system table: state (detection) + faith         (CSV)
     strata: sig-visible positives (8) / non-visible positives (7)
             nominal controls (12) / OOC-candidate controls (3)
  C. readability: mean rank + most-readable picks (tie-aware)  (CSV)
  D. inter-rater: report-level pairwise Spearman per dimension,
     mean Kendall W for readability rankings
  E. human-vs-judge: report-level + system-level Spearman/Pearson
     against metropt_l3.parquet (primary DS judge)
  F. Wilcoxon signed-rank tests (paired by rater x instance):
     overall B8-B1 / B8-B2 / B8-B5; readability B8 vs B1 / B8 vs B2;
     state on signature-visible positives
  G. per-rater system means (consistency of the ordering)
  H. spot-check hooks: which sampled B8 reports carry known L1
     numeric/classification failures

Outputs -> human_validation/analysis/ and prints a digest.
Keys are ALWAYS (vehicle, window_id) -- window_id alone collides across
vehicles (hit 3x before in this project; do not relearn it here).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                 # metropt_5w1h root
PKG = ROOT / "human_validation"
RATINGS = PKG / "ratings_in"
RUNS = ROOT / "runs" / "metropt3"
OUT = PKG / "analysis"
SYSTEMS = ["B1", "B2", "B5", "B8"]
DIMS = ["faith", "compl", "coh", "act", "state", "actionok"]

# strata from analysis/SELECTION_LOG.md (kind, visible) -> name
STRATA = {
    ("positive", True): "sig_visible_pos",
    ("positive", False): "nonvisible_pos",
    ("control", False): "nominal_ctl",
    ("control", True): "ooc_ctl",
}


def load_strata() -> dict[tuple[str, str], str]:
    """Parse SELECTION_LOG.md 'vehicle window_id (kind=..., visible=...)'."""
    strata: dict[tuple[str, str], str] = {}
    txt = (PKG / "analysis" / "SELECTION_LOG.md").read_text(encoding="utf-8")
    for line in txt.splitlines():
        line = line.strip().lstrip("- ")
        if "(kind=" not in line:
            continue
        head, tail = line.split(" (kind=", 1)
        vehicle, window_id = head.split()
        kind = tail.split(",")[0].strip()
        visible = tail.split("visible=")[1].rstrip(")")
        strata[(vehicle, window_id)] = STRATA[(kind, visible == "True")]
    return strata


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
    df = pd.DataFrame(rows)
    df["inst"] = df.vehicle + "|" + df.window_id
    strata = load_strata()
    df["stratum"] = [strata[(v, w)] for v, w in zip(df.vehicle, df.window_id)]
    df["overall"] = df[DIMS].mean(axis=1)
    return df


def kendall_w(mat: np.ndarray) -> float:
    """W for m raters (rows) ranking n items (cols); tie-corrected."""
    m, n = mat.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, mat)
    col_sums = ranks.sum(axis=0)
    S = ((col_sums - col_sums.mean()) ** 2).sum()
    ties = 0.0
    for row in ranks:
        _, cnt = np.unique(row, return_counts=True)
        ties += (cnt**3 - cnt).sum()
    denom = m**2 * (n**3 - n) - m * ties
    return float(m * S / denom) if denom else np.nan


def wilcoxon_paired(df: pd.DataFrame, col: str, a: str, b: str) -> dict:
    """Paired by (rater, inst); returns median diff + Wilcoxon."""
    pa = df[df.system == a].set_index(["rater", "inst"])[col]
    pb = df[df.system == b].set_index(["rater", "inst"])[col]
    common = pa.index.intersection(pb.index)
    d = (pa[common] - pb[common]).dropna()
    if d.empty:
        return {"n": 0}
    res = stats.wilcoxon(d) if (d != 0).any() else None
    return {"n": len(d), "median_diff": float(d.median()),
            "mean_diff": float(d.mean()),
            "p": float(res.pvalue) if res else 1.0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"raters {df.rater.nunique()} | instances {df.inst.nunique()} "
          f"| rows {len(df)} | strata {df.stratum.value_counts().to_dict()}")

    # ---- A. per-system means ------------------------------------------------
    m = df.groupby("system")[DIMS + ["overall", "readability_rank"]].mean()
    m["n"] = df.groupby("system").size()
    m.reindex(SYSTEMS).round(3).to_csv(OUT / "human_system_means.csv")
    print("\n=== A. per-system means ===")
    print(m.reindex(SYSTEMS).round(3).to_string())

    # ---- B. stratum x system ------------------------------------------------
    rows = []
    for (strat, sys_id), g in df.groupby(["stratum", "system"]):
        rows.append({"stratum": strat, "system": sys_id,
                     "state": g.state.mean(), "faith": g.faith.mean(),
                     "actionok": g.actionok.mean(), "overall": g.overall.mean(),
                     "n": len(g)})
    sb = pd.DataFrame(rows).sort_values(["stratum", "system"])
    sb.to_csv(OUT / "human_stratum_system.csv", index=False)
    print("\n=== B. stratum x system (state = state-claim correctness) ===")
    print(sb.pivot(index="stratum", columns="system", values="state").round(2).to_string())
    print("\n=== B2. stratum x system (faith) ===")
    print(sb.pivot(index="stratum", columns="system", values="faith").round(2).to_string())

    # ---- C. readability -----------------------------------------------------
    mr = (df.dropna(subset=["readability_rank"])
            .groupby("system").readability_rank.agg(["mean", "count"]))
    picks = {}
    tie_picks = 0
    for (inst, rater), g in df.dropna(subset=["readability_rank"]) \
            .groupby(["inst", "rater"]):
        rmin = g.readability_rank.min()
        winners = g[g.readability_rank == rmin].system.tolist()
        if len(winners) > 1:
            tie_picks += 1
        for w in winners:
            picks[w] = picks.get(w, 0) + 1
    print("\n=== C. readability ===")
    print(mr.reindex(SYSTEMS).round(3).to_string())
    print("most-readable picks (tie-aware):", {s: picks.get(s, 0) for s in SYSTEMS},
          f"| instances with tied-best: {tie_picks}")
    pd.DataFrame({"mean_rank": mr["mean"], "picks": [picks.get(s, 0) for s in SYSTEMS]},
                 index=SYSTEMS).to_csv(OUT / "human_readability.csv")

    # ---- D. inter-rater -----------------------------------------------------
    print("\n=== D. inter-rater (report-level pairwise Spearman per dim) ===")
    inter = {}
    for dim in DIMS:
        piv = df.pivot_table(index=["inst", "system"], columns="rater", values=dim)
        cors = [piv[a].corr(piv[b], method="spearman")
                for a, b in itertools.combinations(piv.columns, 2)]
        inter[dim] = float(np.mean(cors))
        print(f"  {dim:9s}: {inter[dim]:.3f}")
    piv_o = df.pivot_table(index=["inst", "system"], columns="rater", values="overall")
    inter["overall"] = float(np.mean(
        [piv_o[a].corr(piv_o[b], method="spearman")
         for a, b in itertools.combinations(piv_o.columns, 2)]))
    print(f"  overall  : {inter['overall']:.3f}")
    ws = []
    for inst, g in df.dropna(subset=["readability_rank"]).groupby("inst"):
        mat = (g.pivot(index="rater", columns="system", values="readability_rank")
                 .reindex(columns=SYSTEMS).to_numpy())
        if mat.shape[0] >= 2:
            ws.append(kendall_w(mat))
    print(f"  Kendall W (readability, mean over {len(ws)} windows): {np.nanmean(ws):.3f}")
    pd.Series(inter).to_csv(OUT / "human_inter_rater.csv", header=["mean_pairwise_spearman"])

    # ---- E. human vs judge --------------------------------------------------
    print("\n=== E. human vs automatic judge ===")
    l3p = RUNS / "metropt_l3.parquet"
    if l3p.exists():
        l3 = pd.read_parquet(l3p)
        l3 = l3[l3.ok.astype(str) == "True"]
        l3 = l3.groupby(["system", "vehicle", "window_id"]).L3.mean().reset_index()
        hh = df.groupby(["system", "vehicle", "window_id"]).overall.mean().reset_index()
        j = hh.merge(l3, on=["system", "vehicle", "window_id"], how="inner")
        print(f"  joined reports: {len(j)}")
        sp = j.overall.corr(j.L3, method="spearman")
        pe = j.overall.corr(j.L3)
        print(f"  report-level Spearman {sp:.3f} | Pearson {pe:.3f}")
        hs = df.groupby("system").overall.mean().reindex(SYSTEMS)
        js = l3.groupby("system").L3.mean().reindex(SYSTEMS)
        print(f"  system-level Spearman {hs.corr(js, method='spearman'):.3f}")
        print(f"  judge L3 means: {js.round(3).to_dict()}")
        print(f"  human overall : {hs.round(3).to_dict()}")
        j.to_csv(OUT / "human_vs_judge_reportlevel.csv", index=False)

    # ---- F. tests -----------------------------------------------------------
    print("\n=== F. Wilcoxon signed-rank (paired rater x instance) ===")
    tests = {}
    for other in ["B1", "B2", "B5"]:
        r = wilcoxon_paired(df, "overall", "B8", other)
        tests[f"overall_B8-{other}"] = r
        print(f"  overall  B8-{other}: n={r['n']} mean {r.get('mean_diff', 0):+.3f} "
              f"p={r['p']:.4g}")
    for other in ["B1", "B2", "B5"]:
        r = wilcoxon_paired(df, "readability_rank", "B8", other)
        tests[f"read_B8-{other}"] = r
        print(f"  readable B8-{other}: n={r['n']} mean {r.get('mean_diff', 0):+.3f} "
              f"(negative = B8 easier) p={r['p']:.4g}")
    vis = df[df.stratum == "sig_visible_pos"]
    for other in ["B1", "B2", "B5"]:
        r = wilcoxon_paired(vis, "state", "B8", other)
        tests[f"state_vis_B8-{other}"] = r
        print(f"  state@visible B8-{other}: n={r['n']} mean {r.get('mean_diff', 0):+.3f} "
              f"p={r['p']:.4g}")
    for other in ["B8", "B5"]:
        r = wilcoxon_paired(vis, "state", "B1", other)
        tests[f"state_vis_B1-{other}"] = r
        print(f"  state@visible B1-{other}: n={r['n']} mean {r.get('mean_diff', 0):+.3f} "
              f"p={r['p']:.4g}")
    (OUT / "human_tests.json").write_text(json.dumps(tests, indent=1))

    # ---- G. per-rater system means -------------------------------------------
    print("\n=== G. per-rater system overall means (ordering consistency) ===")
    g = df.pivot_table(index="rater", columns="system", values="overall")
    print(g[SYSTEMS].round(3).to_string())
    for r in g.index:
        print(f"  {r} ordering: {' > '.join(g.loc[r].sort_values(ascending=False).index)}")
    g[SYSTEMS].round(3).to_csv(OUT / "human_per_rater.csv")

    # ---- H. known L1 failures inside the sample ------------------------------
    print("\n=== H. L1-failure overlap with sample (from metropt_l1.csv) ===")
    l1p = RUNS / "metropt_l1.csv"
    if l1p.exists():
        l1 = pd.read_csv(l1p)
        keyed = df[["vehicle", "window_id", "system"]].drop_duplicates()
        for col, ok in [("numeric", "numeric<1"), ("classification", "classification<1"),
                        ("rule", "rule<1")]:
            bad = l1[l1[col] < 1.0]
            merged = keyed.merge(bad[["system", "vehicle", "window_id", col]],
                                 on=["system", "vehicle", "window_id"], how="inner")
            print(f"  {ok} in sample: {len(merged)} -> "
                  f"{merged[['system', 'vehicle', 'window_id']].to_dict('records')}")

    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
