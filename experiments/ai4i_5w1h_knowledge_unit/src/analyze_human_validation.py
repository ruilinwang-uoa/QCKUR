"""Analyze human-validation ratings against the LLM judges (AI4I).

Usage:
    python analyze_human_validation.py rating1.json [rating2.json ...]

Each input is the Export file from rating_form.html (one per rater). Uses
sample_manifest.json to unblind letters -> systems (B1/B2/B5/B8), then:

  1. per-dimension human system means + 5W1H coverage;
  2. human-vs-judge agreement: per-instance Pearson correlations per dimension
     and for the L3 composite, against BOTH the primary DeepSeek judge and the
     family-disjoint Kimi fourth judge; Spearman over system means;
  3. the fluency-inversion test: readability ranks, B1 vs B8 paired;
  4. inter-rater agreement: Fleiss's kappa on the top-readability pick,
     mean pairwise Pearson across dimensions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config as C

HV = C.RUNS_DIR / "human_validation"
DIMS = {"faith": "faithfulness", "compl": "completeness",
        "coh": "coherence", "act": "actionability"}
ASPECTS = ["what", "why", "when", "where", "who", "how"]
L3_W = {"faith": 0.30, "compl": 0.15, "coh": 0.15, "act": 0.15}


def cov6(cov) -> int:
    return int(sum(bool(cov.get(a)) for a in ASPECTS))


def l3_human(r) -> float:
    return round(sum(L3_W[k] * r[k] for k in L3_W) + 0.25 * (cov6(r["cov"]) / 6) * 5, 3)


def load_judge(stem: str, method_map: dict) -> pd.DataFrame | None:
    try:
        d = pd.read_parquet(C.RUNS_DIR / f"{stem}_l3.parquet")
        d = d[d["method"].isin(method_map)]
        d = d.assign(instance_id=d["instance_id"].astype(str))
        return d.set_index(["method", "instance_id"])
    except FileNotFoundError:
        return None


def main() -> None:
    files = [Path(a) for a in sys.argv[1:]]
    if not files:
        print("usage: python analyze_human_validation.py ratings1.json [...]"); return
    manifest = json.loads((HV / "sample_manifest.json").read_text(encoding="utf-8"))
    key = manifest["key"]

    rows = []
    for fidx, f in enumerate(files):
        raw = json.loads(f.read_text(encoding="utf-8"))
        for i, inst in enumerate(manifest["instances"]):
            s = raw.get(str(i)) or raw.get(i) or {}
            for letter, system in key[inst].items():
                r = (s.get("reports") or {}).get(letter)
                if not r:
                    continue
                rows.append({"rater": f"r{fidx+1}", "instance_id": inst,
                             "system": system, **{k: r[k] for k in DIMS},
                             "cov6": cov6(r["cov"]), "l3h": l3_human(r),
                             "rank": (s.get("rank") or {}).get(letter)})
    h = pd.DataFrame(rows)
    if h.empty:
        print("no ratings parsed - check the input files"); return
    print(f"ratings: {h.shape[0]} from {h['rater'].nunique()} rater(s), "
          f"{h['instance_id'].nunique()} instances, systems {sorted(h['system'].unique())}")

    print("\n=== human system means (Layer-3 aligned dimensions) ===")
    g = h.groupby("system").agg(n=("l3h", "size"), faith=("faith", "mean"),
                                compl=("compl", "mean"), coh=("coh", "mean"),
                                act=("act", "mean"), cov6=("cov6", "mean"),
                                L3_human=("l3h", "mean"),
                                rank=("rank", "mean")).round(2)
    print(g.to_string())

    print("\n=== fluency-inversion test (readability rank, lower = more readable) ===")
    wide = h.pivot_table(index=["rater", "instance_id"], columns="system", values="rank")
    if {"B1", "B8"} <= set(wide.columns) and wide["B8"].notna().any():
        d = wide["B8"] - wide["B1"]          # negative = B8 ranked MORE readable
        dd = d.dropna()
        if len(dd):
            try:
                w = stats.wilcoxon(dd, zero_method="wilcox", method="auto")
                pv = f"{w.pvalue:.2e}"
            except ValueError:
                pv = "n/a"
            print(f"  B8 mean rank {wide['B8'].mean():.2f} vs B1 {wide['B1'].mean():.2f}; "
                  f"B8 more readable in {(dd < 0).mean():.0%} of paired samples "
                  f"(n={len(dd)}), Wilcoxon p = {pv}")
        top = h.dropna(subset=["rank"]).query("rank == 1").groupby("system").size()
        print(f"  top-readability picks: {top.to_dict()}")

    SRCJ = {"B1": "reports_main_glm_n100_B1B2B8_v3", "B2": "reports_main_glm_n100_B1B2B8_v3",
            "B5": "reports_main_glm_n100", "B8": "reports_main_glm_n100_B1B2B8_v3"}
    for tag, stem in (("DeepSeek (primary)", None), ("Kimi (fourth judge)",
                                                      "reports_main_glm_n100_kimi")):
        if stem is None:
            frames = [pd.read_parquet(C.RUNS_DIR / f"{SRCJ[m]}_l3.parquet")
                      .query("method == @m").assign(instance_id=lambda d: d["instance_id"].astype(str))
                      for m in SRCJ]
            j = pd.concat(frames).set_index(["method", "instance_id"])
        else:
            j = load_judge(stem, SRCJ)
            if j is None:
                continue
        m = h.copy()
        m[["j_" + k for k in DIMS]] = np.nan
        m["j_L3"] = np.nan
        for _, r in m.iterrows():
            rec = j.loc[(r["system"], r["instance_id"])] if (r["system"], r["instance_id"]) in j.index else None
            if rec is not None:
                for k in DIMS:
                    m.loc[_, "j_" + k] = rec[DIMS[k]]
                m.loc[_, "j_L3"] = rec["L3"]
        print(f"\n=== human vs {tag} (per-instance Pearson, pooled reports) ===")
        for k in DIMS:
            sub = m[["rater", k, "j_" + k]].dropna()
            if len(sub) > 2 and sub[k].nunique() > 1:
                r_, p_ = stats.pearsonr(sub[k], sub["j_" + k])
                print(f"  {DIMS[k]:14s} r = {r_:.2f} (p = {p_:.1e}, n = {len(sub)})")
        sub = m[["l3h", "j_L3"]].dropna()
        if len(sub) > 2 and sub["l3h"].nunique() > 1:
            r_, p_ = stats.pearsonr(sub["l3h"], sub["j_L3"])
            print(f"  {'L3 composite':14s} r = {r_:.2f} (p = {p_:.1e}, n = {len(sub)})")
        hs = h.groupby("system")["l3h"].mean()
        js = j.reset_index().groupby("method")["L3"].mean().reindex(hs.index)
        rho, pv = stats.spearmanr(hs.values, js.values)
        print(f"  Spearman over system means: rho = {rho:.3f} (p = {pv:.1e})")

    if h["rater"].nunique() > 1:
        print("\n=== inter-rater agreement ===")
        piv = h.pivot_table(index=["instance_id", "system"], columns="rater", values="l3h")
        corrs = [stats.pearsonr(piv[a], piv[b])[0]
                 for a in piv.columns for b in piv.columns if a < b]
        print(f"  mean pairwise Pearson on L3-human: {np.mean(corrs):.2f}")
        top_pick = h.dropna(subset=["rank"]).query("rank == 1")
        if len(top_pick):
            mat = top_pick.pivot_table(index="instance_id", columns="system",
                                       values="rater", aggfunc="count").fillna(0)
            print(f"  top-readability pick counts per system:\n{mat.sum().to_string()}")

    h.to_csv(HV / "human_ratings_long.csv", index=False)
    g.to_csv(HV / "human_system_means.csv")
    print(f"\nsaved: {HV / 'human_ratings_long.csv'}, {HV / 'human_system_means.csv'}")


if __name__ == "__main__":
    main()
