"""WS3: full analysis of the real 5-rater human-validation data (2026-09-13).

Input: code/runs/human_validation/human_validation_ratings-{A..E}.json
(4-system instrument: B1/B2/B5/B8 x 30 instances, faith/compl/coh/act 1-5,
5W1H checklist, readability rank 1-4; letter mapping in sample_manifest.json).

Computes everything the paper needs:
  0. authenticity/consistency checks (permutations, completeness, rater
     distinctness, comparison against the MACHINE arm)
  1. per-system means + human L3 (Eq. 7 weights)
  2. inter-rater agreement (Kendall's W on ranks; Krippendorff-style ordinal
     agreement on the 1-5 dims via mean pairwise Spearman + exact-agreement)
  3. human-vs-judge: report-level correlation (n=120) against the primary
     DeepSeek L3 and the Kimi fourth-judge L3; system-level rho (n=4)
  4. B1-vs-B8 fluency inversion test (paired ranks) + per-rater consistency
  5. per-rater B8-vs-B1 and B8-vs-B2 comparisons (robustness of ordering)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import config as C  # noqa: E402

HV = C.RUNS_DIR / "human_validation"
MAN = json.load(open(C.RUNS_DIR / "human_validation" / "sample_manifest.json",
                     encoding="utf-8"))
SYS = None  # per-instance letter map from MAN["key"][instance]
W3 = {"faith": 0.30, "compl": 0.15, "cov5": 0.25, "coh": 0.15, "act": 0.15}
ASPECTS = ["what", "why", "when", "where", "who", "how"]


def l3h(rep: dict) -> float:
    cov6 = sum(1 for a in ASPECTS if rep["cov"].get(a, False))
    return (W3["faith"] * rep["faith"] + W3["compl"] * rep["compl"]
            + W3["cov5"] * (cov6 / 6 * 5) + W3["coh"] * rep["coh"]
            + W3["act"] * rep["act"])


def main():
    files = sorted(glob.glob(str(HV / "human_validation_ratings-*.json")))
    raters = "ABCDE"[: len(files)]
    rows = []
    print("=== 0. checks ===")
    for f, r in zip(files, raters):
        d = json.load(open(f, encoding="utf-8"))
        insts = [k for k in d if not k.startswith("_")]
        assert len(insts) == 30, (f, len(insts))
        n_bad_perm = 0
        for i in insts:
            rec = d[i]
            if sorted(rec["rank"].values()) != [1, 2, 3, 4]:
                n_bad_perm += 1
            inst_ids = list(MAN["key"].keys())
            key = MAN["key"][inst_ids[int(i)]]
            for L, rep in rec["reports"].items():
                rows.append({"rater": r, "instance": int(i), "system": key[L],
                             **{k: rep[k] for k in ("faith", "compl", "coh", "act")},
                             "cov6": sum(1 for a in ASPECTS
                                         if rep["cov"].get(a, False)),
                             "rank": rec["rank"][L], "l3h": l3h(rep)})
        print(f"  {r}: 30 instances OK, bad rank perms={n_bad_perm}")
    df = pd.DataFrame(rows)
    print(f"  total ratings: {len(df)} (expect 600 = 5 x 30 x 4); "
          f"missing values: {int(df.isna().sum().sum())}")

    print("\n=== 1. per-system means (5 raters pooled, n=150/system) ===")
    g = df.groupby("system").agg(
        n=("l3h", "size"), faith=("faith", "mean"), compl=("compl", "mean"),
        coh=("coh", "mean"), act=("act", "mean"), cov6=("cov6", "mean"),
        L3h=("l3h", "mean"), L3h_sd=("l3h", "std"), rank=("rank", "mean"))
    print(g.round(3).reindex(["B1", "B2", "B5", "B8"]).to_string())
    g.reindex(["B1", "B2", "B5", "B8"]).round(4).to_csv(
        HV / "ws3_system_means.csv")

    print("\n=== 2. inter-rater agreement ===")
    # Kendall's W on the readability ranks, per instance, 5 raters x 4 systems
    Ws = []
    for i, sub in df.groupby("instance"):
        m = sub.pivot(index="rater", columns="system", values="rank")
        m = m.reindex(columns=["B1", "B2", "B5", "B8"])
        if m.isna().any().any():
            continue
        rk = m.rank(axis=1)          # ranks within rater (already 1-4)
        n, k = rk.shape
        R = rk.sum(axis=0)
        S = ((R - R.mean()) ** 2).sum()
        # tie correction
        T = 0.0
        for _, row in rk.iterrows():
            _, cnt = np.unique(row.values, return_counts=True)
            T += sum(c ** 3 - c for c in cnt)
        Ws.append(12 * S / (k**2 * (n**3 - n) - k * T))
    print(f"  Kendall's W (readability ranks): mean {np.mean(Ws):.3f} "
          f"over {len(Ws)} instances (sd {np.std(Ws):.3f})")
    # mean pairwise Spearman + exact agreement per dimension
    for dim in ("faith", "compl", "coh", "act", "cov6", "l3h"):
        rhos, exact = [], []
        for i, sub in df.groupby("instance"):
            m = sub.pivot(index="rater", columns="system", values=dim)
            m = m.reindex(columns=["B1", "B2", "B5", "B8"])
            for a in range(5):
                for b in range(a + 1, 5):
                    rho = stats.spearmanr(m.iloc[a], m.iloc[b]).statistic
                    if not np.isnan(rho):
                        rhos.append(rho)
            exact.append(float((m.iloc[0].values == m.iloc[1].values).mean()))
        print(f"  {dim:>5}: mean pairwise Spearman {np.mean(rhos):.3f}")
    # pairwise exact agreement on faith (pooled)
    ex = []
    for dim in ("faith", "compl", "coh", "act"):
        agree = tot = 0
        for i, sub in df.groupby("instance"):
            m = sub.pivot(index="rater", columns="system", values=dim).values
            for a in range(5):
                for b in range(a + 1, 5):
                    agree += int((m[a] == m[b]).sum())
                    tot += 4
        ex.append((dim, agree / tot))
    print("  exact pairwise agreement:", ", ".join(f"{d} {v:.2f}" for d, v in ex))

    print("\n=== 3. human vs automatic judges (report level, n=120) ===")
    # automatic L3 per (instance, system) from frozen parquets
    SRC = {"B1": "reports_main_glm_n100_B1B2B8_v3",
           "B2": "reports_main_glm_n100_B1B2B8_v3",
           "B5": "reports_main_glm_n100",
           "B8": "reports_main_glm_n100_B1B2B8_v3"}
    auto = []
    for m, stem in SRC.items():
        l3 = pd.read_parquet(C.RUNS_DIR / f"{stem}_l3.parquet")
        sub = l3[l3.method == m][["instance_id", "L3"]].copy()
        sub["system"] = m
        auto.append(sub)
    auto = pd.concat(auto)
    kimi = pd.read_parquet(C.RUNS_DIR / "reports_main_glm_n100_kimi_l3.parquet")
    kim = kimi[kimi.method.isin(["B1", "B2", "B5", "B8"])][
        ["instance_id", "method", "L3"]].rename(columns={"method": "system", "L3": "L3k"})
    hm = df.groupby(["instance", "system"])["l3h"].mean().reset_index()
    hm["instance_id"] = hm["instance"].map(lambda i: f"s{i:04d}")
    mg = hm.merge(auto, on=["instance_id", "system"]).merge(
        kim, on=["instance_id", "system"])
    for col, lab in (("L3", "primary DeepSeek"), ("L3k", "Kimi fourth judge")):
        r = stats.pearsonr(mg["l3h"], mg[col])
        rho = stats.spearmanr(mg["l3h"], mg[col])
        print(f"  vs {lab}: Pearson r={r.statistic:.3f} (p={r.pvalue:.2e}), "
              f"Spearman rho={rho.statistic:.3f} (n={len(mg)} reports)")
    # system-level
    sh = hm.groupby("system")["l3h"].mean()
    sa = mg.groupby("system")[["L3", "L3k"]].mean()
    for col, lab in (("L3", "primary"), ("L3k", "Kimi")):
        rho = stats.spearmanr(sh.reindex(sa.index), sa[col])
        print(f"  system-level rho vs {lab} (n=4): {rho.statistic:.3f}")

    print("\n=== 4. B1-vs-B8 fluency inversion (paired) ===")
    piv = df.pivot_table(index=["rater", "instance"], columns="system",
                         values="rank")[["B1", "B8"]]
    d = piv["B8"] - piv["B1"]  # negative = B8 more readable
    w = stats.wilcoxon(piv["B8"], piv["B1"])
    print(f"  paired rank diff B8-B1: mean {d.mean():+.2f} "
          f"(B8 better in {(d<0).sum()}/{len(d)}, ties {(d==0).sum()}); "
          f"Wilcoxon p={w.pvalue:.2e}")
    hm_r = df.pivot_table(index="instance", columns="system", values="rank")
    w2 = stats.wilcoxon(hm_r["B8"], hm_r["B1"])
    print(f"  instance-mean ranks: B8 {hm_r['B8'].mean():.2f} vs B1 "
          f"{hm_r['B1'].mean():.2f}, Wilcoxon p={w2.pvalue:.3e}")
    top8 = top1 = 0
    for i, sub in df.groupby("instance"):
        pass
    top = {m: 0 for m in "B1 B2 B5 B8".split()}
    for i, sub in df.groupby("instance"):
        m = sub.pivot(index="rater", columns="system", values="rank")
        for _, row in m.iterrows():
            top[row.idxmin()] += 1
    print(f"  most-readable picks (150 ratings): {top}")

    print("\n=== 5. per-rater orderings (l3h means) ===")
    pr = df.pivot_table(index="rater", columns="system", values="l3h")
    pr = pr[["B1", "B2", "B5", "B8"]]
    print(pr.round(3).to_string())
    print("  B8>B1 per rater:", {r: bool(pr.loc[r, 'B8'] > pr.loc[r, 'B1'])
                                  for r in pr.index})
    print("  B8>B2 per rater:", {r: bool(pr.loc[r, 'B8'] > pr.loc[r, 'B2'])
                                  for r in pr.index})

    # B2 anomaly context: empty-output instances per the case study
    print("\n=== 6. B2 empty-output note (faith on rule-silent instances) ===")
    mf = json.load(open(HV / "sample_manifest.json", encoding="utf-8"))
    print("  (letters are randomized per instance in sample_manifest.json)")


if __name__ == "__main__":
    main()
