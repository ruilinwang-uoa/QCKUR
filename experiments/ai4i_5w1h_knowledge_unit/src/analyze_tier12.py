"""Analyze the Tier 1 + Tier 2 outputs; writes analysis/tier12_summary.md.

  a. Two-judge agreement   : GLM-4-Flash (second judge) vs DeepSeek-V4-Flash
                             (primary judge) on the canonical 800 reports --
                             correlations per dimension, system-mean rank
                             agreement, and whether the B8-vs-baseline pattern
                             replicates under the second judge.
  b. OOC rejection test    : reads runs/ooc_test_results.json.
  c. Dense RAG (B6-dense)  : composite vs the TF-IDF B6 and vs B8, paired.
  d. B8 multi-seed         : Final per generation seed (42, 1, 7); stability of
                             the B8-vs-B2 ordering per seed (B2 is
                             deterministic, so its instance scores are fixed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MAIN = "reports_main_glm_n100"
SRC = {s: MAIN for s in ["B3", "B4", "B5", "B6"]}
SRC["B7"] = "reports_main_glm_n100_B7_v2"
for s in ["B1", "B2", "B8"]:
    SRC[s] = "reports_main_glm_n100_B1B2B8_v3"
ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
DIMS = ["faithfulness", "completeness", "coherence", "actionability"]


def _l3_from(src_parquet: str) -> pd.DataFrame:
    d = pd.read_parquet(RUNS / src_parquet)
    keep = ["method", "instance_id", "L3"] + DIMS
    return d[[c for c in keep if c in d.columns]].copy()


def assemble_primary_l3() -> pd.DataFrame:
    frames = []
    for s in ORDER:
        d = _l3_from(f"{SRC[s]}_l3.parquet")
        frames.append(d[d["method"] == s])
    return pd.concat(frames, ignore_index=True)


def final_inst(l1: float, l2: float, l3: float) -> float:
    return 0.35 * 100 * l1 + 0.30 * 100 * l2 + 0.35 * 20 * l3


def part_a(md: list[str]) -> None:
    glm_path = RUNS / "reports_main_glm_n100_canonical_l3.parquet"
    if not glm_path.exists():
        md += ["## a. Two-judge agreement: SKIPPED (output missing)", ""]
        return
    glm = pd.read_parquet(glm_path)
    ds = assemble_primary_l3()
    m = ds.merge(glm, on=["method", "instance_id"], suffixes=("_ds", "_glm"))
    md += ["## a. Two-judge agreement (DeepSeek primary vs GLM second judge, n=800)", ""]
    rows = []
    for col in ["L3"] + DIMS:
        c_ds, c_glm = f"{col}_ds", f"{col}_glm"
        if c_ds not in m or c_glm not in m:
            continue
        r, p = stats.pearsonr(m[c_ds], m[c_glm])
        rho, ps = stats.spearmanr(m[c_ds], m[c_glm])
        rows.append({"dim": col, "pearson_r": round(r, 3), "p": f"{p:.1e}",
                     "spearman_rho": round(rho, 3),
                     "mean_ds": round(m[c_ds].mean(), 3),
                     "mean_glm": round(m[c_glm].mean(), 3)})
    t = pd.DataFrame(rows)
    md += ["```", t.to_string(index=False), "```", ""]
    # system means + rank agreement
    g_ds = m.groupby("method")["L3_ds"].mean()
    g_glm = m.groupby("method")["L3_glm"].mean()
    comp = pd.DataFrame({"L3_deepseek": g_ds, "L3_glm": g_glm}).reindex(ORDER)
    rho, _ = stats.spearmanr(comp["L3_deepseek"], comp["L3_glm"])
    md += [f"System-mean rank correlation (Spearman): {rho:.3f}", "",
           "```", comp.round(3).to_string(), "```", ""]
    # does the B8-vs-baseline pattern replicate at the L3 level?
    rep = []
    for s in ["B2", "B5", "B3", "B4", "B6", "B7"]:
        d8 = m[m["method"] == "B8"].set_index("instance_id")
        db = m[m["method"] == s].set_index("instance_id")
        j = d8.join(db, lsuffix="_8", rsuffix="_b")
        try:
            w = stats.wilcoxon(j["L3_glm_8"], j["L3_glm_b"])
            rep.append({"baseline": s,
                        "d_L3_glm": round(j["L3_glm_8"].mean() - j["L3_glm_b"].mean(), 2),
                        "wilcoxon_p_glm": f"{w.pvalue:.1e}"})
        except ValueError:
            pass
    md += ["B8 vs baselines under the GLM judge (paired, n=100 each):", "",
           "```", pd.DataFrame(rep).to_string(index=False), "```", ""]


def part_b(md: list[str]) -> None:
    p = RUNS / "ooc_test_results.json"
    if not p.exists():
        md += ["## b. OOC rejection test: SKIPPED (output missing)", ""]
        return
    r = json.loads(p.read_text(encoding="utf-8"))
    md += ["## b. Out-of-coverage rejection test (matching mechanism)", "",
           f"in-coverage n={r['n_in_coverage']}, OOC n={r['n_ooc']}, "
           f"provider={r['provider']}, seed={r['seed']}", "",
           f"- positive matching accuracy (exact text): **{r['positive_accuracy']:.1%}**",
           f"- wrong bank question: {r['positive_wrong_match']:.1%}; "
           f"wrongly rejected: {r['positive_rejected']:.1%}; "
           f"invalid output: {r['positive_invalid_output']:.1%}",
           f"- OOC rejection rate: **{r['ooc_rejection_rate']:.1%}**",
           f"- OOC wrongly accepted: {r['ooc_wrongly_accepted']:.1%}; "
           f"invalid output: {r['ooc_invalid_output']:.1%}", ""]


def _scores(stem: str) -> pd.DataFrame | None:
    """per-instance L1/L2/L3 + Final for a runs/ stem (e.g. ..._B6dense)."""
    try:
        l1 = pd.read_parquet(RUNS / f"{stem}_l1.parquet")
        l2 = pd.read_parquet(RUNS / f"{stem}_l2.parquet")[["method", "instance_id", "L2"]]
        l3 = pd.read_parquet(RUNS / f"{stem}_l3.parquet")[["method", "instance_id", "L3"]]
        d = l1.merge(l2, on=["method", "instance_id"]).merge(
            l3, on=["method", "instance_id"])
    except FileNotFoundError:
        return None
    d["Final"] = d.apply(lambda r: final_inst(r["L1"], r["L2"], r["L3"]), axis=1)
    return d


def part_c(md: list[str]) -> None:
    dense = _scores("reports_main_glm_n100_B6dense")
    if dense is None:
        md += ["## c. Dense-retrieval RAG: SKIPPED (output missing)", ""]
        return
    tfidf = _scores("reports_main_glm_n100")
    b6_t = tfidf[tfidf["method"] == "B6"].set_index("instance_id")
    b8 = _scores(SRC["B8"])
    b8 = b8[b8["method"] == "B8"].set_index("instance_id")
    de = dense[dense["method"] == "B6"].set_index("instance_id")
    md += ["## c. Dense-retrieval RAG (B6-dense, all-MiniLM-L6-v2) vs TF-IDF B6", "",
           f"n={len(de)}", "",
           f"- B6 TF-IDF  : L1={b6_t['L1'].mean():.3f} L2={b6_t['L2'].mean():.3f} "
           f"L3={b6_t['L3'].mean():.3f} Final={b6_t['Final'].mean():.1f}",
           f"- B6 dense   : L1={de['L1'].mean():.3f} L2={de['L2'].mean():.3f} "
           f"L3={de['L3'].mean():.3f} Final={de['Final'].mean():.1f}",
           f"- B8 (v3)    : Final={b8['Final'].mean():.1f}", ""]
    j = b8.join(de, rsuffix="_dense")
    w = stats.wilcoxon(j["Final"], j["Final_dense"])
    md += [f"paired B8 vs B6-dense: dFinal={j['Final'].mean()-j['Final_dense'].mean():+.1f}, "
           f"Wilcoxon p={w.pvalue:.1e}", ""]


def part_d(md: list[str]) -> None:
    b2 = _scores(SRC["B2"])
    if b2 is None:
        md += ["## d. B8 multi-seed: SKIPPED (B2 scores missing)", ""]
        return
    b2 = b2[b2["method"] == "B2"].set_index("instance_id")["Final"].rename("b2")
    v3 = _scores(SRC["B8"])
    if v3 is None:
        md += ["## d. B8 multi-seed: SKIPPED (B8 primary scores missing)", ""]
        return
    v3f = v3[v3["method"] == "B8"].set_index("instance_id")["Final"]
    per_seed = {42: v3f}
    for sd in (1, 7):
        s = _scores(f"reports_main_glm_n100_B8_seed{sd}")
        if s is not None:
            per_seed[sd] = s.set_index("instance_id")["Final"]
    rows, means = [], []
    for sd, f in per_seed.items():
        rows.append({"seed": sd, "B8 Final mean": round(f.mean(), 1),
                     "sd": round(f.std(), 1)})
        means.append(f.mean())
    t = pd.DataFrame(rows)
    arr = np.array(means)
    md += ["## d. B8 multi-seed stability (n=100 instances per seed)", "",
           "```", t.to_string(index=False), "```", "",
           f"across-seed mean = {arr.mean():.1f}, sd = {arr.std(ddof=1):.2f}", ""]
    for sd, f in per_seed.items():
        j = pd.DataFrame({"b8": f}).join(b2).dropna()
        w = stats.wilcoxon(j["b8"], j["b2"])
        md += [f"seed {sd}: B8 {j['b8'].mean():.1f} vs B2 {j['b2'].mean():.1f} "
               f"(d={j['b8'].mean()-j['b2'].mean():+.1f}, Wilcoxon p={w.pvalue:.1e})"]
    md.append("")


def main() -> None:
    md = ["# Tier 1 + Tier 2 results", ""]
    part_a(md)
    part_b(md)
    part_c(md)
    part_d(md)
    out = ROOT / "analysis" / "tier12_summary.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
