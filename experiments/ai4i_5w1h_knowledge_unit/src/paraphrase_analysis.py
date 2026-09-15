"""Paired analysis: paraphrase conditions (B8PB / B8PA) vs canonical B8.

Compares Final / L1 / L2 / L3 / 5W1H coverage per instance, reports paired
Wilcoxon tests, and relates the observed degradation to the Proposition-1
paraphrase bound (~17% per-question ungrounded-claim probability).
"""
from __future__ import annotations

import json

import pandas as pd
from scipy import stats

import config as C

W = {"L1": 0.35, "L2": 0.30, "L3": 0.35}


def cov6(c) -> int:
    return sum(c.values()) if isinstance(c, dict) else 0


def load_scores(stem: str, method: str) -> pd.DataFrame | None:
    try:
        l1 = pd.read_parquet(C.RUNS_DIR / f"{stem}_l1.parquet").query("method == @method")[["instance_id", "L1"]]
        l2 = pd.read_parquet(C.RUNS_DIR / f"{stem}_l2.parquet").query("method == @method")[["instance_id", "L2", "consistency", "contradiction"]]
        l3 = pd.read_parquet(C.RUNS_DIR / f"{stem}_l3.parquet").query("method == @method")[["instance_id", "L3", "coverage"]]
        d = l1.merge(l2, on="instance_id").merge(l3, on="instance_id")
        d["Final"] = W["L1"] * d["L1"] * 100 + W["L2"] * d["L2"] * 100 + W["L3"] * d["L3"] * 20
        d["cov6"] = d["coverage"].apply(cov6)
        return d.set_index("instance_id")
    except FileNotFoundError:
        return None


st = json.loads((C.RUNS_DIR / "paraphrase_state.json").read_text(encoding="utf-8"))
print("=== matcher routing (exact-text rule) ===")
for key, r in st["routing"].items():
    print(f"  {key:28s} -> {r['verdict']:44s} out: {r['matcher_output'][:60]!r}")

base = load_scores("reports_main_glm_n100_B1B2B8_v3", "B8")
rows = []
for cond, method in (("bank paraphrases", "B8PB"), ("adversarial paraphrases", "B8PA")):
    alt = load_scores("reports_paraphrase_" + ("bank" if method == "B8PB" else "adv") + "_B8", method)
    if alt is None:
        print(f"\n{cond}: scores not ready"); continue
    m = base.join(alt, how="inner", lsuffix="_canon", rsuffix="_para")
    print(f"\n=== {cond} (n = {len(m)} paired instances) ===")
    for col in ("Final", "L1", "L2", "L3", "cov6"):
        d = m[f"{col}_canon"] - m[f"{col}_para"]
        line = (f"  {col:6s} canonical {m[f'{col}_canon'].mean():7.2f} -> "
                f"paraphrase {m[f'{col}_para'].mean():7.2f}  "
                f"(drop {d.mean():+.2f})")
        if col in ("Final", "L3", "cov6") and not (d == 0).all():
            wtest = stats.wilcoxon(d, zero_method="wilcox", method="auto")
            line += f"  p = {wtest.pvalue:.1e}"
        print(line)
        rows.append({"condition": cond, "metric": col,
                     "canonical": round(m[f"{col}_canon"].mean(), 3),
                     "paraphrase": round(m[f"{col}_para"].mean(), 3),
                     "drop": round(d.mean(), 3)})
    contra = m["contradiction_para"].mean()
    print(f"  Layer-2 contradiction fraction under paraphrase: {contra:.3f} "
          f"(canonical B8: {m['contradiction_canon'].mean() if 'contradiction_canon' in m else float('nan'):.3f})")

pd.DataFrame(rows).to_csv(C.ANALYSIS_DIR / "paraphrase_e2e_summary.csv", index=False)
print(f"\nsaved: {C.ANALYSIS_DIR / 'paraphrase_e2e_summary.csv'}")
