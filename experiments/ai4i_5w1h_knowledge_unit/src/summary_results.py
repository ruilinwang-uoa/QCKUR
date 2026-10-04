"""Combine the first-batch Layer-3 scores into the paper's 8-system comparison.

For each model (glm-4-flash primary, deepseek-v4-flash cross-LLM) it assembles
the full 8-system table from:
  * the main L3 file (B1/B2/B3/B6/B7/B8, generated before the B4/B5 fix)
  * the v2 B4/B5 L3 file (fair B4/B5, no 5W1H leakage)
Within each model every system is judged by the SAME judge model (so the
8-system comparison is internally consistent). Writes:
  * runs/first_batch_summary.csv
  * analysis/first_batch_results.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config as C

MODELS = {
    "glm-4-flash": {  # primary
        "main": "reports_n30_glm_l3.parquet",
        "b4b5_v2": "reports_n30_glm_B4B5_v2_l3.parquet",
        "judge": "deepseek-v4-flash",
    },
    "deepseek-v4-flash": {  # cross-LLM
        "main": "reports_n30_deepseek_l3.parquet",
        "b4b5_v2": "reports_n30_deepseek_B4B5_v2_l3.parquet",
        "judge": "glm-4-flash",
    },
}
ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
NAMES = {"B1": "Template-only", "B2": "Traditional D2T", "B3": "Direct LLM",
         "B4": "Few-shot LLM", "B5": "Tool agent", "B6": "RAG", "B7": "RAG+CoVe",
         "B8": "Full framework"}


def _cov6(s):
    cov = s.get("coverage")
    if isinstance(cov, dict):
        return sum(int(v) for v in cov.values())
    if isinstance(cov, str):
        try:
            return sum(int(v) for v in json.loads(cov).values())
        except Exception:
            return 0
    return 0


def _load(path):
    df = pd.read_parquet(C.RUNS_DIR / path)
    df["cov6"] = df.apply(_cov6, axis=1)
    return df


def build_table(spec) -> pd.DataFrame:
    main = _load(spec["main"])
    v2 = _load(spec["b4b5_v2"])
    keep = main[~main["method"].isin(["B4", "B5"])]
    combined = pd.concat([keep, v2], ignore_index=True)
    g = combined.groupby("method").agg(
        n=("L3", "size"), L3=("L3", "mean"), faith=("faithfulness", "mean"),
        compl=("completeness", "mean"), coh=("coherence", "mean"),
        act=("actionability", "mean"), cov6=("cov6", "mean")).round(2)
    g = g.reindex([m for m in ORDER if m in g.index])
    return g


def main() -> None:
    tables = {m: build_table(s) for m, s in MODELS.items()}
    # combined CSV
    rows = []
    for m, t in tables.items():
        for sid, r in t.iterrows():
            rows.append({"model": m, "system": sid, "name": NAMES[sid],
                         "judge": MODELS[m]["judge"], **r.to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(C.RUNS_DIR / "first_batch_summary.csv", index=False)

    # markdown
    L = ["# AI4I first-batch results (n=30 instances/system, seed 42)\n"]
    L.append("Layer-3 LLM-judge scores. Within each model column all systems are judged "
             "by the same judge (fair within-model comparison). L3 in 0-5; coverage in 0-6.\n")
    L.append("Each model judges the OTHER model's reports (cross-judge, avoids self-bias).\n")
    for m, t in tables.items():
        L.append(f"\n## {m}  (judge: {MODELS[m]['judge']})\n")
        t2 = t.copy()
        t2.insert(0, "system", [NAMES[s] for s in t2.index])
        t2.insert(1, "id", t2.index)
        L.append(t2.to_markdown(index=False) + "\n")
        b8 = t.loc["B8", "L3"] if "B8" in t.index else float("nan")
        L.append(f"\n**B8 framework L3 = {b8}**. Advantage vs baselines:\n")
        for sid in ["B3", "B6", "B7", "B4", "B5", "B2"]:
            if sid in t.index:
                d = b8 - t.loc[sid, "L3"]
                L.append(f"- vs {sid} ({NAMES[sid]}): **+{d:.2f}**\n")

    # cross-LLM consistency of B8 advantage
    L.append("\n## Cross-LLM summary\n")
    L.append("| Model | B8 L3 | best non-framework | B8 advantage |\n|---|---|---|---|\n")
    for m, t in tables.items():
        b8 = t.loc["B8", "L3"] if "B8" in t.index else float("nan")
        non = t.drop("B8", errors="ignore")["L3"]
        best = non.max() if len(non) else float("nan")
        L.append(f"| {m} | {b8} | {best:.2f} | +{b8-best:.2f} |\n")
    L.append("\nNote: primary = glm-4-flash (non-reasoning); cross-LLM = deepseek-v4-flash "
             "(reasoning). Cross-model absolute L3 values use different judges and are not "
             "directly comparable; compare the within-model B8-vs-baseline *pattern*.\n")

    (C.ANALYSIS_DIR / "first_batch_results.md").write_text("".join(L), encoding="utf-8")
    print(out.to_string(index=False))
    print("\nWrote runs/first_batch_summary.csv and analysis/first_batch_results.md")


if __name__ == "__main__":
    main()
