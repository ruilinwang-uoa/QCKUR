"""De-circularization analyses (Tier 0, zero API cost) for the ESWA paper.

A. L1 numeric-matcher audit + repaired scorer (L1')
   The original `_numerical` compares every number in [3, 15000] against the
   row's 5 raw sensors + Power + 9 engineering constants.  The audit classifies
   every flagged mismatch into: derived feature (Overstrain / Temp diff missing
   from the truth set), dataset-level statistic, unit conversion (kW<->W,
   rad/s<->rpm), identifier (UDI / sample id), list marker, or genuine miss.
   The repaired scorer extends the truth set, strips identifiers and list
   markers before extraction, and accepts unit conversions.  Applied uniformly
   to all systems.

   Data sources mirror composite_final.py: B1/B2/B8 from the rule-engine v3
   run, B7 from B7_v2, B3-B6 from the main n=100 run.

B. L3 without the 5W1H-coverage dimension (L3' = mean of 4 remaining dims),
   Final recomputed under original / repaired scorers.

C. Composite-weight sensitivity grid for the B8-vs-B2 ordering.

Outputs: analysis/decirc_summary.md + console tables.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data"

MAIN = "reports_main_glm_n100"
B7V2 = "reports_main_glm_n100_B7_v2"
B1B2B8V3 = "reports_main_glm_n100_B1B2B8_v3"
SRC = {s: MAIN for s in ["B3", "B4", "B5", "B6"]}
SRC["B7"] = B7V2
for s in ["B1", "B2", "B8"]:
    SRC[s] = B1B2B8V3
SYSTEMS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
NAMES = {"B1": "Template-only", "B2": "Traditional D2T", "B3": "Direct LLM",
         "B4": "Few-shot", "B5": "Tool agent", "B6": "RAG", "B7": "RAG+CoVe",
         "B8": "Framework"}

_NUM = re.compile(r"\d+\.?\d*")
_ID_PATTERNS = [
    re.compile(r"UDI\s*:?\s*\d+", re.I),
    re.compile(r"[Ss]ample\s*:?\s*s?\d{3,5}"),
    re.compile(r"\bs\d{4}\b", re.I),
    re.compile(r"\b(?:ID|id)\s*:?\s*\d{3,5}\b"),
]
# numbered-list markers ("1. Fault", "**2. Cause", "3)")
_MARKER = re.compile(r"(?m)(?:^|(?<=[\s\*\(\"']))\d{1,2}[.\)](?=\s|[\*:])")
_CONSTS = [200.0, 240.0, 1380.0, 3500.0, 9000.0, 8.6, 11000.0, 12000.0, 13000.0]
_KEY = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "Tool wear [min]", "Power [W]"]
_DERIVED = ["Temp diff [K]", "Overstrain [min*Nm]"]


def clean(text: str) -> str:
    t = text or ""
    for p in _ID_PATTERNS:
        t = p.sub(" ", t)
    t = _MARKER.sub(" ", t)
    # percentages are not sensor-magnitude claims (e.g. "confidence 91.62%")
    t = re.sub(r"\b\d{1,3}\.?\d*\s?%", " ", t)
    return t


_STAT_CTX = re.compile(
    r"(average|mean\b|dataset|typical|overall|standard deviation|\bstd\b|"
    r"distribution|population|imbalance|failure rate|percent|%)", re.I)


def nums_with_ctx(text: str) -> list[tuple[float, bool]]:
    """Numbers of the cleaned text with a flag: does a statistical-context
    keyword appear within +-60 characters of the number?"""
    out = []
    for m in _NUM.finditer(text):
        v = float(m.group())
        if not (3.0 <= v <= 15000.0):
            continue
        lo, hi = max(0, m.start() - 60), min(len(text), m.end() + 60)
        out.append((v, bool(_STAT_CTX.search(text[lo:hi]))))
    return out


def nums_of(text: str, repaired: bool) -> list[float]:
    t = clean(text) if repaired else (text or "")
    return [float(x) for x in _NUM.findall(t) if 3.0 <= float(x) <= 15000.0]


def close(a: float, b: float, tol: float = 0.05) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def unit_hit(n: float, truth: list[float]) -> bool:
    for v in truth:
        if close(n * 1000.0, v) or close(n, v * 1000.0):
            return True
        if 50.0 <= v <= 4000.0 and close(n * 2 * np.pi / 60.0, v):
            return True
    return False


def main() -> None:
    sample = pd.read_parquet(DATA / "sample_315.parquet").set_index("sample_id")
    full = pd.read_parquet(DATA / "ai4i_full_derived.parquet")

    stat_vals = []
    for c in _KEY + _DERIVED:
        stat_vals += [full[c].mean(), full[c].std(), full[c].min(), full[c].max()]
    stat_vals += [3.39, 10000.0]
    stat_vals = [v for v in stat_vals if v and 3.0 <= v <= 15000.0]

    rep_cache = {src: pd.read_parquet(RUNS / f"{src}.parquet") for src in set(SRC.values())}

    # ---------- A. per-system audit + repaired numeric score ----------
    rows, b8_counts = [], {}
    b8_miss = Counter()
    b8_examples = []
    per_rep_num = {}   # (system, instance) -> repaired numeric score
    for s in SYSTEMS:
        sub = rep_cache[SRC[s]]
        sub = sub[sub["method"] == s]
        counts = {"orig_hit": 0, "derived": 0, "dataset_stat": 0, "unit": 0,
                  "identifier_or_marker": 0, "genuine": 0}
        nums_orig_all, nums_rep_all = [], []
        for _, r in sub.iterrows():
            row = sample.loc[r["instance_id"]].to_dict()
            orig_truth = [row.get(c) for c in _KEY if row.get(c)] + _CONSTS
            n_orig = nums_of(r["raw_output"], repaired=False)
            ctx_nums = nums_with_ctx(clean(r["raw_output"]))
            n_rep = [v for v, _ in ctx_nums]
            # numbers removed by identifier/marker stripping that the original
            # scorer counted as misses
            orig_misses = sum(1 for n in n_orig
                              if not any(close(n, v) for v in orig_truth))
            removed_misses = orig_misses - sum(1 for n in n_rep
                                               if not any(close(n, v) for v in orig_truth))
            counts["identifier_or_marker"] += max(0, removed_misses)
            nums_orig_all.append(np.mean([any(close(n, v) for v in orig_truth)
                                          for n in n_orig]) if n_orig else 1.0)
            hits = []
            for n, stat_ctx in ctx_nums:
                if any(close(n, v) for v in orig_truth):
                    hits.append(True); counts["orig_hit"] += 1
                elif any(close(n, v) for v in [row.get(c) for c in _DERIVED if row.get(c)]):
                    hits.append(True); counts["derived"] += 1
                    if s == "B8" and len(b8_examples) < 6:
                        b8_examples.append(
                            f"derived-feature: {n:g} (Overstrain={row.get('Overstrain [min*Nm]', 0):.0f}, dT={row.get('Temp diff [K]', 0):.1f})")
                elif stat_ctx and any(close(n, v) for v in stat_vals):
                    hits.append(True); counts["dataset_stat"] += 1
                    if s == "B8" and len(b8_examples) < 6:
                        b8_examples.append(f"dataset-stat (with context): {n:g}")
                elif unit_hit(n, orig_truth):
                    hits.append(True); counts["unit"] += 1
                else:
                    hits.append(False); counts["genuine"] += 1
                    if s == "B8":
                        b8_miss[n] += 1
            nums_rep_all.append(np.mean(hits) if hits else 1.0)
            per_rep_num[(s, r["instance_id"])] = nums_rep_all[-1]
        if s == "B8":
            b8_counts = counts
        rows.append({"sys": s, "name": NAMES[s],
                     "num_orig": round(float(np.mean(nums_orig_all)), 3),
                     "num_rep": round(float(np.mean(nums_rep_all)), 3)})
    audit = pd.DataFrame(rows)

    # ---------- B. layer assembly from the honest sources ----------
    def layer(col):
        frames = []
        for s in SYSTEMS:
            d = pd.read_parquet(RUNS / f"{SRC[s]}_{col}.parquet")
            d = d[d["method"] == s].copy()
            frames.append(d)
        return pd.concat(frames, ignore_index=True)

    l1 = layer("l1")
    l2 = layer("l2")[["method", "instance_id", "L2"]]
    l3 = layer("l3")[["method", "instance_id", "L3", "faithfulness",
                      "completeness", "coherence", "actionability"]]
    inst = (l1.merge(l2, on=["method", "instance_id"])
              .merge(l3, on=["method", "instance_id"]))
    # original L3 = 0.30*f + 0.15*c + 0.25*cov5 + 0.15*coh + 0.15*act;
    # de-circularized L3' drops coverage and renormalizes the rest (sum 0.75)
    inst["L3_nocov"] = ((0.30 * inst["faithfulness"] + 0.15 * inst["completeness"]
                         + 0.15 * inst["coherence"] + 0.15 * inst["actionability"])
                        / 0.75)
    inst["num_rep"] = [per_rep_num.get((m, i), np.nan)
                       for m, i in zip(inst["method"], inst["instance_id"])]
    # repaired L1 = 0.30*num_rep + 0.30*rule + 0.40*cls (rule/cls unchanged)
    inst["L1_rep"] = 0.30 * inst["num_rep"] + 0.30 * inst["rule"] \
        + 0.40 * inst["classification"]
    # note: original L1 used the same decomposition, so L1_rep just swaps num
    agg = inst.groupby("method")[["L1", "L1_rep", "L2", "L3", "L3_nocov"]].mean()
    agg = agg.reindex(SYSTEMS)

    def final(L1c, L3c):
        return np.round(0.35 * 100 * agg[L1c] + 0.30 * 100 * agg["L2"]
                        + 0.35 * 20 * agg[L3c], 1)

    agg["Final_orig"] = final("L1", "L3")
    agg["Final_L1rep"] = final("L1_rep", "L3")
    agg["Final_L3nocov"] = final("L1", "L3_nocov")
    agg["Final_both"] = final("L1_rep", "L3_nocov")

    # ---------- C. weight grid ----------
    grid = []
    for w1, w2, w3 in [(0.35, 0.30, 0.35), (1/3, 1/3, 1/3), (0.50, 0.25, 0.25),
                       (0.25, 0.50, 0.25), (0.25, 0.25, 0.50), (0.60, 0.20, 0.20)]:
        f = {s: w1 * 100 * agg.loc[s, "L1_rep"] + w2 * 100 * agg.loc[s, "L2"]
             + w3 * 20 * agg.loc[s, "L3_nocov"] for s in SYSTEMS}
        grid.append({"w_L1": round(w1, 2), "w_L2": round(w2, 2), "w_L3": round(w3, 2),
                     "B8": round(f["B8"], 1), "B2": round(f["B2"], 1),
                     "B8-B2": round(f["B8"] - f["B2"], 1)})
    grid = pd.DataFrame(grid)

    # ---------- console + markdown ----------
    print("=== A. B8 numeric-mismatch audit ===")
    print(json.dumps(b8_counts, indent=2))
    print("\nB8 recovery examples:", *b8_examples, sep="\n  ")
    print("\nB8 remaining genuine-miss values (top 15 by frequency):")
    for v, c in b8_miss.most_common(15):
        print(f"   {v:g}  x{c}")
    print("\n=== A2. numeric scorer: original vs repaired ===")
    print(audit.to_string(index=False))
    print("\n=== B. Final under original / repaired L1 / L3-no-coverage ===")
    print(agg.round(3).to_string())
    print("\n=== C. Weight grid (repaired L1 + L3-no-cov) ===")
    print(grid.to_string(index=False))

    md = ["# De-circularization analyses (Tier 0)", "",
          "## A. B8 numeric-mismatch audit", "", "```json",
          json.dumps(b8_counts, indent=2), "```", "",
          "Recovery examples: " + "; ".join(b8_examples), "",
          "Remaining genuine-miss values: "
          + ", ".join(f"{v:g}(x{c})" for v, c in b8_miss.most_common(15)), "",
          "## A2. Numeric scorer original vs repaired", "",
          "```", audit.to_string(index=False), "```", "",
          "## B. Final variants", "",
          "```", agg.round(3).to_string(), "```", "",
          "## C. Weight grid", "", "```", grid.to_string(index=False), "```", ""]
    (ROOT / "analysis" / "decirc_summary.md").write_text("\n".join(md),
                                                         encoding="utf-8")
    print("\nSaved: analysis/decirc_summary.md")


if __name__ == "__main__":
    main()
