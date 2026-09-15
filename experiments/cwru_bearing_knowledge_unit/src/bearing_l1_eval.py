"""Layer-1 evaluation for the bearing campaign (Phase 3).

Three check families, mirroring the AI4I Layer-1 structure but built on the
REPAIRED numeric rules from day one (identifiers/list markers/percentages
stripped before number extraction; unit conversions accepted) so the
artifact trap documented for the original AI4I scorer cannot recur here:

  numeric        every number in the report must be consistent with the
                 instance's truth set (rpm, prominences, thresholds, band,
                 kurtosis, characteristic multiples/frequencies, defect
                 sizes, sampling rate, kmax) within 5% relative tolerance
                 (or exact for small constants); score = consistent/extracted
  classification  the fault-component claim parsed from the text vs the
                 manifest ground truth (normal/IR/B/OR)
  rule            the same parsed claim vs the frozen v3 two-tier verdict
                 (did the system's stated diagnosis match the deployed
                 handler, independent of the label)

Output: code/runs/bearing_l1.csv (per system x instance) + console summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from bearing_knowledge import SPEC, load_v3

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SYSTEMS = ["B1", "B2", "B3", "B4", "B5", "B8"]
TOL_REL = 0.05

COMP_PATTERNS = [
    ("IR", r"inner[- ]race|\bIR\b"),
    ("OR", r"outer[- ]race|\bOR\b"),
    ("B", r"rolling[- ]element|\bball\b|\bBSF\b"),
    ("normal", r"no fault|healthy|normal\b|no component fault|no defect"),
]


def truth_set(rec: pd.Series, thresholds: dict) -> list[float]:
    M = SPEC["defect_freq_multiples_x_fr"]
    fr = rec.rpm / 60.0
    vals = [rec.rpm, fr, 12000, 12, 10,
            rec.BPFO, rec.BPFI, rec.BSF, rec.BALL, rec.FTF, rec.BALLFAM,
            rec.kurt_band, rec.kurt_band_lo,
            thresholds["tau_ir"], thresholds["tau_or"],
            rec.rms if "rms" in rec else None,
            rec.defect_in * 1000 if rec.defect_in else None,
            rec.defect_in * 25.4 if rec.defect_in else None]
    vals += [rec.defect_in] if rec.defect_in else []
    vals += [M[k] for k in ("BPFO", "BPFI", "BSF", "ball_defect_2xBSF", "FTF")]
    vals += [M[k] * fr for k in
             ("BPFO", "BPFI", "BSF", "ball_defect_2xBSF", "FTF")]
    vals += [8, 0.5, 2.4, 6, 9, 39.0, 7.94, 3.134, 25.0, 52.0, 15.0]
    # legitimate template-derived dominance/margin ratios (h_why_verdict,
    # h_margin): each component vs the strongest competing component
    comps = {"BPFO": rec.BPFO, "BPFI": rec.BPFI, "BALLFAM": rec.BALLFAM}
    for k, v in comps.items():
        others = [x for kk, x in comps.items() if kk != k]
        vals += [v / max(others), v / min(others)]
    return [v for v in vals if v is not None]


def extract_numbers(text: str) -> list[float]:
    """Repaired extraction: strip list markers / file ids / percentages."""
    t = re.sub(r"\b\d+\.mat\b", " ", text)
    t = re.sub(r"\b620[35]-?2RS\b", " ", t)        # bearing model designators
    t = re.sub(r"(?m)^\s*\d+[\.\)]\s", " ", t)      # list markers
    t = re.sub(r"(?<![\w.])\d+\s*\.\s*(?=\s[A-Z(])", " ", t)  # dangling "3."
    out = []
    for m in re.finditer(r"\d+(?:\.\d+)?", t):
        s = m.group(0)
        # skip bare small ordinals like "1)" already stripped; keep values
        try:
            out.append(float(s))
        except ValueError:  # pragma: no cover
            continue
    return out


def consistent(v: float, truths: list[float]) -> bool:
    for t in truths:
        if t == 0:
            if abs(v) < 1e-9:
                return True
        elif abs(v - t) <= TOL_REL * max(abs(t), 1e-9) or v == round(t):
            return True
        # 5% relative or exact rounding of a truth value
        for rt in (round(t), round(t, 1)):
            if v == rt:
                return True
    return False


def parse_component(text: str) -> str | None:
    low = text.lower()
    hits = []
    for comp, pat in COMP_PATTERNS:
        if re.search(pat, low):
            hits.append(comp)
    # a report that says "no fault" AND names a component: prefer explicit
    # no-fault phrasing only when no specific component is asserted
    if "normal" in hits and len(hits) > 1:
        hits = [h for h in hits if h != "normal"] or ["normal"]
    if not hits:
        return None
    return hits[0]


def main() -> None:
    df, thresholds = load_v3()
    gt = json.load(open(RUNS / "bearing_gt.json", encoding="utf-8"))
    rows = []
    for sys_id in SYSTEMS:
        p = RUNS / f"bearing_reports_{sys_id}.parquet"
        if not p.exists():
            print(f"{sys_id}: no reports yet, skipping")
            continue
        rep = pd.read_parquet(p)
        for _, r in rep.iterrows():
            rec = df[df.file == r.instance_id].iloc[0]
            truths = truth_set(rec, thresholds)
            nums = extract_numbers(str(r.text))
            good = [v for v in nums if consistent(v, truths)]
            num_score = (len(good) / len(nums)) if nums else 1.0
            claimed = parse_component(str(r.text))
            cls = 1.0 if claimed == rec.truth else (
                0.0 if claimed is not None else None)
            rule = 1.0 if claimed == rec.pred else (
                0.0 if claimed is not None else None)
            rows.append({"system": sys_id, "instance_id": r.instance_id,
                         "truth": rec.truth, "v3_pred": rec.pred,
                         "claimed": claimed,
                         "n_numbers": len(nums), "n_bad": len(nums) - len(good),
                         "numeric": num_score,
                         "classification": cls, "rule": rule,
                         "ok": r.get("ok", True)})
    out = pd.DataFrame(rows)
    out.to_csv(RUNS / "bearing_l1.csv", index=False)
    print(f"per-instance rows -> {RUNS / 'bearing_l1.csv'}")
    summ = out.groupby("system").agg(
        n=("instance_id", "count"),
        numeric=("numeric", "mean"),
        classification=("classification", "mean"),
        rule=("rule", "mean"),
        bad_numbers=("n_bad", "sum"),
        parse_fail=("claimed", lambda s: s.isna().sum()))
    print("\nLayer-1 summary (bearing):")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
