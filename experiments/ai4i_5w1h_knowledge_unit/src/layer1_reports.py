"""Layer 1 programmatic fact-checking on GENERATED reports (paper Phase-4 L1).

Unlike layer1_eval.py (which validates the programmatic GT), this scores real
report TEXT from any system. It is fully programmatic (zero variance) and
applies uniformly to all 8 systems:

  * classification accuracy : did the report identify the correct failure status/mode?
  * numerical consistency   : do the numbers it states match the true values (+/-5%)?
  * rule consistency        : for each active deterministic mode, is a matching cause named?

L1 = 0.30*numerical + 0.30*rule + 0.40*classification  (in 0-1).
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

import config as C

_NUM = re.compile(r"\d+\.?\d*")
# mode -> detection keywords (failure-name forms only, to avoid collisions with
# normal discussion e.g. "tool-wear stage" on healthy samples)
_MODE_KW = {
    "TWF": ["twf", "tool wear failure", "tool-wear failure"],
    "HDF": ["hdf", "heat dissipation failure", "heat dissipation", "thermal failure"],
    "PWF": ["pwf", "power failure"],
    "OSF": ["osf", "overstrain failure", "over-strain failure"],
    "RNF": ["rnf", "random failure"],
}
_HEALTHY_KW = ["no failure", "no active", "no fault", "healthy", "normal operation",
               "operating normally", "fault-free", "no anomaly"]
_RULE_CAUSE_KW = {
    "PWF": ["power"],
    "HDF": ["heat", "thermal", "temperature", "cool"],
    "OSF": ["overstrain", "strain", "torque", "wear"],
    "TWF": ["wear"],
}
_KEY_VALUES = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
               "Torque [Nm]", "Tool wear [min]", "Power [W]"]


def _nums(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(text or "")]


def _mentioned_modes(text: str) -> set[str]:
    low = (text or "").lower()
    return {m for m, kws in _MODE_KW.items() if any(k in low for k in kws)}


def _classification(text, true_modes: set[str]) -> float:
    mentioned = _mentioned_modes(text)
    low = (text or "").lower()
    healthy_stated = any(k in low for k in _HEALTHY_KW)
    if not true_modes:  # truly healthy
        if mentioned:
            return 0.0  # wrongly asserts a failure mode
        return 1.0 if healthy_stated else 0.5  # silent vs explicit no-failure
    # truly faulted
    if true_modes & mentioned:
        return 1.0
    if mentioned:  # mentioned a WRONG mode
        return 0.3
    return 0.0  # missed the fault entirely


_CONSTS = {200.0, 240.0, 1380.0, 3500.0, 9000.0, 8.6, 11000.0, 12000.0, 13000.0}


def _numerical(text, row) -> float:
    """Precision of stated numbers: of the sensor-magnitude numbers in the text,
    how many match a true value (or a documented engineering constant) within
    +/-5%? Stating no numbers (or only correct ones) scores 1.0; invented/wrong
    numbers lower it. This is the hallucination angle, not recall."""
    nums = [n for n in _nums(text) if 3.0 <= n <= 15000.0]  # plausible sensor band
    if not nums:
        return 1.0
    truth = [row.get(c) for c in _KEY_VALUES if row.get(c)] + list(_CONSTS)
    hit = sum(1 for n in nums if any(abs(n - v) <= 0.05 * abs(v) for v in truth))
    return hit / len(nums)


def _rule(text, true_modes: set[str]) -> float:
    det = {m for m in ["PWF", "HDF", "OSF", "TWF"] if m in true_modes}
    if not det:
        return 1.0  # no deterministic rule to satisfy (healthy / RNF-only)
    low = (text or "").lower()
    ok = sum(1 for m in det if any(k in low for k in _RULE_CAUSE_KW[m]))
    return ok / len(det)


def score_report(text, row) -> dict:
    true_modes = {m for m in C.FAILURE_MODES if row.get(m) == 1}
    cls = _classification(text, true_modes)
    num = _numerical(text, row)
    rule = _rule(text, true_modes)
    return {"classification": cls, "numerical": num, "rule": rule,
            "L1": round(0.30 * num + 0.30 * rule + 0.40 * cls, 3),
            "true_modes": "+".join(sorted(true_modes)) or "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", help="reports parquet under runs/")
    args = ap.parse_args()
    rep = pd.read_parquet(C.RUNS_DIR / args.reports)
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet").set_index("sample_id")

    rows = []
    for _, r in rep.iterrows():
        s = sample.loc[r["instance_id"]]
        sc = score_report(r["raw_output"], s.to_dict())
        sc.update({"method": r["method"], "instance_id": r["instance_id"],
                   "machine_failure": int(r["machine_failure"])})
        rows.append(sc)
    df = pd.DataFrame(rows)
    out = C.RUNS_DIR / args.reports.replace(".parquet", "_l1.parquet")
    df.to_parquet(out, index=False)

    print(f"=== Layer 1 on {args.reports} (n={len(df)}) ===")
    g = df.groupby("method").agg(L1=("L1", "mean"), cls=("classification", "mean"),
                                 num=("numerical", "mean"), rule=("rule", "mean")).round(3)
    g = g.reindex([m for m in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"] if m in g.index])
    print(g.to_string())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
