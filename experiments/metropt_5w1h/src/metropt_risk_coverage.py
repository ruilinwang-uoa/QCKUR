"""
metropt_risk_coverage.py -- selective-prediction risk-coverage analysis
for the MetroPT campaign (novelty item 1: the conformal view).

Objects:
  * nonconformity score s(w) = robust z_max (score tier); rule-fired
    windows carry s = +inf (confident detection outputs)
  * acceptance at threshold t: accept iff rule fired OR z <= t;
    otherwise ABSTAIN (the handler's selective-prediction knob)
  * HANDLER risk on accepted = P(verdict contradicts GT visibility)
  * REPORT risk on accepted  = L1 failure rate of the system's report
    (numeric-only and numeric+classification variants) and the
    decision-usefulness mismatch rate
  * conformal calibration: per vehicle, the rank of t among the healthy
    calibration z-scores gives the implied miscoverage alpha-hat;
    compared with the realized risk on accepted campaign windows
    (finite-sample validity under window exchangeability, disclosed as
    approximate because windows are time-ordered)

Usage: python metropt_risk_coverage.py
Outputs: metropt_risk_coverage.csv, metropt_conformal_table.csv (+ print)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metropt_handlers import _z, handle, load_spec

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
SOURCES = [("metropt3", "features_metropt3.parquet"),
           ("metropt2022", "features_metropt2022.parquet"),
           ("metropt2022B", "features_metropt2022B.parquet")]
THRESHOLDS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, np.inf]


def window_z(feat: dict, v: dict) -> float:
    zmax = 0.0
    for key in v["score_features"]:
        val = feat.get(key)
        if val is None or isinstance(val, str) or val != val:
            continue
        st = v["baseline_stats"][key]
        zmax = max(zmax, _z(float(val), st["median"], st["p01"], st["p99"]))
    return zmax


def load_windows():
    spec = load_spec()
    rows = []
    for tag, par in SOURCES:
        f = pd.read_parquet(RUNS / par)
        for wid, r in f.iterrows():
            feat = r.to_dict()
            v = handle(feat, tag, spec)
            rows.append({
                "vehicle": tag, "window_id": wid, "kind": r["kind"],
                "z": window_z(feat, spec["per_vehicle"][tag]),
                "rule_fired": v["tier"] == "rule",
                "verdict": v["label"],
                "oob_n": len(v["oob_flags"]),
            })
    W = pd.DataFrame(rows)
    gt = {f"{g['vehicle']}::{g['window_id']}": g for g in json.load(
        open(RUNS / "metropt_gt.json", encoding="utf-8"))["windows"]}
    W["gt_visible"] = [any(gt[f"{r.vehicle}::{r.window_id}"]["visible"]
                           .values()) for r in W.itertuples()]
    W["gt_visible_or_ooc"] = [
        r.gt_visible or gt[f"{r.vehicle}::{r.window_id}"].get(
            "ooc_candidate", False) for r in W.itertuples()]
    return spec, W, gt


def handler_risk(W: pd.DataFrame, t: float) -> dict:
    acc = W[(W.rule_fired) | (W.z <= t)].copy()
    # a handler output is WRONG if a GT-visible signature (or OOC
    # candidate) is accepted as healthy, or a control is rule-flagged
    wrong = ((acc.gt_visible_or_ooc & acc.verdict.isin(
        ["healthy", "monitor_deviation"]))
        | ((acc.kind == "control") & ~acc.gt_visible_or_ooc
           & acc.verdict.isin(["air_leak_indication",
                               "oil_degradation_indication",
                               "oil_thermal_anomaly"])))
    return {"threshold": t, "coverage": len(acc) / len(W),
            "n_accepted": len(acc),
            "risk": round(float(wrong.mean()) if len(acc) else np.nan, 4),
            "n_wrong": int(wrong.sum())}


def report_risk(W: pd.DataFrame, t: float, l1: pd.DataFrame,
                acts: pd.DataFrame | None) -> dict:
    acc_keys = set(zip(W[(W.rule_fired) | (W.z <= t)].vehicle,
                       W[(W.rule_fired) | (W.z <= t)].window_id))
    out = {"threshold": t, "coverage": len(acc_keys) / len(W)}
    for s in ("B8", "B3", "B5"):
        d = l1[l1.system == s]
        d = d[[ (r.vehicle, r.window_id) in acc_keys
                for r in d.itertuples()]]
        out[f"{s}_risk_num"] = round(float(
            (1.0 - d.numeric).mean()) if len(d) else np.nan, 4)
        out[f"{s}_risk_full"] = round(float(
            ((1.0 - d.numeric).astype(bool)
             | (1.0 - d.classification).astype(bool)
             | (1.0 - d.rule).astype(bool)).mean()) if len(d)
            else np.nan, 4)
        out[f"{s}_n"] = len(d)
    if acts is not None:
        a = acts[[ (r.system == "B8") and ((r.vehicle, r.window_id)
                                            in acc_keys)
                   for r in acts.itertuples()]]
        a = a[a.gt_action != "adjudicate"]
        out["B8_action_risk"] = round(float(
            1.0 - a.match.mean()) if len(a) else np.nan, 4)
        out["B8_action_n"] = len(a)
    return out


def conformal_table(spec, W):
    """Implied alpha-hat per threshold from the calibration sample."""
    rows = []
    for tag, _ in SOURCES:
        bw = pd.read_parquet(RUNS / f"baseline_windows_{tag}.parquet")
        excl = spec["per_vehicle"][tag].get("ooc_ledger", {})
        keep = bw.copy()
        keep["t_end_dt"] = pd.to_datetime(keep["t_end"])
        n0 = len(keep)
        if excl:
            ex = pd.to_datetime(excl.get("excluded_window_ends", []))
            keep = keep[~keep.t_end_dt.isin(ex)]
        zs = []
        for _, r in keep.iterrows():
            f = r.to_dict()
            zs.append(window_z(f, spec["per_vehicle"][tag]))
        zs = np.array(sorted(zs))
        n = len(zs)
        for t in THRESHOLDS:
            rank = np.searchsorted(zs, t, side="right")
            alpha_hat = 1.0 - rank / (n + 1)
            rows.append({"vehicle": tag, "n_cal": n, "n_excluded": n0 - n,
                         "threshold": t,
                         "alpha_hat": round(float(alpha_hat), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    spec, W, gt = load_windows()
    l1 = pd.read_csv(RUNS / "metropt_l1.csv")
    try:
        acts = pd.read_parquet(RUNS / "metropt_actions.parquet")
        acts = acts[acts.ok.astype(str) == "True"]
    except FileNotFoundError:
        acts = None

    hr = [handler_risk(W, t) for t in THRESHOLDS]
    rr = [report_risk(W, t, l1, acts) for t in THRESHOLDS]
    H, R = pd.DataFrame(hr), pd.DataFrame(rr)
    print("=== handler selective risk-coverage ===")
    print(H.to_string(index=False))
    print("\n=== report-level selective risk on accepted windows ===")
    cols = ["threshold", "coverage", "B8_risk_num", "B8_risk_full",
            "B8_action_risk", "B3_risk_num", "B5_risk_num"]
    print(R[[c for c in cols if c in R.columns]].to_string(index=False))
    H.to_csv(RUNS / "metropt_risk_coverage.csv", index=False)
    R.to_csv(RUNS / "metropt_risk_coverage_reports.csv", index=False)

    C = conformal_table(spec, W)
    piv = C.pivot(index="threshold", columns="vehicle",
                  values="alpha_hat")
    print("\n=== conformal calibration: implied alpha-hat ===")
    print(piv.round(3).to_string())
    C.to_csv(RUNS / "metropt_conformal_table.csv", index=False)

    # headline: frozen operating point (suspect band 6 + OOB>=2 abstain)
    frozen = R[R.threshold == 6.0]
    if len(frozen):
        r = frozen.iloc[0]
        print(f"\nHEADLINE (frozen t=6): coverage {r.coverage:.1%}, "
              f"B8 numeric-error risk {r.B8_risk_num:.1%}, "
              f"full-L1 risk {r.B8_risk_full:.1%}"
              + (f", action mismatch {r.B8_action_risk:.1%}"
                 if "B8_action_risk" in r else ""))


if __name__ == "__main__":
    main()
