"""
PART 1: EMPIRICAL CONFORMAL COVERAGE CHECK
  Per vehicle: chronologically split the (post-exclusion) calibration
  pool into halves A/B; recompute baseline stats from A; compute the
  nonconformity score s(w) for all windows under A's stats; the rank of
  the frozen thresholds t in {2,4,6} on A gives alpha-hat_A(t) =
  1 - rank/(n_A+1); the REALIZED false-abstention rate on held-out B is
  compared against it (and with A/B swapped). A monthly-block variant
  (even months calibrate, odd months test) probes the time-ordering
  caveat. Pure-local computation; no LLM calls.

PART 2: LEAVE-ONE-EVENT-OUT DETECTION FOLDS
  For each documented event: disable the rule branches that fire on its
  signature windows (simulating never having authored rules for that
  failure signature), then re-run the handler on (a) the held-out
  event's positive windows -- does any REMAINING branch still deliver
  the correct fault class? -- and (b) all 68 nominal controls -- false
  alarms unchanged? Also emits the branch x event firing matrix.

Outputs: runs/metropt3/metropt_conformal_coverage.csv,
         runs/metropt3/metropt_loeo.csv (+ print).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metropt_handlers import _z, handle, load_spec
from metropt_l1 import _feat_row
from metropt_analysis import load_all, classification_check, FAULT

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
SOURCES = [("metropt3", "features_metropt3.parquet"),
           ("metropt2022", "features_metropt2022.parquet"),
           ("metropt2022B", "features_metropt2022B.parquet")]
THRESH = [2.0, 4.0, 6.0]


# ---------------------------------------------------------------- part 1
def stats_from(df: pd.DataFrame, feats: list) -> dict:
    st = {}
    for k in feats:
        x = pd.to_numeric(df[k], errors="coerce").dropna()
        st[k] = {"p01": float(x.quantile(0.01)),
                 "median": float(x.median()),
                 "p99": float(x.quantile(0.99))}
    return st


def zvec(df: pd.DataFrame, feats: list, st: dict) -> np.ndarray:
    out = np.zeros(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        z = 0.0
        for k in feats:
            v = r.get(k)
            if v is None or isinstance(v, str) or v != v:
                continue
            s = st[k]
            z = max(z, _z(float(v), s["median"], s["p01"], s["p99"]))
        out[i] = z
    return out


def conformal_coverage() -> pd.DataFrame:
    spec = load_spec()
    rows = []
    for tag, par in SOURCES:
        v = spec["per_vehicle"][tag]
        feats = v["score_features"]
        cal = pd.read_parquet(RUNS / par)
        cal = cal.assign(_te=pd.to_datetime(cal["t_end"], format="mixed")).sort_values("_te").drop(columns=["_te"]).reset_index(drop=True)
        n = len(cal)
        halves = [("chrono A->B", cal.iloc[:n // 2], cal.iloc[n // 2:]),
                  ("chrono B->A", cal.iloc[n // 2:], cal.iloc[:n // 2])]
        # monthly blocking: even months calibrate, odd test; then swap
        months = pd.to_datetime(cal["t_end"], format="mixed").dt.month
        for name, A, B in halves + [
                ("month even->odd", cal[months % 2 == 0], cal[months % 2 == 1]),
                ("month odd->even", cal[months % 2 == 1], cal[months % 2 == 0])]:
            st = stats_from(A, feats)
            zA, zB = zvec(A, feats, st), zvec(B, feats, st)
            nA = len(zA)
            for t in THRESH:
                rank = int((zA <= t).sum())
                ahat = 1.0 - rank / (nA + 1)
                k = int((zB > t).sum())
                realized = k / len(zB)
                rows.append({"vehicle": tag, "split": name, "t": t,
                             "n_cal": nA, "n_test": len(zB),
                             "alpha_hat": round(ahat, 4),
                             "realized": round(realized, 4),
                             "k_over_n": f"{k}/{len(zB)}",
                             "covered": realized <= ahat})
    out = pd.DataFrame(rows)
    out.to_csv(RUNS / "metropt_conformal_coverage.csv", index=False)
    print("=== conformal coverage check (split-half + monthly blocks) ===")
    piv = out.pivot_table(index=["vehicle", "t"], columns="split",
                          values="covered")
    print(piv.to_string())
    ok = out.groupby("t").covered.mean()
    print("\nfraction of split-configs with realized <= alpha-hat:")
    print(ok.round(3).to_string())
    print("median |realized - alpha_hat| =",
          (out.realized - out.alpha_hat).abs().median().round(4))
    return out


# ---------------------------------------------------------------- part 2
BRANCH_KEYS = ["A1_mc_on_frac", "A1_tp2_frac_ge9_5", "A1b_mc_on_maxrun_min",
               "A2_lps_maxrun_min", "A3_tp2_p95_max", "A3_mc_on_frac_min",
               "D1_oil_elev_c", "D2_oil_trend_c", "D3_oil_level_frac"]
BRANCH_OF_KEY = {"A1_mc_on_frac": "A1a", "A1_tp2_frac_ge9_5": "A1a",
                 "A1b_mc_on_maxrun_min": "A1b", "A2_lps_maxrun_min": "A2",
                 "A3_tp2_p95_max": "A3", "A3_mc_on_frac_min": "A3",
                 "D1_oil_elev_c": "D1", "D2_oil_trend_c": "D2",
                 "D3_oil_level_frac": "D3"}
# disarm values make the branch impossible to fire
DISARM = {"A1_mc_on_frac": 1e9, "A1_tp2_frac_ge9_5": -1e9,
          "A1b_mc_on_maxrun_min": 1e9, "A2_lps_maxrun_min": 1e9,
          "A3_tp2_p95_max": -1e9, "A3_mc_on_frac_min": 1e9,
          "D1_oil_elev_c": 1e9, "D2_oil_trend_c": 1e9,
          "D3_oil_level_frac": 1e9}


def loeo() -> pd.DataFrame:
    spec0 = load_spec()
    spec, gt, l1, l2, l3 = load_all()
    rep0 = pd.read_parquet(RUNS / "reports_metropt_B1.parquet")
    # signature windows and firing branches (own calibration)
    sig, branch_events = [], {}
    for _, r in rep0.iterrows():
        v = handle(_feat_row(r.vehicle, r.window_id), r.vehicle, spec)
        if v["tier"] == "rule":
            fired = [k for k, on in (v.get("evidence", {})
                                     .get("rule_branches_fired", {}) or
                                     v.get("rule_branches_fired", {})
                                     .items()) if on]
            branches = sorted({BRANCH_OF_KEY[k] for k in fired
                               if k in BRANCH_OF_KEY})
            sig.append((r.vehicle, r.window_id, r.event_id, branches))
            branch_events.setdefault((r.vehicle, r.event_id), set()).update(
                branches)
    print("\nbranch x event firing:")
    for k, b in sorted(branch_events.items(), key=lambda x: str(x[0])):
        print(" ", k, sorted(b))

    # positives per event
    events = {}
    for _, r in rep0.iterrows():
        if r.kind == "positive":
            events.setdefault((r.vehicle, r.event_id), []).append(
                (r.window_id,))

    rows = []
    for (veh, evid), wins in sorted(events.items(), key=lambda x: str(x[0])):
        removed = sorted(branch_events.get((veh, evid), set()))
        if not removed:
            rows.append({"vehicle": veh, "event": evid, "branches_removed": "",
                         "heldout_windows": len(wins),
                         "detected_after_removal": "-",
                         "control_FA": "-",
                         "note": "no rule branch fires on this event "
                                 "(no signature windows)"})
            continue
        import copy
        spec_e = copy.deepcopy(spec)
        for b in removed:
            for k, bk in BRANCH_OF_KEY.items():
                if bk == b and k in spec_e["per_vehicle"][veh]["thresholds"]:
                    spec_e["per_vehicle"][veh]["thresholds"][k] = DISARM[k]
        det = 0
        for (wid,) in wins:
            v = handle(_feat_row(veh, wid), veh, spec_e)
            if v["tier"] == "rule" and v["label"] in FAULT:
                det += 1
        fa = 0
        nctl = 0
        for _, r in rep0.iterrows():
            if r.kind != "control":
                continue
            nctl += 1
            v = handle(_feat_row(r.vehicle, r.window_id), r.vehicle, spec_e)
            fa += int(v["tier"] == "rule" and v["label"] in FAULT
                      and v["label"] == "oil_thermal_anomaly"
                      and (r.vehicle, r.window_id) not in
                      {(s[0], s[1]) for s in sig})
        rows.append({"vehicle": veh, "event": evid,
                     "branches_removed": "+".join(removed),
                     "heldout_windows": len(wins),
                     "detected_after_removal": det,
                     "control_FA": fa,
                     "note": ""})
    out = pd.DataFrame(rows)
    out.to_csv(RUNS / "metropt_loeo.csv", index=False)
    print("\n=== leave-one-event-out ===")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    conformal_coverage()
    loeo()
