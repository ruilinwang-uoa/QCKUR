"""
metropt_crossvehicle.py -- cross-vehicle detection folds. For each ordered pair (src -> tgt) of the three
vehicle-epochs, run the handler with SRC's per-vehicle block (thresholds + baselines frozen on src's healthy pool) over TGT's windows:

  * detection: on tgt's signature windows (the rule-fired set), does the
    src-calibrated handler fire the rule tier with the correct label?
  * nominal-side cost: rule-tier fire rate on tgt's nominal controls
    (excluding the adjudicated OOC-candidate controls, whose signatures
    are real by the ledger -- firing there is correct identification of
    an undocumented episode, not a false alarm).

This is the out-of-sample test the r9 referee asked for: the signature
RULES were authored on the documented events of all three units, but
the THRESHOLDS are per-vehicle calibrations; transferring src's
calibration to tgt measures how much of detection survives without
ever seeing tgt's data. Zero API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metropt_l1 import _feat_row, classification_check  # noqa: E402
from metropt_handlers import handle, load_spec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
VEHICLES = ["metropt3", "metropt2022", "metropt2022B"]
FAULT = {"air_leak_indication", "oil_degradation_indication",
         "air_and_oil_indication", "oil_thermal_anomaly"}
# OOC-candidate controls (real ghost signatures per the adjudication
# ledger); rule fires there = correct, not FA
OOC_CTRLS = {("metropt2022", "ctl_ev1_o24_0"), ("metropt2022", "ctl_ev1_o72_1"),
             ("metropt3", "ctl_ev2_o6_0")}


def main() -> None:
    spec = load_spec()
    gt = {f"{g['vehicle']}::{g['window_id']}": g for g in json.load(
        open(RUNS / "metropt_gt.json", encoding="utf-8"))["windows"]}
    rep0 = pd.read_parquet(RUNS / "reports_metropt_B1.parquet")

    # signature windows = rule tier fires under the OWN-vehicle block
    sig = []
    for _, r in rep0.iterrows():
        v = handle(_feat_row(r.vehicle, r.window_id), r.vehicle, spec)
        if v["tier"] == "rule":
            sig.append((r.vehicle, r.window_id, r.kind, v["label"]))
    print(f"signature windows (own calibration): {len(sig)}")

    rows = []
    for src in VEHICLES:
        for tgt in VEHICLES:
            if src == tgt:
                continue
            det = n_sig = 0
            fa = n_nom = 0
            ooc_hit = n_ooc = 0
            for _, r in rep0.iterrows():
                if r.vehicle != tgt:
                    continue
                g = gt[f"{r.vehicle}::{r.window_id}"]
                v = handle(_feat_row(r.vehicle, r.window_id), src, spec)
                fires = v["tier"] == "rule" and v["label"] in FAULT
                if (r.vehicle, r.window_id) in {(s[0], s[1]) for s in sig
                                                if s[0] == tgt}:
                    n_sig += 1
                    own = [s[3] for s in sig if s[0] == tgt and
                           s[1] == r.window_id][0]
                    det += int(fires and v["label"] == own)
                elif r.kind == "control":
                    if (r.vehicle, r.window_id) in OOC_CTRLS:
                        n_ooc += 1
                        ooc_hit += int(fires)
                    else:
                        n_nom += 1
                        fa += int(fires)
            rows.append({"src": src, "tgt": tgt,
                         "sig_detected": f"{det}/{n_sig}",
                         "det_rate": round(det / n_sig, 2) if n_sig else None,
                         "nominal_FA": f"{fa}/{n_nom}",
                         "ooc_identified": f"{ooc_hit}/{n_ooc}"})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(RUNS / "metropt_crossvehicle_folds.csv", index=False)
    print(f"-> {RUNS / 'metropt_crossvehicle_folds.csv'}")


if __name__ == "__main__":
    main()
