"""
metropt_analysis.py -- three-layer summary + lead-time curves +
decision-usefulness for the MetroPT campaign.

Formulas are protocol-identical to AI4I/bearing:
  L1    = 0.30*numeric + 0.30*rule + 0.40*classification
  Final = 0.35*100*L1 + 0.30*100*L2 + 0.35*20*L3

Additional RESS-native outputs:
  * lead-time detection curve: per offset {6,12,24,48,72} h, the fraction
    of signature-visible positive windows whose report claims the fault
  * decision-usefulness: action-class match vs GT disposition anchor
  * per-vehicle breakdown (fleet transfer view)
  * miss/hallucination decomposition of classification errors

Usage: python metropt_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metropt_l1 import _feat_row, classification_check
from metropt_handlers import handle, load_spec

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
SYSTEMS = ["B1", "B2", "B3", "B3R", "B5", "B8"]

FAULT = {"air_leak_indication", "oil_degradation_indication",
         "air_and_oil_indication", "oil_thermal_anomaly"}


def load_all():
    spec = load_spec()
    gt = {f"{g['vehicle']}::{g['window_id']}": g for g in json.load(
        open(RUNS / "metropt_gt.json", encoding="utf-8"))["windows"]}
    l1 = pd.read_csv(RUNS / "metropt_l1.csv")
    l2 = pd.read_parquet(RUNS / "metropt_l2.parquet")
    l2 = l2[l2.ok.astype(str) == "True"]
    l3 = pd.read_parquet(RUNS / "metropt_l3.parquet")
    l3 = l3[l3.ok.astype(str) == "True"]
    return spec, gt, l1, l2, l3


def main() -> None:
    spec, gt, l1, l2, l3 = load_all()

    # ---- three-layer main table ------------------------------------------
    l1["L1"] = (0.30 * l1.numeric + 0.30 * l1.rule
                + 0.40 * l1.classification)
    m = (l1.groupby("system")
         .agg(L1=("L1", "mean"), n=("window_id", "count")).reset_index())
    m = m.merge(l2.groupby("system").agg(
        L2=("L2", "mean"), cons=("consistency", "mean"),
        contra=("contradiction", "mean"),
        L2cov=("coverage", "mean")).reset_index(), on="system")
    m = m.merge(l3.groupby("system").agg(
        L3=("L3", "mean"), cov=("cov_count", "mean"),
        faith=("faithfulness", "mean"), act=("actionability", "mean"),
        coh=("coherence", "mean")).reset_index(), on="system")
    m["Final"] = 0.35 * 100 * m.L1 + 0.30 * 100 * m.L2 + 0.35 * 20 * m.L3
    order = {s: i for i, s in enumerate(SYSTEMS)}
    m = m.sort_values("system", key=lambda s: s.map(order))
    print("=== three-layer main table (judged-only; L2 floor-imputed twin) ===")
    print(m.round(3).to_string(index=False))
    # floor-imputed twin: unrecovered L2 rows scored at the zero-claim
    # floor 0.25 (AI4I conservative convention; bracket vs judged-only)
    l2f = l2.copy()
    l2f["ok"] = l2f["ok"].astype(str)
    have = {(r.system, r.vehicle, r.window_id) for r in l2f.itertuples()}
    for s in SYSTEMS:
        for _, r in l1[l1.system == s].iterrows():
            if (s, r.vehicle, r.window_id) not in have:
                l2f.loc[len(l2f)] = {"system": s, "vehicle": r.vehicle,
                                     "window_id": r.window_id, "L2": 0.25,
                                     "ok": "True"}
    mf = m.copy()
    mf = mf.drop(columns=["L2", "Final"]).merge(
        l2f.groupby("system").agg(L2=("L2", "mean")).reset_index(),
        on="system")
    mf["Final"] = 0.35 * 100 * mf.L1 + 0.30 * 100 * mf.L2 + 0.35 * 20 * mf.L3
    mf = mf.sort_values("system", key=lambda s: s.map(order))
    cols = ["system", "n", "L1", "L2", "L3", "Final"]
    print(mf[cols].round(3).to_string(index=False))
    m.to_csv(RUNS / "metropt_main_summary.csv", index=False)
    mf.to_csv(RUNS / "metropt_main_summary_floorimp.csv", index=False)

    # ---- lead-time detection curve ---------------------------------------
    print("\n=== lead-time detection (signature-visible positives) ===")
    curve = []
    for s in SYSTEMS:
        rep = pd.read_parquet(RUNS / f"reports_metropt_{s}.parquet")
        for off in (6, 12, 24, 48, 72):
            n = hit = 0
            for _, r in rep.iterrows():
                g = gt[f"{r.vehicle}::{r.window_id}"]
                if (r.kind != "positive") or (r.offset_h != off):
                    continue
                feat = _feat_row(r.vehicle, r.window_id)
                v = handle(feat, r.vehicle, spec)
                cc = classification_check(str(r.text), v)
                if cc["expected"] not in FAULT:
                    continue          # detection-opportunity set only: the
                                    # correct answer must be a fault claim
                                    # (r9 fix: the old lenient criterion
                                    # admitted healthy/monitor-expected
                                    # windows and credited "correctly said
                                    # healthy" as a detection, e.g.
                                    # metropt2022/ev1_o24; it also dropped
                                    # the thermal-visible windows, so the
                                    # curve covered 7 windows, not the 9
                                    # the text describes)
                n += 1
                if cc["claimed"] == cc["expected"]:
                    hit += 1
            curve.append({"system": s, "offset_h": off, "n": n,
                          "detected": hit,
                          "rate": round(hit / n, 3) if n else None})
    cv = pd.DataFrame(curve)
    piv = cv.pivot(index="offset_h", columns="system", values="rate")
    print(piv[SYSTEMS].round(2).to_string())
    cv.to_csv(RUNS / "metropt_leadtime_curve.csv", index=False)

    # ---- decision usefulness (blind action classifier; adjudicate rows
    # have no ground-truth action and are excluded) ------------------------
    print("\n=== decision-usefulness (blind action classifier vs GT) ===")
    try:
        act = pd.read_parquet(RUNS / "metropt_actions.parquet")
        act = act[act.ok.astype(str) == "True"]
        act = act[act.gt_action != "adjudicate"]
        d = (act.groupby("system").match
             .agg(["count", "sum", "mean"]).round(3).reset_index()
             .rename(columns={"count": "n", "sum": "match",
                              "mean": "rate"}))
        print(d.to_string(index=False))
        d.to_csv(RUNS / "metropt_decision_usefulness.csv", index=False)
        xt = pd.crosstab(act.gt_action, act.action_claim)
        print("\nGT x claimed confusion:")
        print(xt.to_string())
    except FileNotFoundError:
        print("metropt_actions.parquet missing -- run metropt_action_eval")

    # ---- per-vehicle transfer view ---------------------------------------
    # r9 fix: compute on the SAME basis as the main table (per-vehicle
    # layer means over each layer's own ok rows, then combine) -- the old
    # triple-inner-join basis selected only windows with all three layers
    # judged (52/80 for B1/B8) and read 1+ point higher.
    print("\n=== per-vehicle Final (main-table basis: layer means combined) ===")
    l1v = l1.groupby(["system", "vehicle"]).L1.agg(["mean", "count"])
    l2v = l2.groupby(["system", "vehicle"]).L2.agg(["mean", "count"])
    l3v = l3.groupby(["system", "vehicle"]).L3.agg(["mean", "count"])
    pv = (l1v.rename(columns={"mean": "L1", "count": "n1"})
          .join(l2v.rename(columns={"mean": "L2", "count": "n2"}))
          .join(l3v.rename(columns={"mean": "L3", "count": "n3"})))
    pv["Final"] = 0.35 * 100 * pv.L1 + 0.30 * 100 * pv.L2 + 0.35 * 20 * pv.L3
    piv = pv.reset_index().pivot_table(index="vehicle", columns="system",
                                       values="Final")[SYSTEMS]
    print(piv.round(1).to_string())
    print("\njudged-only n per vehicle (L2):")
    print(pv.reset_index().pivot_table(index="vehicle", columns="system",
                                       values="n2")[SYSTEMS].to_string())
    piv.round(2).to_csv(RUNS / "metropt_per_vehicle_final.csv")

    # ---- miss / hallucination decomposition ------------------------------
    print("\n=== classification error decomposition ===")
    for s in SYSTEMS:
        rep = pd.read_parquet(RUNS / f"reports_metropt_{s}.parquet")
        miss = hallu = 0
        for _, r in rep.iterrows():
            feat = _feat_row(r.vehicle, r.window_id)
            v = handle(feat, r.vehicle, spec)
            cc = classification_check(str(r.text), v)
            if cc["ok"] or cc["ambiguous"]:
                continue
            if cc["expected"] in FAULT and cc["claimed"] not in FAULT:
                miss += 1
            elif cc["expected"] not in FAULT and cc["claimed"] in FAULT:
                hallu += 1
        print(f"{s}: misses {miss}, hallucinated-fault-claims {hallu}")

    print("\n-> metropt_main_summary / leadtime_curve / "
          "decision_usefulness CSVs in runs/metropt3/")


if __name__ == "__main__":
    main()
