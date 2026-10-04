"""LOEO part 2 standalone (branch-name fix). Run after blockers part 1."""
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metropt_handlers import handle, load_spec  # noqa: E402
from metropt_l1 import _feat_row  # noqa: E402
from metropt_analysis import load_all, FAULT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"

BRANCH2KEYS = {
    "A1a": ["A1_mc_on_frac", "A1_tp2_frac_ge9_5"],
    "A1b": ["A1b_mc_on_maxrun_min"],
    "A2": ["A2_lps_maxrun_min"],
    "A3": ["A3_tp2_p95_max", "A3_mc_on_frac_min"],
    "D1": ["D1_oil_elev_c"],
    "D2": ["D2_oil_trend_c"],
    "D3": ["D3_oil_level_frac"],
}
# disarm values make the branch impossible to fire
DISARM = {"A1_mc_on_frac": 1e9, "A1_tp2_frac_ge9_5": -1e9,
          "A1b_mc_on_maxrun_min": 1e9, "A2_lps_maxrun_min": 1e9,
          "A3_tp2_p95_max": -1e9, "A3_mc_on_frac_min": 1e9,
          "D1_oil_elev_c": 1e9, "D2_oil_trend_c": 1e9,
          "D3_oil_level_frac": 1e9}


def fired_branches(verdict) -> set:
    ev = verdict.get("evidence", {})
    f = ev.get("rule_branches_fired", {})
    return {k.split("_")[0] for k, on in f.items() if on}


def main() -> None:
    spec, gt, l1, l2, l3 = load_all()
    rep0 = pd.read_parquet(RUNS / "reports_metropt_B1.parquet")
    verdicts = {}
    for _, r in rep0.iterrows():
        v = handle(_feat_row(r.vehicle, r.window_id), r.vehicle, spec)
        verdicts[(r.vehicle, r.window_id)] = v

    # branch x event firing
    branch_events, sig = {}, set()
    for key, v in verdicts.items():
        if v["tier"] == "rule":
            sig.add(key)
            veh, wid = key
            evid = rep0[(rep0.vehicle == veh)
                        & (rep0.window_id == wid)].iloc[0].event_id
            branch_events.setdefault((veh, evid), set()).update(
                fired_branches(v))
    print("branch x event firing:")
    for k, b in sorted(branch_events.items(), key=lambda x: str(x[0])):
        print("  ", k, sorted(b))

    events = {}
    for _, r in rep0.iterrows():
        if r.kind == "positive":
            events.setdefault((r.vehicle, r.event_id), []).append(r.window_id)

    rows = []
    for (veh, evid), wins in sorted(events.items(),
                                     key=lambda x: str(x[0])):
        removed = sorted(branch_events.get((veh, evid), set()))
        if not removed:
            rows.append({"vehicle": veh, "event": evid,
                         "branches_removed": "(none fire on this event)",
                         "heldout_windows": len(wins),
                         "sig_windows": 0,
                         "detected_after_removal": "-",
                         "control_FA_delta": "-"})
            continue
        spec_e = copy.deepcopy(spec)
        th = spec_e["per_vehicle"][veh]["thresholds"]
        for b in removed:
            for k in BRANCH2KEYS[b]:
                if k in th:
                    th[k] = DISARM[k]
        det = 0
        sig_w = 0
        for wid in wins:
            v0 = verdicts[(veh, wid)]
            if v0["tier"] == "rule":
                sig_w += 1
            v = handle(_feat_row(veh, wid), veh, spec_e)
            if v["tier"] == "rule" and v["label"] in FAULT:
                det += 1
        # control FA change
        fa0 = fa1 = 0
        for _, r in rep0.iterrows():
            if r.kind != "control":
                continue
            v = handle(_feat_row(r.vehicle, r.window_id), r.vehicle, spec_e)
            v0 = verdicts[(r.vehicle, r.window_id)]
            if v0["tier"] == "rule":
                fa0 += 1
            if v["tier"] == "rule":
                fa1 += 1
        rows.append({"vehicle": veh, "event": evid,
                     "branches_removed": "+".join(removed),
                     "heldout_windows": len(wins), "sig_windows": sig_w,
                     "detected_after_removal": det,
                     "control_FA_delta": f"{fa0}->{fa1}"})
    out = pd.DataFrame(rows)
    out.to_csv(RUNS / "metropt_loeo.csv", index=False)
    print("\n=== leave-one-event-out ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
