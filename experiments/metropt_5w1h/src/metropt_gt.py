"""
metropt_gt.py -- ground-truth builder for the MetroPT campaign.

Semantics (pre-registered, PREREGISTRATION.md sections 3/6):
  * POSITIVE windows carry lead-time semantics: a window ending L hours
    before a documented failure is labelled by WHAT THE TELEMETRY SHOWS,
    plus the future event as context. If the pre-declared evidence
    criteria fire, the expected report detects; if not, the expected
    report states no actionable indication (a fabricated fault is
    wrong, a silent nominal report is right).
  * CONTROL windows are expected nominal; a control whose telemetry
    shows an unexplained signature is flagged control_with_signature
    (OOC candidate, adjudicated manually -- never silently healthy).
  * Numeric truths = the frozen feature digest (L1 data side).
  * Decision-usefulness GT = expected action class, stratified by
    (visibility x failure class x documented disposition).

Pre-declared visibility criteria (independent of the handler's firing
thresholds -- they only stratify expectations):
    vis_air = lps_frac > 0.01  OR lps_maxrun_min >= 10
              OR (tp2_frac_ge9_5 <= 0.05 AND mc_on_frac >= 0.8)
    vis_oil = oil_elev >= 1.0 degC  OR oil_level_frac >= 0.5 (the latter
              only on vehicles with documented polarity)
(warming trend is NOT oil visibility: within-window warming of several
degC is routine daily operation -- it belongs to the air-leak thermal
story, and the 2026-09-21 smoke showed trend-based visibility flags
21/68 healthy controls with NEGATIVE elevation; corrected same day,
before any report generation)

Usage: python metropt_gt.py
Writes runs/metropt3/metropt_gt.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
DATA = ROOT / "data" / "metropt3"
SPEC = ROOT / "knowledge" / "metropt" / "metropt_spec.json"

SOURCES = [
    # tag, window manifest, features parquet, data manifest
    ("metropt3", "window_manifest.json",
     "features_metropt3.parquet", "metropt3_manifest.json"),
    ("metropt2022", "window_manifest_metropt2022.json",
     "features_metropt2022.parquet", "dataset_train_2022_manifest.json"),
    ("metropt2022B", "window_manifest_metropt2022B.json",
     "features_metropt2022B.parquet", "metropt2_trainB_manifest.json"),
]

# event-id -> (failure_mode, severity_class, disposition)
# severity_class: severe (removed/next-day repair) / mild (continued ops
# or self-resolved). NOTE the 2020 UCI table labels ALL four events
# "High stress"; the behavioural anchor here is the documented
# disposition (ev1 ran 12 days before repair, ev2 self-resolved), which
# is what the decision-usefulness GT needs. Label/behaviour tension is
# disclosed in the paper, not hidden.
EVENT_META = {
    ("metropt3", 1): ("air_leak", "mild", "repaired_after_12d"),
    ("metropt3", 2): ("air_leak", "mild", "self_resolved"),
    ("metropt3", 3): ("air_leak", "severe", "repaired"),
    ("metropt3", 4): ("air_leak", "severe", "repaired"),
    ("metropt2022", 1): ("air_leak", "severe", "removed_from_service"),
    ("metropt2022", 2): ("air_leak", "mild", "continued_in_operation"),
    ("metropt2022", 3): ("oil_leak", "severe", "removed_from_service"),
    ("metropt2022B", 1): ("air_leak", "severe", "removed_from_service"),
    ("metropt2022B", 2): ("oil_leak", "severe", "removed_from_service"),
}


def expected_action(visible, mode, sev, disposition, lead_h) -> str:
    if not visible:
        return "no_action_or_monitor"      # acting now would be premature
    if mode == "oil_leak":
        return "schedule_inspection" if lead_h > 12 else "remove_or_urgent_inspection"
    if sev == "severe":
        return "remove_or_urgent_inspection" if lead_h <= 48 else "schedule_inspection"
    return "monitor_and_verify"            # mild air-leak class


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    gt = {"version": "1.0", "visibility_criteria": {
              "vis_air": "lps_frac > 0.01 OR lps_maxrun_min >= 10 OR "
                         "(tp2_frac_ge9_5 <= 0.05 AND mc_on_frac >= 0.8)",
              "vis_oil": "oil_elev >= 1.0C OR oil_trend_c >= 2.0C"},
          "windows": []}

    for tag, wman, fpar, dman in SOURCES:
        events = json.loads((DATA / dman).read_text(
            encoding="utf-8"))["failure_events"]
        ev_by_id = {e["id"]: e for e in events}
        feats = pd.read_parquet(RUNS / fpar)
        v = spec["per_vehicle"][tag]
        for _, row in feats.iterrows():
            f = row.to_dict()
            oil_elev = round(f.get("oil_max", np.nan)
                             - v["oil_max_p99"], 2)
            vis_air = bool((f.get("lps_frac") or 0) > 0.01
                           or (f.get("lps_maxrun_min") or 0) >= 10
                           or ((f.get("tp2_frac_ge9_5") or 1) <= 0.05
                               and (f.get("mc_on_frac") or 0) >= 0.8))
            # oil-level visibility only on vehicles with documented
            # polarity (2020/B sit at 1.0 in healthy operation)
            pol_ok = (v["oil_level_polarity"]
                      == "documented_1_below_expected")
            vis_oil = bool((oil_elev >= 1.0 if not np.isnan(oil_elev)
                            else False)
                           or (pol_ok
                               and f.get("oil_level_frac") is not None
                               and not pd.isna(f.get("oil_level_frac"))
                               and f.get("oil_level_frac") >= 0.5))
            # strong warming beyond typical daily operation = thermal
            # (air-leak overload) visibility; threshold = the vehicle's
            # healthy p99 so ~1% of nominal windows naturally cross it
            # (those become OOC candidates, which is the point)
            trend_p99 = v["baseline_stats"]["oil_trend_c"]["p99"]
            tv = f.get("oil_trend_c")
            vis_thermal = bool(tv is not None and not pd.isna(tv)
                               and tv >= trend_p99)
            any_vis = vis_air or vis_oil or vis_thermal
            entry = {
                "window_id": row.name, "vehicle": tag, "kind": f["kind"],
                "t_start": str(f["t_start"]), "t_end": str(f["t_end"]),
                "oil_elev_c": oil_elev,
                "visible": {"air": vis_air, "oil": vis_oil,
                            "thermal": vis_thermal},
                "numeric_truth": {k: (None if pd.isna(x) else x)
                                  for k, x in f.items()
                                  if isinstance(x, (int, float,
                                                    np.integer,
                                                    np.floating))},
            }
            if f["kind"] == "positive":
                eid = int(f["event_id"])
                mode, sev, disp = EVENT_META[(tag, eid)]
                lead = float(f["offset_h"])
                entry["event"] = {
                    "id": eid, "mode": mode, "severity_class": sev,
                    "disposition": disp,
                    "anchor": str(ev_by_id[eid]["end"]),
                    "lead_h": lead,
                }
                if vis_air or vis_oil:
                    what = f"{mode} signature visible"
                elif vis_thermal:
                    what = "thermal-anomaly signature visible " \
                           "(overload warming pattern)"
                else:
                    what = "no actionable indication at this lead"
                entry["expected"] = {
                    "what": what,
                    "action_class": expected_action(
                        any_vis, mode, sev, disp, lead),
                }
            else:
                entry["event"] = None
                entry["expected"] = {
                    "what": ("UNEXPLAINED SIGNATURE -- OOC candidate"
                             if any_vis else "nominal"),
                    "action_class": ("adjudicate" if any_vis
                                     else "no_action_or_monitor"),
                }
                if any_vis:
                    entry["ooc_candidate"] = True
            gt["windows"].append(entry)

    dest = RUNS / "metropt_gt.json"
    dest.write_text(json.dumps(gt, indent=1), encoding="utf-8")
    pos = [w for w in gt["windows"] if w["kind"] == "positive"]
    ctl = [w for w in gt["windows"] if w["kind"] != "positive"]
    vis_pos = [w for w in pos
               if w["expected"]["what"].endswith("visible")
               or "visible" in w["expected"]["what"]]
    ooc_ctl = [w for w in ctl if w.get("ooc_candidate")]
    print(f"GT -> {dest}")
    print(f"positives {len(pos)} (signature visible in {len(vis_pos)}), "
          f"controls {len(ctl)} (OOC candidates {len(ooc_ctl)})")
    for w in vis_pos:
        print(f"  VIS+ {w['window_id']:18s} lead {w['event']['lead_h']:4.0f}h "
              f"{w['event']['mode']:9s} action={w['expected']['action_class']}")
    for w in ooc_ctl:
        print(f"  OOC? {w['window_id']:18s} visible={w['visible']} "
              f"oil_elev={w['oil_elev_c']}")


if __name__ == "__main__":
    main()
