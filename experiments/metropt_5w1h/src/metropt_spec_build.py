"""
metropt_spec_build.py -- assemble + FREEZE the MetroPT two-tier knowledge
spec (rules, score bands, per-vehicle baselines, provenance).

Threshold policy (v2, pre-declared, bearing-lesson discipline):
  rule thresholds are chosen for ZERO false alarms on the healthy
  calibration sample, with a safety margin and a physics floor.
  v1 (naive lowpress_highload / lps_frac / tp2_p95-min rules) was
  FALSIFIED by the 2020 healthy sample (offload running MC~4A holds TP2
  low for hours in healthy states; healthy lps_frac reaches 3.7%;
  healthy oil_max reaches 83C on the 2020 unit) -- recorded here as
  provenance. v2 air-leak rules (duty-cycle physics):
    A1 mc_on_frac >= max(0.90, healthy p99+0.05) AND tp2_frac_ge9_5
       <= 0.05      (continuous working duty, cutoff never reached)
    A2 lps_maxrun_min >= max(60, 1.5 x healthy max)   (sustained LPS)
    A3 tp2_p95 <= healthy p01 - 0.5 AND mc_on_frac >= 0.5
  oil rules (band-relative by design, per-vehicle baselines):
    D1 oil_elev (over baseline oil_max p99) >= max(2.0, p99-p50 degC)
    D2 oil_trend_c            >= max(2.5, 1.5 x healthy_p99)
    D3 oil_level_frac         >= 0.5 IF (and only if) the epoch's healthy
       sample shows oil_level_frac ~= 0 (documented polarity "1 = oil
       below expected"); epochs where healthy windows sit near 1.0 are
       flagged polarity_suspect and D3 is DISABLED for them.
  detection-side justification = documented failure signatures (W2 label
  forensics), not the healthy sample; any post-freeze change goes into a
  dated addendum, never silently.

Usage:
  python metropt_spec_build.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
DEST = ROOT / "knowledge" / "metropt"
TAGS = [("metropt3", "2020 epoch unit (UCI MetroPT-3)"),
        ("metropt2022", "train A (dataset_train_2022.csv)"),
        ("metropt2022B", "train B (MetroPT2.csv)")]

# Manually adjudicated baseline exclusions (ghost windows = unexplained
# operational signatures inside the nominal pool; each entry needs a
# dated adjudication note below). Excluded windows are logged in the
# spec's ooc_ledger -- they are findings, not noise to hide.
# Adjudicated 2026-09-21 from baseline_windows_*.parquet:
#   metropt2022  2022-02-12 14:39:22  STRONG undocumented air-leak-like
#     episode: LPS sustained 360 min (frac 0.30) + TP2 p95 7.53 bar +
#     oil 88.7C (compressor overload) -- 16 days BEFORE documented F1;
#     absent from Veloso 2022 Table 2 and every secondary source.
#   metropt3     2020-03-29 16:29:00  STRONG duty anomaly: mc_on_frac
#     1.00, TP2 never >= 9.5 bar, LPS 22.6 min, oil 83.1C -- not in the
#     UCI table (4 documented events).
#   metropt3     2020-06-25 06:04:59  MODERATE duty anomaly: mc_on_frac
#     0.96 with cutoff never reached, no LPS -- atypical demand cannot
#     be fully excluded; excluded conservatively and disclosed.
GHOST_EXCLUSIONS = {
    "metropt2022": ["2022-02-12 14:39:22"],
    "metropt3": ["2020-03-29 16:29:00", "2020-06-25 06:04:59"],
}
GHOST_ADJUDICATION_NOTES = {
    "metropt2022": "2026-09-21: window ending 2022-02-12 14:39 shows a "
                   "sustained-LPS air-leak signature (360 min below 7 "
                   "bar) with no entry in any documented failure table "
                   "-- treated as an UNDOCUMENTED operational episode "
                   "(OOC), excluded from the nominal baseline and "
                   "logged; first piece of evidence for the real-OOC "
                   "rate (prereg H3).",
    "metropt3": "2026-09-21: two duty anomalies (2020-03-29: full-day "
                "working duty, cutoff never reached, brief LPS; "
                "2020-06-25: 96% working duty without cutoff, no LPS) "
                "-- not among the four documented UCI events; excluded "
                "and logged as OOC candidates (the second conservatively "
                "-- atypical demand not fully excludable).",
}


def stats_from_parquet(bw, exclude=None):
    """Robust per-feature stats over per-window baseline features."""
    keep = bw.copy()
    keep["t_end_dt"] = pd.to_datetime(keep["t_end"])
    if exclude:
        ex = pd.to_datetime(list(exclude))
        keep = keep[~keep["t_end_dt"].isin(ex)]
    out = {}
    for col in keep.select_dtypes(include="number").columns:
        v = keep[col].dropna()
        if len(v) == 0:
            continue
        out[col] = {"n": int(len(v)),
                    "median": round(float(v.median()), 4),
                    "p01": round(float(v.quantile(0.01)), 4),
                    "p99": round(float(v.quantile(0.99)), 4),
                    "min": round(float(v.min()), 4),
                    "max": round(float(v.max()), 4)}
    return out, keep


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    baselines = {}
    for tag, desc in TAGS:
        p = RUNS / f"baseline_{tag}.json"
        baselines[tag] = json.loads(p.read_text(encoding="utf-8"))
        baselines[tag]["description"] = desc

    vehicles = {}
    for tag, _ in TAGS:
        b = baselines[tag]
        bw = pd.read_parquet(RUNS / f"baseline_windows_{tag}.parquet")
        excl = GHOST_EXCLUSIONS.get(tag, [])
        f, kept = stats_from_parquet(bw, excl)
        d1 = max(2.0, round(f["oil_max"]["p99"] - f["oil_max"]["median"], 1))
        d2 = max(2.5, round(1.5 * f["oil_trend_c"]["p99"], 1))
        ol = f.get("oil_level_frac", {})
        ol_max = ol.get("max", 0.0)
        ol_med = ol.get("median", 0.0)
        polarity_suspect = bool(ol_med > 0.5)
        d3_enabled = (not polarity_suspect) and ol_max < 0.05
        vehicles[tag] = {
            "baseline_n_windows": int(len(kept)),
            "baseline_n_excluded": len(excl),
            "baseline_seed": b["seed"],
            "baseline_stats": f,
            "oil_max_p99": f["oil_max"]["p99"],
            "oil_max_p50": f["oil_max"]["median"],
            "tp2_p95_healthy_min": f["tp2_p95"]["min"],
            "thresholds": {
                "A1_mc_on_frac": max(0.90, round(
                    f["mc_on_frac"]["p99"] + 0.05, 3)),
                "A1_tp2_frac_ge9_5": 0.05,
                "A1b_mc_on_maxrun_min": 240.0,   # joint-FA-checked below
                "A2_lps_maxrun_min": max(60.0, round(
                    1.5 * f["lps_maxrun_min"]["max"], 0)),
                "A3_tp2_p95_max": round(f["tp2_p95"]["p01"] - 0.5, 2),
                "A3_mc_on_frac_min": 0.5,
                "D1_oil_elev_c": d1,
                "D2_oil_trend_c": d2,
                "D3_oil_level_frac": 0.5 if d3_enabled else None,
            },
            "oil_level_polarity": (
                "suspect_1_is_normal_or_chronic" if polarity_suspect
                else "documented_1_below_expected"),
            "oil_level_frac_healthy": {"median": ol_med, "max": ol_max},
            "score_features": [
                "tp2_mean", "tp2_p95", "mc_mean", "mc_frac_offload",
                "dv_frac_load", "oil_max", "oil_p95", "oil_mean",
                "h1_mean", "lps_frac", "lowpress_highload_min",
            ],
        }
        if excl:
            vehicles[tag]["ooc_ledger"] = {
                "excluded_window_ends": excl,
                "adjudication": GHOST_ADJUDICATION_NOTES.get(tag, ""),
            }

        # ---- joint zero-false-alarm check on the calibration sample ----
        bw = kept
        th = vehicles[tag]["thresholds"]
        fa_report = {"n_windows": int(len(bw)), "fa_air": None,
                     "fa_oil": None, "tightened": []}

        def air_fa(d, t):
            return int((((d["mc_on_frac"] >= t["A1_mc_on_frac"])
                         & (d["tp2_frac_ge9_5"] <= t["A1_tp2_frac_ge9_5"]))
                        | ((d["mc_on_maxrun_min"] >= t["A1b_mc_on_maxrun_min"])
                           & (d["tp2_frac_ge9_5"] <= t["A1_tp2_frac_ge9_5"]))
                        | (d["lps_maxrun_min"] >= t["A2_lps_maxrun_min"])
                        | ((d["tp2_p95"] <= t["A3_tp2_p95_max"])
                           & (d["mc_on_frac"] >= t["A3_mc_on_frac_min"]))
                        ).sum())

        def oil_fa(d, t, base_p99):
            elev = d["oil_max"] - base_p99
            return int(((elev >= t["D1_oil_elev_c"])
                        | (d["oil_trend_c"] >= t["D2_oil_trend_c"])
                        | ((t["D3_oil_level_frac"] is not None)
                           & (d["oil_level_frac"] >=
                              (t["D3_oil_level_frac"] or 1.0)))
                        ).sum())

        steps = 0
        while air_fa(bw, th) > 0 and steps < 20:
            steps += 1
            if steps % 3 == 1:
                th["A1_mc_on_frac"] = round(
                    min(1.0, th["A1_mc_on_frac"] + 0.005), 3)
            elif steps % 3 == 2:
                th["A1_tp2_frac_ge9_5"] = round(
                    max(0.0, th["A1_tp2_frac_ge9_5"] - 0.01), 3)
            else:
                th["A1b_mc_on_maxrun_min"] = round(
                    min(1440.0, th["A1b_mc_on_maxrun_min"] * 1.5), 0)
            th["A2_lps_maxrun_min"] = round(
                max(th["A2_lps_maxrun_min"],
                    1.05 * float(bw["lps_maxrun_min"].max())), 0)
            th["A3_tp2_p95_max"] = round(
                min(th["A3_tp2_p95_max"],
                    float(bw["tp2_p95"].min()) - 0.5), 2)
            vehicles[tag]["thresholds"] = th
        fa_report["fa_air"] = air_fa(bw, th)
        fa_report["tightened"] = (f"{steps} tightening steps" if steps
                                  else "none needed")
        fa_oil = oil_fa(bw, th, f["oil_max"]["p99"])
        if fa_oil > 0:
            # escalate oil thresholds on the actual violating windows
            viol = bw[(bw["oil_max"] - f["oil_max"]["p99"]
                       >= th["D1_oil_elev_c"])]
            if len(viol):
                th["D1_oil_elev_c"] = round(
                    float(viol["oil_max"].max()) - f["oil_max"]["p99"]
                    + 0.5, 1)
            vt = bw[bw["oil_trend_c"] >= th["D2_oil_trend_c"]]
            if len(vt):
                th["D2_oil_trend_c"] = round(
                    float(vt["oil_trend_c"].max()) + 0.5, 1)
            vehicles[tag]["thresholds"] = th
            fa_oil = oil_fa(bw, th, f["oil_max"]["p99"])
        fa_report["fa_oil"] = fa_oil
        vehicles[tag]["fa_check"] = fa_report
        if fa_report["fa_air"] > 0 or fa_oil > 0:
            raise SystemExit(
                f"ZERO-FA UNATTAINABLE for {tag} after tightening: "
                f"{fa_report} -- manual rule redesign required")

    spec = {
        "spec": "metropt_two_tier_v1",
        "version": "1.0",
        "frozen": str(date.today()),
        "preregistration": "runs/metropt3/PREREGISTRATION.md "
                           "(frozen 2026-09-21)",
        "provenance": {
            "operator_rules": "official UCI PDF attribute info (Data "
                              "Description_Metro.pdf) + Veloso 2022 Sci "
                              "Data: MPG<8.2bar, LPS<7bar, MC states "
                              "0/4/7/9A, DV_pressure=0 under load, "
                              "Oil_level=1 below expected",
            "failure_signatures": "W2 label forensics 2026-09-21 "
                                  "(label_check/W2_label_crossvalidation"
                                  ".md): acute air leak = LPS sustained + "
                                  "TP2 cannot build while MC continuous; "
                                  "oil leak = slow oil-temp climb to the "
                                  "80.2C class ceiling",
            "physical_ceilings": {
                "oil_overload_max_c": 97.9, "oil_degradation_max_c": 80.2,
                "note": "recur on both trains; sensor/physical limits -- "
                        "never thresholds; rules are band-relative by "
                        "design"},
            "threshold_policy": "zero false alarms on healthy calibration "
                                "sample + physics floor; see module "
                                "docstring",
            "fairness": "all systems receive the SAME frozen window "
                        "digest (metropt_features.py); only QCKUR-side "
                        "systems additionally receive bank+verdicts",
        },
        "rule_tier": {
            "air_leak_acute": {
                "fires_if": "A1: mc_on_frac >= A1_on AND tp2_frac_ge9_5 "
                            "<= A1_tp2 (continuous working duty while "
                            "pressure never reaches cutoff) OR A2: "
                            "lps_maxrun_min >= A2 (sustained low-pressure "
                            "warning) OR A3: tp2_p95 <= A3_tp2 AND "
                            "mc_on_frac >= A3_on (deep depression while "
                            "compressor working)",
                "design_note": "v2 duty-cycle physics: the 2020 healthy "
                               "sample falsified the naive lowpress-"
                               "highload rule (offload running MC~4A "
                               "holds TP2 low for hours in healthy "
                               "parked/low-demand states); the leak "
                               "signature is near-continuous WORKING "
                               "duty (MC>4.5A) with cutoff never reached",
                "evidence_fields": ["mc_on_frac", "mc_on_maxrun_min",
                                    "tp2_frac_ge9_5", "tp2_p95",
                                    "lps_maxrun_min", "lps_frac",
                                    "lps_first_before_end_h",
                                    "lowpress_highload_min"],
            },
            "oil_leak_degradation": {
                "fires_if": "oil_elev >= D1 (sustained elevation over the "
                            "vehicle baseline) OR oil_level_frac >= D3 "
                            "where enabled; D2 oil_trend_c fires a "
                            "mechanism-NEUTRAL oil_thermal_anomaly "
                            "(fast transients typically accompany air-"
                            "leak compressor overload, cf. the 97.9C "
                            "ceiling) and never labels oil degradation "
                            "on its own",
                "evidence_fields": ["oil_max", "oil_elev", "oil_trend_c",
                                    "oil_level_frac", "oil_p95"],
                "note": "oil_elev = window oil_max - vehicle baseline "
                        "oil_max_p99 (per-vehicle calibration; RSQ3/H6); "
                        "on train A the documented oil leak stays inside "
                        "the healthy band at digest level -- disclosed "
                        "boundary, not a hidden miss",
            },
            "precedence": "air_leak_acute then oil_leak_degradation then "
                          "oil_thermal_anomaly; air+oil both firing => "
                          "combined verdict (disclosed, never silently "
                          "merged)",
        },
        "score_tier": {
            "method": "robust z-scores of score_features vs the vehicle "
                      "baseline (median/p01/p99), z = max over features "
                      "of the two-sided deviation",
            "bands": {"monitor_z": 3.0, "suspect_z": 6.0},
            "note": "one-class deviation band, the analogue of AI4I's "
                    "probabilistic stratum; no supervised training (9 "
                    "events -- no fake supervision)",
        },
        "abstention": {
            "coverage_min": 0.8,
            "gap_max_h": 6.0,
            "outputs": "ABSTAIN_telemetry_hole | ABSTAIN_out_of_bank",
            "oob_rule": "any score_feature outside 3x its baseline "
                        "[p01,p99] span => out-of-bank flag",
        },
        "per_vehicle": vehicles,
        "calibration_files": {t: f"runs/metropt3/baseline_{t}.json"
                              for t, _ in TAGS},
    }
    out = DEST / "metropt_spec.json"
    out.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"spec frozen -> {out}")
    for tag, _ in TAGS:
        v = vehicles[tag]
        t = v["thresholds"]
        print(f"  {tag}: A1_on>={t['A1_mc_on_frac']} "
              f"A1_tp2<={t['A1_tp2_frac_ge9_5']} "
              f"A2_lps>={t['A2_lps_maxrun_min']} "
              f"A3_tp2<={t['A3_tp2_p95_max']} "
              f"D1={t['D1_oil_elev_c']} D2={t['D2_oil_trend_c']} "
              f"D3={t['D3_oil_level_frac']} "
              f"oil_polarity={v['oil_level_polarity']}")


if __name__ == "__main__":
    main()
