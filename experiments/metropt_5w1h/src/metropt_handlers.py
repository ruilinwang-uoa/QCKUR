"""
metropt_handlers.py -- two-tier handler over the frozen window digest
(MetroPT case). Reads metropt_spec.json; returns a verdict dict used by
the question bank, the report templates, and the GT stratification.

Verdict schema:
  {label, tier, confidence, evidence: {...}, oob_flags: [...]}
  label in {healthy, air_leak_indication, oil_degradation_indication,
            air_and_oil_indication, monitor_deviation, ABSTAIN_*}
  tier   in {rule, score, abstain}
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "knowledge" / "metropt" / "metropt_spec.json"

LABELS = {
    "air": "air_leak_indication",
    "oil": "oil_degradation_indication",
    "thermal": "oil_thermal_anomaly",
    "both": "air_and_oil_indication",
    "monitor": "monitor_deviation",
    "healthy": "healthy",
}


def num(feat: dict, key: str, default: float) -> float:
    """Numeric feature lookup -- 0.0 is a legitimate value, only None/NaN
    fall back to the default (the falsy-zero bug caught in smoke)."""
    v = feat.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return default
    return float(v)


def load_spec(path: str | Path = SPEC) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _z(val, med, p01, p99) -> float:
    """Two-sided robust z against the baseline span."""
    if p99 <= p01:
        return 0.0
    span_up = p99 - med if p99 > med else 1e-9
    span_dn = med - p01 if med > p01 else 1e-9
    if val > med:
        return (val - med) / max(span_up / 3.0, 1e-9)
    return (med - val) / max(span_dn / 3.0, 1e-9)


def handle(feat: dict, vehicle: str, spec: dict | None = None) -> dict:
    spec = spec or load_spec()
    v = spec["per_vehicle"][vehicle]
    th = v["thresholds"]
    ev: dict = {}

    # ---- abstention -------------------------------------------------------
    cov = num(feat, "coverage", 0.0)
    if cov < spec["abstention"]["coverage_min"]:
        return {"label": "ABSTAIN_telemetry_hole", "tier": "abstain",
                "confidence": 1.0,
                "evidence": {"coverage": cov,
                             "gaps_n": feat.get("gaps_n"),
                             "gap_max_h": feat.get("gap_max_h")},
                "oob_flags": []}
    if num(feat, "gap_max_h", 0.0) > spec["abstention"]["gap_max_h"]:
        return {"label": "ABSTAIN_telemetry_hole", "tier": "abstain",
                "confidence": 1.0,
                "evidence": {"coverage": cov,
                             "gap_max_h": feat.get("gap_max_h")},
                "oob_flags": []}

    # ---- out-of-bank flags (abstention / OOC channel) ---------------------
    oob = _oob_flags(feat, v, spec)

    # ---- rule tier (v2 duty-cycle physics; see spec provenance) ----------
    a1a = (num(feat, "mc_on_frac", 0.0) >= th["A1_mc_on_frac"]
           and num(feat, "tp2_frac_ge9_5", 1.0) <= th["A1_tp2_frac_ge9_5"])
    a1b = (num(feat, "mc_on_maxrun_min", 0.0) >= th["A1b_mc_on_maxrun_min"]
           and num(feat, "tp2_frac_ge9_5", 1.0) <= th["A1_tp2_frac_ge9_5"])
    a2 = num(feat, "lps_maxrun_min", 0.0) >= th["A2_lps_maxrun_min"]
    a3 = (num(feat, "tp2_p95", 99.0) <= th["A3_tp2_p95_max"]
          and num(feat, "mc_on_frac", 0.0) >= th["A3_mc_on_frac_min"])
    air = a1a or a1b or a2 or a3
    oil_elev = round(num(feat, "oil_max", 0.0) - v["oil_max_p99"], 2)
    d3 = th.get("D3_oil_level_frac")
    # D1 sustained elevation / D3 low-oil signal = oil-leak degradation;
    # D2 fast trend = mechanism-NEUTRAL thermal anomaly (typically
    # air-leak overload heating, cf. the 97.9C ceiling) -- never labelled
    # oil degradation on trend alone
    oil = (oil_elev >= th["D1_oil_elev_c"]
           or (d3 is not None
               and num(feat, "oil_level_frac", 0.0) >= d3))
    thermal = num(feat, "oil_trend_c", 0.0) >= th["D2_oil_trend_c"]

    if air or oil or thermal:
        ev = {
            "rule_branches_fired": {"A1a_duty_frac_no_buildup": a1a,
                                    "A1b_duty_episode_no_buildup": a1b,
                                    "A2_sustained_lps": a2,
                                    "A3_deep_depression": a3,
                                    "D1_oil_elev": oil_elev >= th["D1_oil_elev_c"],
                                    "D2_thermal_trend": thermal,
                                    "D3_oil_level": (d3 is not None and
                                                     num(feat, "oil_level_frac", 0.0) >= d3)},
            "mc_on_frac": feat.get("mc_on_frac"),
            "mc_on_maxrun_min": feat.get("mc_on_maxrun_min"),
            "tp2_frac_ge9_5": feat.get("tp2_frac_ge9_5"),
            "tp2_p95": feat.get("tp2_p95"),
            "lowpress_highload_min": feat.get("lowpress_highload_min"),
            "lps_frac": feat.get("lps_frac"),
            "lps_maxrun_min": feat.get("lps_maxrun_min"),
            "lps_first_before_end_h": feat.get("lps_first_before_end_h"),
            "mc_frac_load": feat.get("mc_frac_load"),
            "mc_maxrun5_min": feat.get("mc_maxrun5_min"),
            "oil_max": feat.get("oil_max"),
            "oil_elev_c": oil_elev,
            "oil_trend_c": feat.get("oil_trend_c"),
            "oil_level_frac": feat.get("oil_level_frac"),
            "baseline_oil_max_p99": v["oil_max_p99"],
        }
        label = (LABELS["both"] if air and oil else
                 LABELS["air"] if air else
                 LABELS["oil"] if oil else LABELS["thermal"])
        return {"label": label, "tier": "rule", "confidence": 0.9,
                "evidence": ev, "oob_flags": oob}

    # out-of-bank with no rule fire: a mode the bank does not model
    if len(oob) >= 2:
        return {"label": "ABSTAIN_out_of_bank", "tier": "abstain",
                "confidence": 1.0,
                "evidence": {"oob_features": oob,
                             "oil_elev_c": oil_elev},
                "oob_flags": oob}

    # ---- score tier -------------------------------------------------------
    zmax, zfeat = 0.0, ""
    for key in v["score_features"]:
        val = feat.get(key)
        if val is None or isinstance(val, str):
            continue
        med = v["baseline_stats"][key]["median"]
        z = _z(val, med, v["baseline_stats"][key]["p01"],
               v["baseline_stats"][key]["p99"])
        if z > zmax:
            zmax, zfeat = z, key
    bands = spec["score_tier"]["bands"]
    if zmax >= bands["suspect_z"]:
        return {"label": LABELS["monitor"], "tier": "score",
                "confidence": 0.5,
                "evidence": {"z_max": round(zmax, 2), "z_feature": zfeat,
                             "oil_elev_c": oil_elev},
                "oob_flags": oob}
    if zmax >= bands["monitor_z"]:
        return {"label": LABELS["monitor"], "tier": "score",
                "confidence": 0.35,
                "evidence": {"z_max": round(zmax, 2), "z_feature": zfeat},
                "oob_flags": oob}
    return {"label": LABELS["healthy"], "tier": "score",
            "confidence": 0.85,
            "evidence": {"z_max": round(zmax, 2), "z_feature": zfeat},
            "oob_flags": oob}


def _oob_flags(feat: dict, v: dict, spec: dict) -> list:
    """Features outside 3x the baseline [p01,p99] span => out-of-bank."""
    flags = []
    for key in v["score_features"]:
        val = feat.get(key)
        st = v.get("baseline_stats", {}).get(key)
        if val is None or isinstance(val, str) or not st:
            continue
        span = st["p99"] - st["p01"]
        if span <= 0:
            continue
        if (val < st["p01"] - 3 * span) or (val > st["p99"] + 3 * span):
            flags.append(key)
    return flags
