"""
metropt_render.py -- prompt/report rendering for the MetroPT campaign.

Shared pieces:
  METROPT_BACKGROUND  public operator knowledge (official PDF semantics +
                      dataset paper facts) -- given to EVERY LLM system
  digest_text(feat)   the frozen window digest as text (B3/B5 context)
  slot_fill(...)      {W1..A4} template slots from handler verdict +
                      features (B1 template output, B8 verified findings)
  b2_sentences(...)   traditional D2T sentences (B2, no 5W1H battery)
  canonical_kus(...)  the 6-category canonical battery for B8

Fairness: B3 gets background + digest only; B5 adds primitive tools (no
verdicts, no baselines, no thresholds); B1/B2/B8 add the knowledge bank
and handler verdicts. REPORT_TASK is a fixed task spec for all.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOW = ROOT / "knowledge" / "metropt"

METROPT_BACKGROUND = """\
BACKGROUND -- Air Production Unit (APU) of a Porto metro train (public
dataset knowledge). The APU compressor feeds braking and door circuits.
Key signals: TP2 = compressor pressure (bar; load cutoff ~10.2-10.5);
H1 = reservoir-loop pressure; DV_pressure ~0 while the compressor works
under load; Motor_current (A): ~0 off, ~4 off-loaded running, ~7 under
load, ~9 starting; LPS activates when pressure drops below 7 bar (a
driver-facing warning); Oil_temperature (degC) on the compressor;
Oil_level is documented as active when oil is below expected; Towers
alternates the air-dryer towers; COMP/DV_eletric are intake/outlet
valve states; Flowmeter (2022 vehicles only) measures airflow to the
reservoirs. The operator requires detecting a developing failure at
least two hours before the unit becomes non-operational. Absolute
temperature thresholds differ between vehicles; oil-temperature bands
are vehicle-specific."""

REPORT_TASK = (
    "Write a concise predictive-maintenance report for THIS 24-hour "
    "telemetry window. In <=180 words, cover: (1) What is the current "
    "state of the unit, (2) Why -- the telemetry evidence, (3) When to "
    "act / timeframe, (4) Where -- which subsystem, (5) Who should "
    "respond, (6) How -- the recommended action. Use only the provided "
    "data; do not invent numbers. If the telemetry shows no actionable "
    "indication, say so plainly.")

# PROVENANCE NOTE (2026-09-25): during the 2026-09-21/22 campaign the
# digest_text flowmeter line contained a literal f-string placeholder
# ("{_fmt(...)}") and the flow/GPS lines printed "n/a" on 2020-epoch
# windows (NaN passes `is not None`); fixed post-campaign before the
# human-validation instrument was distributed. All systems received the
# IDENTICAL blemished digest, so campaign fairness and scores are
# unaffected; published reports/prompts in the repo archive reflect the
# as-run version.
VEHICLE_DESC = {
    "metropt3": "2020-epoch unit (10 s sampling, no Flowmeter/GPS)",
    "metropt2022": "train A, 2022 (1 Hz thinned, Flowmeter+GPS present)",
    "metropt2022B": "train B, 2022 (1 Hz, Flowmeter+GPS present)",
}

_LABEL_KEY = {
    "air_leak_indication": "air",
    "oil_degradation_indication": "oil",
    "air_and_oil_indication": "air",
    "oil_thermal_anomaly": "air",     # thermal text handled explicitly
    "monitor_deviation": "monitor",
    "healthy": "healthy",
    "ABSTAIN_telemetry_hole": "abstain",
    "ABSTAIN_out_of_bank": "abstain",
}

_STATE_TEXT = {
    "air_leak_indication":
        "Acute air-leak indication: the compressor runs in working duty "
        "without the pressure reaching the load cutoff",
    "oil_degradation_indication":
        "Oil-system degradation indication: oil temperature is elevated "
        "above this vehicle's healthy band",
    "air_and_oil_indication":
        "Combined air-leak and oil-degradation indications",
    "oil_thermal_anomaly":
        "Oil-temperature thermal anomaly: abnormal warming dynamics "
        "(typically compressor overload), without sustained elevation "
        "above the vehicle band",
    "monitor_deviation":
        "Nominal operation with deviations worth monitoring",
    "healthy": "Nominal operation",
    "ABSTAIN_telemetry_hole":
        "Not assessable: telemetry gap inside the window",
    "ABSTAIN_out_of_bank":
        "Not assessable: operating mode outside the knowledge bank",
}


def _fmt(x, nd=1, suffix=""):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.{nd}f}{suffix}"


def _has(x) -> bool:
    """Not-None AND not-NaN (the NaN-passes-`is not None` trap)."""
    return x is not None and x == x


def digest_text(feat: dict, vehicle: str) -> str:
    """The frozen digest as prompt text (no analysis, no baselines)."""
    fl = (f"Flowmeter mean {_fmt(feat.get('flow_mean'),1)} / max "
          f"{_fmt(feat.get('flow_max'),1)}. "
          if _has(feat.get("flow_mean")) else "")
    duty = (f"GPS duty fraction {_fmt(feat.get('duty_frac'),2)}. "
            if _has(feat.get("duty_frac")) else "")
    return (
        f"VEHICLE/UNIT: {VEHICLE_DESC[vehicle]}\n"
        f"WINDOW: 24 h ending {feat.get('t_end')} "
        f"(data coverage {_fmt((feat.get('coverage') or 0)*100,0,'%')}, "
        f"{feat.get('gaps_n')} gaps, longest {_fmt(feat.get('gap_max_h'),1,' h')}).\n"
        f"TELEMETRY DIGEST (frozen feature extract):\n"
        f"- TP2 compressor pressure: mean {_fmt(feat.get('tp2_mean'),2)} bar, "
        f"p95 {_fmt(feat.get('tp2_p95'),2)} bar, max {_fmt(feat.get('tp2_max'),2)} bar; "
        f"fraction of samples >= 9.5 bar: {_fmt((feat.get('tp2_frac_ge9_5') or 0)*100,1,'%')}.\n"
        f"- H1 reservoir-loop: mean {_fmt(feat.get('h1_mean'),2)}, "
        f"min {_fmt(feat.get('h1_min'),2)} bar. "
        f"DV_pressure > 0.5 fraction {_fmt((feat.get('dv_frac_load') or 0)*100,1,'%')}.\n"
        f"- Motor current: mean {_fmt(feat.get('mc_mean'),2)} A; state fractions "
        f"off {_fmt((feat.get('mc_frac_off') or 0)*100,0,'%')}, "
        f"off-loaded {_fmt((feat.get('mc_frac_offload') or 0)*100,0,'%')}, "
        f"under load {_fmt((feat.get('mc_frac_load') or 0)*100,0,'%')}, "
        f"starting {_fmt((feat.get('mc_frac_start') or 0)*100,0,'%')}; "
        f"fraction of time in working duty (>4.5 A): "
        f"{_fmt((feat.get('mc_on_frac') or 0)*100,1,'%')} "
        f"(longest continuous run {_fmt(feat.get('mc_on_maxrun_min'),0,' min')}).\n"
        f"- LPS low-pressure warning: active fraction "
        f"{_fmt((feat.get('lps_frac') or 0)*100,2,'%')}; longest continuous "
        f"activation {_fmt(feat.get('lps_maxrun_min'),0,' min')}. {fl}"
        f"- Oil temperature: mean {_fmt(feat.get('oil_mean'),1)} degC, "
        f"p95 {_fmt(feat.get('oil_p95'),1)}, max {_fmt(feat.get('oil_max'),1)}; "
        f"warming over the window "
        f"{_fmt(feat.get('oil_trend_c'),1,' degC')}.\n"
        f"- Digital signals (active fractions): COMP "
        f"{_fmt((feat.get('comp_frac') or 0)*100,0,'%')}, Towers "
        f"{_fmt((feat.get('towers_frac') or 0)*100,0,'%')}, Pressure_switch "
        f"{_fmt((feat.get('pswitch_frac') or 0)*100,0,'%')}, Oil_level "
        f"{_fmt((feat.get('oil_level_frac') or 0)*100,0,'%')}. {duty}")


def slot_fill(verdict: dict, feat: dict, spec: dict, vehicle: str) -> dict:
    """Template slots {W1..A4} from the handler verdict + digest."""
    lab = verdict["label"]
    tier = verdict["tier"]
    ev = verdict["evidence"]
    v = spec["per_vehicle"][vehicle]
    st = v["baseline_stats"]
    is_rule = tier == "rule"
    s: dict = {}

    s["W1"] = _STATE_TEXT[lab]
    s["W2"] = {"air_leak_indication":
                   "Compressed-air production and distribution (compressor "
                   "cannot build pressure against continuous demand)",
               "oil_degradation_indication": "Compressor oil system",
               "air_and_oil_indication":
                   "Air system with concurrent oil-system indication",
               "oil_thermal_anomaly":
                   "Compressor oil-temperature dynamics (overload pattern)",
               "monitor_deviation":
                   "Inconclusive: deviation without a single responsible "
                   "subsystem",
               "healthy": "None",
               }.get(lab, "Unknown")
    tier_txt = ("Rule tier (deterministic signature)" if is_rule
                else "Score tier (baseline deviation band)"
                if tier == "score" else "Abstained")
    s["W3"] = f"{tier_txt}; confidence {verdict['confidence']:.2f}"
    oil_elev = ev.get("oil_elev_c")
    s["W4"] = (
        f"TP2 p95 {_fmt(feat.get('tp2_p95'),2)} bar "
        f"({_fmt((feat.get('tp2_frac_ge9_5') or 0)*100,1,'%')} of samples "
        f">=9.5 bar); working-duty fraction "
        f"{_fmt((feat.get('mc_on_frac') or 0)*100,1,'%')} "
        f"(longest run {_fmt(feat.get('mc_on_maxrun_min'),0,' min')}); LPS "
        f"active {_fmt((feat.get('lps_frac') or 0)*100,2,'%')} "
        f"(longest {_fmt(feat.get('lps_maxrun_min'),0,' min')}); oil max "
        f"{_fmt(feat.get('oil_max'),1)} degC vs this vehicle's baseline "
        f"p99 {_fmt(v['oil_max_p99'],1)} degC "
        f"(elevation {_fmt(oil_elev,1,' degC')}, window warming "
        f"{_fmt(feat.get('oil_trend_c'),1,' degC')})")

    fired = ev.get("rule_branches_fired") or {}
    if is_rule:
        active = [k for k, on in fired.items() if on]
        s["Y1"] = ("Rule branches fired: " + (", ".join(active) if active
                    else "score excursion") + "; " + s["W4"])
    else:
        s["Y1"] = (f"No rule signature; score deviation "
                   f"z={ev.get('z_max')} on {ev.get('z_feature')}; " + s["W4"])
    s["Y2"] = {
        "air_leak_indication":
            "Air escapes as fast as the compressor pumps: near-continuous "
            "working duty while TP2 never reaches the 10.2-10.5 bar cutoff",
        "oil_degradation_indication":
            "Oil loss degrades lubrication/cooling so the temperature "
            "climbs above this vehicle's own healthy band",
        "oil_thermal_anomaly":
            "Heavy compressor duty heats the oil; fast warming without "
            "sustained elevation is an overload pattern, not necessarily "
            "an oil leak",
        "monitor_deviation":
            "Mild deviation from the vehicle baseline without a definite "
            "mechanism",
        "healthy": "Measurements sit inside the vehicle baseline band",
    }.get(lab, "n/a")
    s["Y3"] = (f"Oil max elevation over this vehicle's baseline p99: "
               f"{_fmt(oil_elev,1,' degC')} (baseline median "
               f"{_fmt(st['oil_max']['median'],1)} degC); TP2 p95 vs "
               f"baseline median {_fmt(st['tp2_p95']['median'],2)} bar: "
               f"{_fmt(feat.get('tp2_p95'),2)} bar; working-duty fraction "
               f"vs baseline median "
               f"{_fmt(st['mc_on_frac']['median']*100,1,'%')}: "
               f"{_fmt((feat.get('mc_on_frac') or 0)*100,1,'%')}")
    s["Y4"] = {
        "air_leak_indication":
            "Healthy off-load running also holds TP2 low, but at ~4 A with "
            "full tanks; heavy service shows fill-rest cycles that reach "
            "the cutoff; continuous working duty without cutoff is the "
            "leak signature",
        "oil_degradation_indication":
            "Hot duty days raise oil temperature but stay inside the "
            "vehicle band without sustained elevation",
    }.get(lab, "No indication to explain")

    strat_air = lab in ("air_leak_indication", "air_and_oil_indication",
                        "oil_thermal_anomaly")
    strat_oil = lab in ("oil_degradation_indication",
                        "air_and_oil_indication")
    s["N1"] = ("Immediately: the operator needs detection at least two "
               "hours before the unit becomes non-operational"
               if strat_air and is_rule else
               "Within the next maintenance window (24-48 h)"
               if strat_oil and is_rule else
               "At the next planned inspection" if tier == "score"
               else "Routine schedule")
    lps_h = feat.get("lps_first_before_end_h")
    s["N2"] = (f"LPS first activated {_fmt(lps_h,1,' h')} before the "
               f"window end" if lps_h is not None and lps_h == lps_h else
               "No LPS activation inside the window")
    s["N3"] = ("Air leaks in this fleet showed a mean 6.8 h anomaly lead "
               "before the maintenance event; sustained signatures leave "
               "hours, not days" if strat_air else
               "Oil degradation in this fleet developed over ~2-3 days "
               "before removal" if strat_oil else "n/a")
    s["N4"] = ("Continuous monitoring until intervention" if strat_air
               and is_rule else "Every 12 h" if strat_oil and is_rule
               else "Daily" if tier == "score" else "Routine")

    s["L1"] = ("Compressed-air circuit: compressor outlet, air-dryer "
               "towers, distribution to clients" if strat_air or
               lab == "oil_thermal_anomaly" else
               "Compressor oil circuit" if strat_oil else
               "No implicated subsystem")
    s["L2"] = ("Distribution pipes/valves feeding clients, or the "
               "air-dryer pneumatic pilot valve and drain lines"
               if strat_air else "Compressor oil seals / oil circuit"
               if strat_oil else "None")
    s["L3"] = {"air_leak_indication":
                   "TP2, Motor_current, LPS, DV_pressure/Towers",
               "oil_degradation_indication":
                   "Oil_temperature, Motor_current load share",
               "oil_thermal_anomaly": "Oil_temperature, Motor_current",
               }.get(lab, "All nominal inside baseline")
    s["L4"] = (f"{VEHICLE_DESC[vehicle]}; in-motion duty fraction "
               f"{_fmt((feat.get('duty_frac') or 0)*100,0,'%')}"
               if feat.get("duty_frac") is not None else
               f"{VEHICLE_DESC[vehicle]}; duty context not instrumented "
               f"in this epoch")

    s["R1"] = ("Maintenance team, immediately" if strat_air and is_rule
               else "Maintenance technician with planner"
               if strat_oil and is_rule else
               "Depot technician at next inspection" if tier == "score"
               else "None beyond routine")
    s["R2"] = ("Fleet supervisor if removal from service is a candidate"
               if is_rule else "No extra notification")
    s["R3"] = ("Maintenance lead decides removal from service; this "
               "report supports but does not make the decision")
    s["R4"] = ("Heed the low-pressure warning; the unit may become "
               "non-operational -- prepare to hand over the vehicle"
               if (feat.get("lps_maxrun_min") or 0) >= 10 else
               "Continue operation; no driver-facing warning is active")

    s["A1"] = ("Remove from service at the next safe opportunity and "
               "locate the leak" if strat_air and is_rule else
               "Schedule compressor oil-system inspection within 24-48 h; "
               "remove from service if the trend accelerates"
               if strat_oil and is_rule else
               "Inspect at the next window and keep monitoring"
               if tier == "score" and lab == "monitor_deviation"
               else "No action; routine monitoring" if tier == "score"
               else "No recommendation assessable: request re-transmission "
               "/ re-check telemetry")
    s["A2"] = ("Leak search on the distribution circuit (soap/gas "
               "detection), dryer drain-valve and pilot-valve check"
               if strat_air else "Oil level and leak inspection of the "
               "compressor oil circuit" if strat_oil else "n/a")
    s["A3"] = ("Treat as severe: remove from service (a blown client pipe "
               "in this fleet forced removal)" if strat_air else
               "Remove from service if oil temperature still exceeds the "
               "vehicle band at re-check" if strat_oil else "n/a")
    s["A4"] = ("Post-repair verification window: confirm return to the "
               "vehicle baseline band" if is_rule else
               "Daily trend of the deviating quantity" if tier == "score"
               else "Routine")
    return s


def b1_text(slots: dict, label: str) -> str:
    """Fill the per-verdict template skeleton with the slots."""
    tpl = json.loads((KNOW / "report_templates_metropt.json")
                     .read_text(encoding="utf-8"))["templates"]
    key = ("air" if label in ("air_leak_indication",
                              "air_and_oil_indication")
           else "oil" if label == "oil_degradation_indication"
           else "monitor" if label == "monitor_deviation"
           else "abstain" if label.startswith("ABSTAIN")
           else "healthy")
    return tpl[key].format(**slots)


def b2_sentences(verdict: dict, feat: dict, spec: dict, vehicle: str) -> str:
    """Traditional D2T: 3-4 canned sentences, no 5W1H battery."""
    lab = verdict["label"]
    v = spec["per_vehicle"][vehicle]
    parts = []
    if lab == "healthy":
        parts.append("The unit operates nominally.")
    elif lab == "monitor_deviation":
        parts.append("The unit operates with a deviation worth monitoring.")
    else:
        parts.append(f"The unit shows {_STATE_TEXT[lab].lower()}.")
    parts.append(
        f"TP2 p95 {_fmt(feat.get('tp2_p95'),2)} bar with "
        f"{_fmt((feat.get('tp2_frac_ge9_5') or 0)*100,1,'%')} of samples "
        f"reaching 9.5 bar; working-duty fraction "
        f"{_fmt((feat.get('mc_on_frac') or 0)*100,1,'%')}; oil max "
        f"{_fmt(feat.get('oil_max'),1)} degC (vehicle baseline p99 "
        f"{_fmt(v['oil_max_p99'],1)}).")
    if verdict["tier"] == "rule":
        parts.append("Action: immediate maintenance-team response.")
    elif verdict["tier"] == "score" and lab == "monitor_deviation":
        parts.append("Action: inspect at the next window.")
    else:
        parts.append("Action: continue routine monitoring.")
    return " ".join(parts)


def canonical_kus(slots: dict) -> list[dict]:
    """The canonical 6-category battery for B8 (and its claims trace)."""
    bank = json.loads((KNOW / "question_bank_metropt.json")
                      .read_text(encoding="utf-8"))
    order = {"What": 0, "Why": 1, "When": 2, "Where": 3, "Who": 4,
             "How": 5}
    kus = []
    for qd in sorted(bank["questions"], key=lambda q: (order[q["category"]],
                                                       q["id"])):
        kus.append({"ku_id": f"KU_{qd['id']}", "category": qd["category"],
                    "question": qd["question"],
                    "answer": slots[qd["id"]]})
    return kus
