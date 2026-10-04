"""
metropt_knowledge.py -- question bank + KU inventory + report templates
for the MetroPT industrial case (QCKUR, attempt 4).

24 questions = 6 categories (5W1H) x 4. Every question carries:
  * verdict-stratified content (healthy / air / oil / both / monitor /
    abstain) -- wording mirrors the two-tier handler outcome
  * answer_source: which handler/feature/GT field fills it
  * epoch availability (Flowmeter/GPS questions are 2022-only)
  * numeric slots are L1-checkable against the frozen feature digest

Stratified action wording (bearing precedent): rule-tier indications ->
immediate action, maintenance team; score-tier monitor -> maintenance
window, technician; healthy -> routine, operator.

Outputs (code/knowledge/metropt/):
  question_bank_metropt.json   -- the 24 questions
  ku_inventory_metropt.json    -- one KU per question with T_template
  report_templates_metropt.json -- per-verdict report skeletons for B1/B8

Usage: python metropt_knowledge.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "knowledge" / "metropt"

EPOCHS_ALL = ["metropt3", "metropt2022", "metropt2022B"]
EPOCHS_2022 = ["metropt2022", "metropt2022B"]

Q = []


def q(qid, cat, name, text, source, epochs=EPOCHS_ALL, strat=None,
      numeric=False):
    Q.append({"id": qid, "category": cat, "short_name": name,
              "question": text, "answer_source": source,
              "epochs": epochs, "stratified": strat or {},
              "numeric": numeric})


# ---------------- WHAT --------------------------------------------------
q("W1", "What", "current_state",
  "What is the current operating state of the Air Production Unit?",
  "verdict.label + tier + confidence",
  strat={
      "air": "Acute air-leak indication (rule tier, high confidence)",
      "oil": "Oil-system degradation indication (rule tier, high "
             "confidence)",
      "both": "Combined air-leak and oil-degradation indications "
              "(rule tier)",
      "monitor": "Nominal operation with deviations worth monitoring "
                 "(score tier)",
      "healthy": "Nominal operation (score tier)",
      "abstain": "Not assessable from this window (telemetry gap or "
                 "out-of-bank mode)"})
q("W2", "What", "subsystem",
  "Which subsystem does the indication involve?",
  "verdict.label",
  strat={
      "air": "Compressed-air production and distribution (compressor "
             "unable to build pressure against continuous demand)",
      "oil": "Compressor oil system (temperature/degradation trend)",
      "both": "Air system with concurrent oil-system indication",
      "monitor": "Inconclusive -- deviation without a single "
                 "responsible subsystem",
      "healthy": "None",
      "abstain": "Unknown"})
q("W3", "What", "tier_confidence",
  "On which tier, and with what confidence, is the assessment made?",
  "verdict.tier + verdict.confidence",
  strat={
      "rule": "Rule tier (deterministic signature), confidence >= 0.9",
      "score": "Score tier (baseline deviation band), confidence per "
               "band",
      "abstain": "No tier -- abstained"})
q("W4", "What", "key_numbers",
  "What are the key measured values in this window?",
  "features: tp2_p95, tp2_frac_ge9_5, mc_on_frac, lps_maxrun_min, "
  "lps_frac, oil_max, oil_elev_c, oil_trend_c (flow_mean/max where "
  "available)", numeric=True)

# ---------------- WHY ----------------------------------------------------
q("Y1", "Why", "evidence",
  "What telemetry evidence supports the assessment?",
  "verdict.evidence.rule_branches_fired + numeric evidence fields",
  numeric=True)
q("Y2", "Why", "mechanism",
  "What physical mechanism explains the evidence?",
  "bank physics",
  strat={
      "air": "Air escapes as fast as the compressor pumps it: the "
             "compressor runs in working duty nearly continuously "
             "(Motor_current > 4.5 A) yet TP2 never reaches the "
             "10.2-10.5 bar load cutoff; sustained low pressure can "
             "activate the LPS warning (< 7 bar)",
      "oil": "Oil loss degrades lubrication/cooling so the oil "
             "temperature climbs above this vehicle's own healthy band "
             "(absolute thresholds do not transfer between vehicles)",
      "monitor": "Mild deviation from the vehicle baseline without a "
                 "definite mechanism",
      "healthy": "Measurements sit inside the vehicle baseline band",
      "abstain": "No mechanism claim -- window not assessable"})
q("Y3", "Why", "baseline_deviation",
  "By how much do the measurements deviate from this vehicle's "
  "baseline?",
  "spec per_vehicle baseline stats + features (oil_elev_c, tp2 vs "
  "baseline, mc_on vs typical cycling)", numeric=True)
q("Y4", "Why", "differential",
  "Why are alternative explanations less consistent?",
  "bank physics",
  strat={
      "air": "Healthy off-load running also holds TP2 low, but at "
             "Motor_current ~4 A with full tanks; heavy service shows "
             "fill-rest cycles that reach the cutoff; the observed "
             "continuous working duty without cutoff is the leak "
             "signature",
      "oil": "Hot ambient/duty days raise oil temperature but stay "
             "within the vehicle band and lack the sustained upward "
             "trend; Oil-level polarity is epoch-specific and is not "
             "used where its healthy state is ambiguous",
      "healthy": "No indication to explain", "abstain": "n/a"})

# ---------------- WHEN --------------------------------------------------
q("N1", "When", "action_window",
  "Within what timeframe should action be taken?",
  "verdict stratified action",
  strat={
      "air_rule": "Immediately: the operator requires detection at "
                  "least two hours before the unit becomes "
                  "non-operational",
      "oil_rule": "Within the next maintenance window (24-48 h); "
                  "earlier if the trend accelerates",
      "monitor": "At the next planned inspection",
      "healthy": "Routine schedule", "abstain": "n/a"})
q("N2", "When", "evidence_timeline",
  "When within the window did the critical evidence occur?",
  "lps_first_before_end_h, oil climb phase", numeric=True)
q("N3", "When", "lead_assessment",
  "If a failure is developing, how much lead time may remain?",
  "evidence onset + documented mean 6.8 h anomaly lead for air leaks "
  "(Veloso 2022)",
  strat={
      "air": "Air leaks in this fleet showed a mean 6.8 h anomaly lead "
             "before the maintenance event; sustained signatures leave "
             "hours, not days",
      "oil": "Oil degradation in this fleet developed over ~2-3 days "
             "before removal; the current elevation suggests days of "
             "margin at most",
      "healthy": "n/a", "abstain": "n/a"})
q("N4", "When", "recheck",
  "When should telemetry be re-checked?",
  "verdict stratified",
  strat={"air_rule": "Continuous monitoring until intervention",
         "oil_rule": "Every 12 h", "monitor": "Daily",
         "healthy": "Routine"})

# ---------------- WHERE -------------------------------------------------
q("L1", "Where", "system_location",
  "Where in the APU is the implicated subsystem?",
  "verdict.label",
  strat={"air": "Compressed-air circuit: compressor outlet, air-dryer "
                "towers, distribution to clients",
         "oil": "Compressor oil circuit", "healthy": "n/a"})
q("L2", "Where", "component_suspicion",
  "Which specific components are most suspect?",
  "bank",
  strat={"air": "Distribution pipes/valves feeding clients, or the "
                "air-dryer pneumatic pilot valve and drain lines",
         "oil": "Compressor oil seals / oil circuit connections",
         "healthy": "n/a"})
q("L3", "Where", "sensor_sources",
  "Which sensors carry the evidence?",
  "features used by fired branches",
  strat={"air": "TP2 (compressor pressure), Motor_current, LPS, "
                "DV_pressure/Towers (dryer duty)",
         "oil": "Oil_temperature (+ Oil_level where its polarity is "
                "unambiguous), Motor_current load share"})
q("L4", "Where", "vehicle_context",
  "Which vehicle/unit and duty context produced this window?",
  "epoch tag + duty_frac (2022 only)", epochs=EPOCHS_ALL)

# ---------------- WHO ---------------------------------------------------
q("R1", "Who", "responder",
  "Who should respond to this report?",
  "verdict stratified",
  strat={"air_rule": "Maintenance team, immediately",
         "oil_rule": "Maintenance technician with planner (schedule "
                     "the window)",
         "monitor": "Depot technician at next inspection",
         "healthy": "None beyond routine"})
q("R2", "Who", "notification",
  "Who else must be informed?",
  "bank",
  strat={"rule": "Fleet supervisor if removal from service is a "
                 "candidate (service impact)",
         "else": "No extra notification"})
q("R3", "Who", "decision_authority",
  "Who decides removal from service?",
  "bank",
  strat={"any": "Maintenance lead, per operator protocol; the report "
                "supports but does not make the decision"})
q("R4", "Who", "operator_now",
  "What should the train operator do right now?",
  "lps evidence",
  strat={"lps_active": "Heed the low-pressure warning; the unit may "
                       "become non-operational -- prepare to hand over "
                       "the vehicle",
         "no_lps": "Continue operation; no driver-facing warning is "
                   "active"})

# ---------------- HOW ---------------------------------------------------
q("A1", "How", "recommended_action",
  "What action does the report recommend?",
  "verdict stratified (decision-usefulness anchor)",
  strat={"air_rule": "Remove from service at the next safe "
                     "opportunity and locate the leak",
         "oil_rule": "Schedule compressor oil-system inspection within "
                     "24-48 h; remove from service if the trend "
                     "accelerates",
         "monitor": "Inspect at the next window; keep monitoring",
         "healthy": "No action; routine monitoring",
         "abstain": "No recommendation assessable -- request "
                    "re-transmission / re-check telemetry"})
q("A2", "How", "verification",
  "How can the indication be verified on site?",
  "bank",
  strat={"air": "Leak search on the distribution circuit (soap/gas "
                "detection), dryer drain-valve and pilot-valve check",
         "oil": "Oil level and leak inspection of the compressor oil "
                "circuit"})
q("A3", "How", "conservative_option",
  "What is the conservative fallback if verification is inconclusive?",
  "bank",
  strat={"air": "Treat as severe: remove from service (a blown client "
                "pipe in this fleet forced removal)",
         "oil": "Remove from service if oil temperature still exceeds "
                "the vehicle band at re-check"})
q("A4", "How", "followup",
  "What follow-up monitoring is required?",
  "verdict stratified",
  strat={"rule": "Post-repair verification window: confirm return to "
                 "the vehicle baseline band",
         "monitor": "Daily trend of the deviating quantity",
         "healthy": "Routine"})


TEMPLATES = {
    "healthy": ("CURRENT STATE: {W1}. KEY VALUES: {W4}. EVIDENCE: {Y1}; "
                "mechanism: {Y2}; baseline deviation: {Y3}. TIMELINE: "
                "{N2}; action window: {N1}; re-check: {N4}. SUBSYSTEM: "
                "{W2}; location: {L1}; sensors: {L3}; context: {L4}. "
                "RESPONDER: {R1}; operator: {R4}. RECOMMENDED ACTION: "
                "{A1}; verification: {A2}; fallback: {A3}; follow-up: "
                "{A4}."),
    "air": ("CURRENT STATE: {W1} -- {W3}. SUBSYSTEM: {W2} ({L1}); "
            "suspect components: {L2}. KEY VALUES: {W4}. EVIDENCE: {Y1}; "
            "mechanism: {Y2}; deviation from this vehicle's baseline: "
            "{Y3}; differential: {Y4}. LEAD TIME: {N3}; evidence "
            "timeline: {N2}; action window: {N1}; re-check: {N4}. "
            "SENSORS: {L3}; context: {L4}. RESPONDER: {R1}; inform: "
            "{R2}; decision authority: {R3}; operator now: {R4}. "
            "RECOMMENDED ACTION: {A1}; on-site verification: {A2}; "
            "conservative fallback: {A3}; follow-up: {A4}."),
    "oil": ("CURRENT STATE: {W1} -- {W3}. SUBSYSTEM: {W2} ({L1}); "
            "suspect components: {L2}. KEY VALUES: {W4}. EVIDENCE: {Y1}; "
            "mechanism: {Y2}; deviation from this vehicle's baseline: "
            "{Y3}; differential: {Y4}. LEAD TIME: {N3}; evidence "
            "timeline: {N2}; action window: {N1}; re-check: {N4}. "
            "SENSORS: {L3}; context: {L4}. RESPONDER: {R1}; inform: "
            "{R2}; decision authority: {R3}; operator now: {R4}. "
            "RECOMMENDED ACTION: {A1}; on-site verification: {A2}; "
            "conservative fallback: {A3}; follow-up: {A4}."),
    "monitor": ("CURRENT STATE: {W1} -- {W3}. KEY VALUES: {W4}. "
                "EVIDENCE: {Y1}; baseline deviation: {Y3}. TIMELINE: "
                "{N2}; action window: {N1}; re-check: {N4}. LOCATION: "
                "{L1}; context: {L4}. RESPONDER: {R1}. RECOMMENDED "
                "ACTION: {A1}; verification: {A2}; follow-up: {A4}."),
    "abstain": ("CURRENT STATE: {W1}. REASON: telemetry gap or "
                "out-of-bank operating mode not covered by the "
                "knowledge bank ({Y1} if partial data). RECOMMENDED "
                "ACTION: {A1}; re-check: {N4}."),
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    bank = {"bank": "metropt_v1", "n_questions": len(Q),
            "categories": ["What", "Why", "When", "Where", "Who", "How"],
            "spec": "code/knowledge/metropt/metropt_spec.json",
            "questions": Q}
    (DEST / "question_bank_metropt.json").write_text(
        json.dumps(bank, indent=1), encoding="utf-8")

    kus = []
    for item in Q:
        kus.append({
            "ku_id": f"KU_{item['id']}",
            "question_id": item["id"],
            "category": item["category"],
            "T_template": "{" + item["id"] + "}",
            "answer_source": item["answer_source"],
            "epochs": item["epochs"],
            "numeric": item["numeric"],
        })
    inv = {"inventory": "metropt_v1", "n_kus": len(kus),
           "bank_by_handler": "verdict.label selects the template; "
                              "handler evidence fields fill the slots",
           "kus": kus}
    (DEST / "ku_inventory_metropt.json").write_text(
        json.dumps(inv, indent=1), encoding="utf-8")

    (DEST / "report_templates_metropt.json").write_text(
        json.dumps({"templates": TEMPLATES}, indent=1), encoding="utf-8")
    print(f"question bank ({len(Q)} questions) -> {DEST}")
    print(f"KU inventory ({len(kus)}) -> {DEST}")
    print(f"report templates ({len(TEMPLATES)}) -> {DEST}")


if __name__ == "__main__":
    main()
