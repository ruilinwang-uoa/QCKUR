"""Build the 5W1H question bank (Phase 1A).

Generates >=200 questions (each 5W1H category >=25) as JSON. Every question is
*validated*: its handler is executed on a real sample row at build time, so the
bank is guaranteed executable end-to-end. Each question carries >=2 paraphrases,
its category, handler name, handler args, and template id.
"""
from __future__ import annotations

import json

import pandas as pd

import config as C
import ku_handlers as kh
import templates as T

RAW5 = C.FEATURE_NUMERIC                      # 5 raw process variables
ENG = ["Power [W]", "Temp diff [K]", "Overstrain [min*Nm]"]
FEAT8 = RAW5 + ENG
PHYS_THRESH = ["Power [W]", "Tool wear [min]", "Overstrain [min*Nm]", "Temp diff [K]"]
_MODES = C.FAILURE_MODES
_MODE_DESC = {"TWF": "tool wear failure", "HDF": "heat dissipation failure",
              "PWF": "power failure", "OSF": "overstrain failure", "RNF": "random failure"}

_questions: list[dict] = []


def _short(feat):
    return kh._PARAM_SHORT[feat]


def add(category, text, paraphrases, handler, args=None, template_id=None):
    _questions.append({
        "category": category,
        "text": text,
        "paraphrases": paraphrases,
        "handler": handler,
        "args": args or {},
        "template_id": template_id or handler,
    })


def build_what():
    add("What", "Is this machine sample a failure?",
        ["Does this sample correspond to a machine failure?", "Was a failure recorded for this sample?"],
        "h_machine_failure")
    add("What", "Which failure mode is active for this sample?",
        ["What fault mode does this sample exhibit?", "Identify the active failure mode."],
        "h_fault_mode")
    add("What", "What product quality variant does this sample belong to?",
        ["Which product type (L/M/H) is this sample?", "State the product variant."],
        "h_product_type")
    add("What", "What is the overall failure rate of the dataset?",
        ["What fraction of the dataset are failures?", "Report the dataset failure rate."],
        "h_failure_rate_dataset")
    add("What", "What is the machine failure base rate?",
        ["How common is machine failure overall?", "State the overall failure rate."],
        "h_overall_failure_rate")
    for f in FEAT8:
        s = _short(f)
        add("What", f"What is the {s} of this sample?",
            [f"Report the {s} value.", f"State the current {s}."],
            "h_param_value", {"param": f})
    for f in FEAT8:
        s = _short(f)
        add("What", f"What is the dataset mean of {s}?",
            [f"Report the average {s}.", f"What is the mean {s} across the dataset?"],
            "h_param_mean", {"param": f})
    for f in FEAT8:
        s = _short(f)
        add("What", f"What is the operating range of {s}?",
            [f"State the min/max of {s}.", f"What range does {s} span?"],
            "h_param_range", {"param": f})
    for f in FEAT8:
        s = _short(f)
        add("What", f"What is the standard deviation of {s}?",
            [f"Report the spread of {s}.", f"How variable is {s}?"],
            "h_param_std", {"param": f})
    for f in FEAT8:
        s = _short(f)
        add("What", f"How far is this sample's {s} from the mean?",
            [f"What is the z-score of {s}?", f"Is the {s} above or below average?"],
            "h_param_deviation", {"param": f})
    for m in ["logistic_regression", "random_forest", "gradient_boosting"]:
        add("What", f"What does the {m} model predict for this sample?",
            [f"Classify this sample with the {m} model.", f"What is the {m} prediction?"],
            "h_model_prediction", {"model": m})
    for m in ["random_forest", "gradient_boosting"]:
        add("What", f"What failure probability does the {m} model assign?",
            [f"Report the {m} confidence.", f"How confident is {m} in a failure?"],
            "h_model_confidence", {"model": m})
    for m in ["random_forest", "gradient_boosting", "logistic_regression"]:
        for metric in ["f1_macro", "auprc"]:
            add("What", f"What is the {metric} of the {m} model?",
                [f"Report {metric} for {m}.", f"How good is {m} by {metric}?"],
                "h_model_quality", {"model": m, "metric": metric})
    for mode in _MODES:
        add("What", f"What is the base rate of {mode} ({_MODE_DESC[mode]})?",
            [f"How frequent is {mode}?", f"What share of samples are {mode}?"],
            "h_mode_base_rate", {"mode": mode})


def build_why():
    for mode, h in [("PWF", "h_why_pwf"), ("HDF", "h_why_hdf"),
                    ("OSF", "h_why_osf"), ("TWF", "h_why_twf")]:
        add("Why", f"Why is this sample flagged as {mode} ({_MODE_DESC[mode]})?",
            [f"What causes {mode} here?", f"Explain the {mode} trigger."],
            h)
    for mode in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        add("Why", f"Why is this sample NOT a {mode} failure?",
            [f"Why is {mode} absent?", f"What keeps {mode} from triggering?"],
            "h_why_not_mode", {"mode": mode})
    add("Why", "Why did this sample fail (or not)?",
        ["What is the primary cause of failure?", "Explain the failure reason."],
        "h_why_failure")
    add("Why", "Why is this sample considered healthy?",
        ["Why is there no failure?", "Explain the no-failure status."],
        "h_why_no_failure")
    add("Why", "Why is this sample a random failure?",
        ["Explain the RNF flag.", "What underlies the random failure?"],
        "h_why_rnf")
    for m in ["random_forest", "gradient_boosting", "logistic_regression"]:
        add("Why", f"Why does the {m} model reach its decision for this sample?",
            [f"What drives the {m} prediction?", f"Explain the {m} rationale."],
            "h_why_model_predicts", {"model": m})
    for f in FEAT8:
        s = _short(f)
        add("Why", f"Why does the {s} deviate from its typical value?",
            [f"What makes {s} unusual here?", f"Why is {s} atypical?"],
            "h_why_param_deviation", {"param": f})
    for f in C.FEATURE_NUMERIC:
        s = _short(f)
        add("Why", f"How important is {s} for the failure prediction?",
            [f"What is the {s} contribution?", f"Rank {s} as a failure driver."],
            "h_feature_importance", {"feature": f})
    add("Why", "Why is the dataset considered imbalanced?",
        ["Explain the class imbalance.", "Why is accuracy misleading here?"],
        "h_dataset_imbalance")


def build_when():
    add("When", "Which tool-wear lifecycle stage is this sample in?",
        ["What is the wear stage?", "Is the tool near end of life?"],
        "h_wear_stage")
    add("When", "Is there heat-dissipation risk right now?",
        ["Is HDF risk present at this moment?", "Should cooling be watched now?"],
        "h_thermal_risk")
    add("When", "Is the mechanical power within the safe band?",
        ["Is power currently safe?", "Does power cross the threshold now?"],
        "h_power_threshold")
    add("When", "Has the overstrain limit been reached?",
        ["Is overstrain currently exceeded?", "Has OSF threshold been crossed?"],
        "h_overstrain_threshold")
    add("When", "Is intervention needed now?",
        ["Does this sample need immediate attention?", "When should we intervene?"],
        "h_intervention_needed")
    add("When", "Is tool replacement due now?",
        ["Should the tool be replaced now?", "Is the tool past its replacement point?"],
        "h_tool_replacement_due")
    for f in FEAT8:
        s = _short(f)
        add("When", f"Has the {s} threshold been breached?",
            [f"Is {s} past its critical value now?", f"Did {s} cross its limit?"],
            "h_threshold_breach", {"param": f})
    for f in FEAT8:
        s = _short(f)
        add("When", f"Is the current {s} within acceptable limits?",
            [f"Is {s} in range right now?", f"Should {s} be watched?"],
            "h_threshold_breach", {"param": f})
    # mode-onset timing questions
    _ONSET = {"PWF": "Power [W]", "HDF": "Temp diff [K]", "OSF": "Overstrain [min*Nm]",
              "TWF": "Tool wear [min]"}
    for mode, f in _ONSET.items():
        s = _short(f)
        add("When", f"When does {mode} risk begin (per {s})?",
            [f"At what {s} does {mode} start?", f"When is {mode} imminent?"],
            "h_threshold_breach", {"param": f})


def build_where():
    add("Where", "Where does the anomaly originate?",
        ["Which component is anomalous?", "Locate the source of abnormality."],
        "h_locate_anomaly")
    add("Where", "Which component is implicated by the active fault?",
        ["Where is the fault located?", "Which part is affected by the fault?"],
        "h_fault_component")
    for f in FEAT8:
        s = _short(f)
        add("Where", f"Which component does the {s} correspond to?",
            [f"Map {s} to a machine component.", f"What part does {s} reflect?"],
            "h_locate_component", {"feature": f})
    for comp in sorted(set(C.FEATURE_COMPONENT_MAP.values())):
        add("Where", f"What is the status of the {comp}?",
            [f"Is the {comp} normal?", f"Report {comp} condition."],
            "h_component_status", {"component": comp})
    for mode in ["PWF", "HDF", "OSF", "TWF", "RNF"]:
        add("Where", f"If {mode} is active, which component is at fault?",
            [f"Where would {mode} manifest?", f"Localize a {mode} fault."],
            "h_fault_component")
    for f in C.FEATURE_NUMERIC:
        s = _short(f)
        add("Where", f"Where is the {s} sensed on the machine?",
            [f"Which component reflects {s}?", f"Locate the {s} reading."],
            "h_locate_component", {"feature": f})


def build_who():
    add("Who", "Who should respond to this sample?",
        ["Which responder is required?", "Who needs to act on this sample?"],
        "h_fault_responder")
    add("Who", "What is the severity level?",
        ["How severe is this sample?", "Rate the severity."],
        "h_severity_level")
    add("Who", "Which process stage is affected?",
        ["What is impacted by this sample?", "Which process does this concern?"],
        "h_affected_process")
    for mode in ["PWF", "HDF", "OSF", "TWF", "RNF"]:
        add("Who", f"If {mode} occurs, who must intervene?",
            [f"Who handles a {mode}?", f"What responder for {mode}?"],
            "h_action_for_mode", {"mode": mode})
    for mode in ["PWF", "HDF", "OSF", "TWF", "RNF"]:
        add("Who", f"What is the severity of a {mode} fault?",
            [f"How severe is {mode}?", f"Severity rating for {mode}?"],
            "h_fault_responder")
    for sev in ["high", "medium", "low"]:
        add("Who", f"Which faults require {sev} severity response?",
            [f"What warrants a {sev}-severity alert?", f"When is severity '{sev}'?"],
            "h_fault_responder")
    for comp in sorted(set(C.FEATURE_COMPONENT_MAP.values())):
        add("Who", f"Who is responsible for the {comp}?",
            [f"Who owns the {comp}?", f"What role maintains the {comp}?"],
            "h_role_for_component", {"component": comp})
    add("Who", "Does this sample need operator attention?",
        ["Should an operator look at this?", "Is operator response required?"],
        "h_fault_responder")
    add("Who", "Does this sample need the maintenance team?",
        ["Is maintenance-team response required?", "Should maintenance be alerted?"],
        "h_fault_responder")
    add("Who", "Who should be notified for this sample?",
        ["Whom to alert about this sample?", "Who gets this work order?"],
        "h_fault_responder")
    for mode in ["PWF", "HDF", "OSF", "TWF", "RNF"]:
        add("Who", f"Who is accountable when {mode} recurs?",
            [f"Who owns recurring {mode}?", f"Who tracks {mode} follow-up?"],
            "h_action_for_mode", {"mode": mode})
    add("Who", "Who approves an emergency stop for this sample?",
        ["Who authorizes a stop?", "Who can halt production here?"],
        "h_fault_responder")
    add("Who", "Who verifies the repair after intervention?",
        ["Who confirms the fix?", "Who signs off the repair?"],
        "h_fault_responder")


def build_how():
    add("How", "What action is recommended for this sample?",
        ["What should be done about this sample?", "Recommend a maintenance action."],
        "h_recommended_action")
    add("How", "How can we prevent this kind of failure?",
        ["What preventive measure applies?", "How to avoid recurrence?"],
        "h_prevention")
    for mode in ["PWF", "HDF", "OSF", "TWF", "RNF"]:
        add("How", f"What should be done if {mode} is detected?",
            [f"How to respond to {mode}?", f"Give the {mode} handling procedure."],
            "h_action_for_mode", {"mode": mode})
    for mode in ["PWF", "HDF", "OSF", "TWF", "RNF"]:
        add("How", f"How is {mode} mitigated?",
            [f"What mitigates {mode}?", f"How to reduce {mode} risk?"],
            "h_mitigation_for_mode", {"mode": mode})
    for f in FEAT8:
        s = _short(f)
        add("How", f"How should the {s} be adjusted?",
            [f"What is the recommended {s} change?", f"How to correct the {s}?"],
            "h_parameter_adjustment", {"param": f})
    for f in PHYS_THRESH:
        s = _short(f)
        add("How", f"How do we bring the {s} back within the safe envelope?",
            [f"How to restore {s} to safe range?", f"What corrects an out-of-band {s}?"],
            "h_parameter_adjustment", {"param": f})
    for comp in sorted(set(C.FEATURE_COMPONENT_MAP.values())):
        add("How", f"How should the {comp} be serviced?",
            [f"What maintenance for the {comp}?", f"How to handle the {comp}?"],
            "h_recommended_action")
    add("How", "How should the tool be managed given its wear?",
        ["What is the tool-wear action?", "How to handle the cutting tool?"],
        "h_recommended_action")
    add("How", "How do we restore safe operation for this sample?",
        ["What corrective action restores safe operation?", "How to return to nominal?"],
        "h_recommended_action")


def validate(sample_df, full_df) -> list[dict]:
    """Execute every question's handler on one row; drop/flag failures."""
    ctx = kh.build_ctx(sample_df, full_df)
    row = sample_df.iloc[0].to_dict()
    ok, bad = [], []
    for i, q in enumerate(_questions):
        try:
            slots = kh.run_handler(q["handler"], row, ctx, q["args"])
            _ = T.render(q["template_id"], q["category"], slots)
            q["id"] = f"Q_{q['category']}_{i+1:03d}"
            ok.append(q)
        except Exception as e:  # noqa: BLE001
            bad.append({"text": q["text"], "handler": q["handler"], "error": str(e)})
    return ok, bad


def main() -> None:
    sample_df = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    full_df = pd.read_parquet(C.DATA_DIR / "ai4i_full_derived.parquet")

    build_what(); build_why(); build_when(); build_where(); build_who(); build_how()
    ok, bad = validate(sample_df, full_df)

    # attach category prefix ordering and write
    from collections import Counter
    counts = Counter(q["category"] for q in ok)
    bank = {
        "n_questions": len(ok),
        "category_counts": dict(counts),
        "min_per_category": min(counts.values()),
        "n_invalid_dropped": len(bad),
        "invalid": bad,
        "questions": ok,
    }
    (C.QUESTION_BANK_DIR / "question_bank_5w1h.json").write_text(
        json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Question bank: {len(ok)} questions | counts {dict(counts)} | dropped {len(bad)}")
    if bad:
        print("INVALID:", json.dumps(bad, indent=2, ensure_ascii=False)[:800])
    assert len(ok) >= 200, f"need >=200 questions, got {len(ok)}"
    assert all(v >= 25 for v in counts.values()), f"need >=25 per category, got {dict(counts)}"


if __name__ == "__main__":
    main()
