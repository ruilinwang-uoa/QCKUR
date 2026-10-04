"""Shared background knowledge, per-instance data context, and the canonical
5W1H KU-answer assembly used by the systems.

All systems receive identical background knowledge and identical per-instance
inputs (fairness protocol, paper Section 4.2.3). B8 additionally assembles one
representative knowledge-unit answer per 5W1H category, which it then passes to
the LLM for linguistic refinement.
"""
from __future__ import annotations

import json

import pandas as pd

import config as C
import ku_handlers as kh
import physics_rules as pr
import templates as Tpl

# one representative KU per 5W1H category -> (handler, args)
CANONICAL_KUS = [
    ("What", "h_fault_mode", {}),
    ("Why", "h_why_failure", {}),
    ("When", "h_intervention_needed", {}),
    ("Where", "h_locate_anomaly", {}),
    ("Who", "h_fault_responder", {}),
    ("How", "h_recommended_action", {}),
]

# lazily-loaded question bank for ku_id / question-text lookup
_BANK = None


def _bank():
    global _BANK
    if _BANK is None:
        _BANK = json.loads((C.QUESTION_BANK_DIR / "question_bank_5w1h.json")
                           .read_text(encoding="utf-8"))["questions"]
    return _BANK


def _lookup_ku(handler, args):
    for q in _bank():
        if q["handler"] == handler and q["args"] == args:
            return q
    return {"id": f"manual_{handler}", "text": handler, "category": "?",
            "template_id": handler}


def background_text(ctx: dict) -> str:
    q = ctx["quality"]
    rf = q.get("random_forest", {})
    return (
        "DATASET: AI4I 2020 Predictive Maintenance Dataset (10,000 milling-machine "
        "records; ~3.4% are failures, so the problem is highly imbalanced).\n"
        "FEATURES (per record): Type (quality variant L/M/H); Air temperature [K]; "
        "Process temperature [K]; Rotational speed [rpm]; Torque [Nm]; Tool wear [min].\n"
        "DERIVED: mechanical power [W] = torque * rpm * 2*pi/60; temperature difference "
        "[K] = process - air; overstrain [min*Nm] = tool_wear * torque.\n"
        "FAILURE MODES (engineering rules): TWF tool-wear failure (wear in [200,240) min "
        "probabilistic, >=240 min certain); HDF heat-dissipation failure "
        "((process-air) temp diff < 8.6 K AND rpm < 1380); PWF power failure (power outside "
        "[3500, 9000] W); OSF overstrain failure (tool_wear*torque exceeds the type limit "
        "L=11000, M=12000, H=13000); RNF random failure (stochastic, no deterministic rule).\n"
        f"FITTED CLASSIFIERS (70/15/15 split, 5 seeds): random_forest F1-macro "
        f"{rf.get('f1_macro')} AUPRC {rf.get('auprc')}; gradient_boosting and "
        f"logistic_regression also trained. Use these as the analytical models.\n"
        "FEATURE->COMPONENT MAP: rpm->spindle/drive; torque->tool region/drive train; "
        "tool_wear->cutting tool; air temp->cooling/ambient; process temp->process/workpiece.\n"
    )


def instance_context(row: dict) -> str:
    """Sensor-only description of one sample (no label leaked)."""
    sid = row.get("sample_id", f"UDI{row['UDI']}")
    power = pr.mechanical_power(row["Torque [Nm]"], row["Rotational speed [rpm]"])
    tdiff = pr.temp_difference(row["Air temperature [K]"], row["Process temperature [K]"])
    ov = pr.overstrain(row["Tool wear [min]"], row["Torque [Nm]"])
    return (
        f"Sample {sid} (UDI {row['UDI']}, Type {row['Type']}): "
        f"Air temp {row['Air temperature [K]']} K, Process temp {row['Process temperature [K]']} K, "
        f"Rotational speed {row['Rotational speed [rpm]']} rpm, Torque {row['Torque [Nm]']} Nm, "
        f"Tool wear {row['Tool wear [min]']} min. Derived: power {power:.0f} W, "
        f"temp diff {tdiff:.1f} K, overstrain {ov:.0f} min*Nm."
    )


def _rule_row(row: dict) -> dict:
    """A copy of the row whose failure-mode flags are set by the ENCODED PHYSICAL
    RULES (not the dataset labels). Used for the framework's report-time M, so the
    framework provably derives the fault from sensor values + rules rather than
    reading ground-truth labels. RNF is not rule-determinable (stays 0)."""
    pred = pr.predict_modes_from_physics(row)
    r = dict(row)
    r["TWF"] = 1 if pred["TWF_pred"] else 0
    r["HDF"] = 1 if pred["HDF_pred"] else 0
    r["PWF"] = 1 if pred["PWF_pred"] else 0
    r["OSF"] = 1 if pred["OSF_pred"] else 0
    r["RNF"] = 0  # random: not rule-derivable
    r["Machine failure"] = 1 if any(pred.values()) else 0
    return r


def canonical_5w1h(row: dict, ctx: dict, use_rules: bool = False) -> list[dict]:
    """Run one representative KU per 5W1H category on the instance.

    ``use_rules=False`` (default) -> uses the TRUE row, i.e. the verified facts
    / ground truth that Layer-2/3 judges check reports against.
    ``use_rules=True`` -> derives the fault from the encoded physical rules (the
    framework's report-time M, no oracle labels); used by B1/B8 generation.
    """
    if use_rules:
        row = _rule_row(row)
    out = []
    for category, handler, args in CANONICAL_KUS:
        q = _lookup_ku(handler, args)
        slots = kh.run_handler(handler, row, ctx, args)
        text = Tpl.render(q.get("template_id", handler), category, slots)
        out.append({"category": category, "ku_id": q["id"], "question": q["text"],
                    "handler": handler, "slots": slots, "answer": text})
    return out


def render_ku_backbone(kus: list[dict]) -> str:
    return "\n".join(f"[{k['category']}] {k['answer']}" for k in kus)


def build_ctx() -> dict:
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    full = pd.read_parquet(C.DATA_DIR / "ai4i_full_derived.parquet")
    return kh.build_ctx(sample, full)
