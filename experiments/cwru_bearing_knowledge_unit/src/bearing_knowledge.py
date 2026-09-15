"""Bearing instantiation of the question-centered KU framework (Phase 3).

Mirrors the AI4I knowledge assets (question_bank_5w1h.json / ku_inventory.json
/ ku_handlers.py / templates.py / background.canonical_5w1h) for the CWRU
12 kHz drive-end corpus, on top of the frozen v3 two-tier handler
(bearing_v3.py + physics_rules_spec_bearing.json).

Artifacts emitted to code/knowledge/:
  question_bank/question_bank_bearing.json   (24 questions, 6 per category)
  ku_inventory_bearing.json                  (same + T_template / M_component)

Runtime modes:
  --emit-bank   write the two JSON artifacts above
  --gt          build per-file 5W1H ground-truth answers -> code/runs/bearing_gt.json
  --smoke-b1    render template-only (B1) reports for three representative files

Report-time M (what B1/B8 generate from) = the v3 two-tier decision, no
oracle labels; GT mode uses the manifest. All numeric slots (rpm, expected
characteristic frequencies, band, prominences) are computed from the signal
and spec, so Layer-1 numeric checks need no labels either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KNOW = ROOT / "knowledge"
RUNS = ROOT / "runs"
SPEC = json.load(open(KNOW / "maps" / "physics_rules_spec_bearing.json",
                      encoding="utf-8"))
M_MULT = SPEC["defect_freq_multiples_x_fr"]

CHAR_NAME = {"BPFO": "outer-race (BPFO)", "BPFI": "inner-race (BPFI)",
             "BALL": "rolling-element (2xBSF)"}

LOC = {"IR": "the inner race of the drive-end bearing",
       "OR": "the outer race of the drive-end bearing (load-zone side)",
       "B": "the rolling elements of the drive-end bearing",
       "none": "no component (no fault detected)"}
ACTION = {
    "IR": "Stop the machine at the next safe opportunity, replace the "
          "drive-end bearing, and inspect the shaft seat for fretting or "
          "scoring.",
    "OR": "Replace the drive-end bearing and inspect the housing bore and "
          "outer-ring fit for wear or spalling debris.",
    "B": "Replace the drive-end bearing, check rolling-element and cage "
         "condition, and verify lubrication.",
    "none": "No action is required; continue routine vibration monitoring."}
RESPONDER = {"rule": "the maintenance team", "model": "a technician with the "
             "maintenance planner", "ground-truth": "the maintenance team",
             "none": "the operator (routine monitoring)"}
SEVERITY = {"rule": "high", "model": "elevated (model-tier detection)",
            "ground-truth": "high", "none": "none"}

# ---------------------------------------------------------------- bank defn
BANK: list[dict] = [
    # What
    ("What", "Does the vibration record indicate a bearing fault?",
     "h_bearing_status", {}, "h_bearing_status",
     "The bearing is assessed as {status} (verdict: {verdict})."),
    ("What", "Which bearing component is faulted?",
     "h_fault_component", {}, "h_fault_component",
     "The identified fault component is {component} ({tier} tier)."),
    ("What", "What is the shaft speed of this record?",
     "h_shaft_speed", {}, "h_shaft_speed",
     "The shaft rotates at {rpm} rpm ({fr} Hz), from the tachometer channel."),
    ("What", "What outer-race characteristic frequency applies at this speed?",
     "h_char_freq", {"which": "BPFO"}, "h_char_freq_bpfo",
     "At {fr} Hz shaft frequency the outer-race ball-pass frequency is "
     "{mult} x fr = {fc} Hz."),
    # Why
    ("Why", "Why was this verdict reached?",
     "h_why_verdict", {}, "h_why_verdict",
     "{reason}"),
    ("Why", "Which resonance band carries the diagnostic information?",
     "h_why_band", {}, "h_why_band",
     "The diagnostic band is {band} Hz (band kurtosis {kurt})."),
    ("Why", "By what margin does the evidence exceed the competing components?",
     "h_margin", {}, "h_margin",
     "The winning component's prominence is {ratio} times the strongest "
     "competing component ({winner} {win} vs {loser} {lose})."),
    ("Why", "Which detection threshold applies to this verdict?",
     "h_threshold_info", {"which": "BPFO"}, "h_threshold_info",
     "The applicable threshold is {which} prominence >= {tau} on the "
     "band-scanned envelope spectrum."),
    # When
    ("When", "Is intervention needed?",
     "h_intervention", {}, "h_intervention",
     "Intervention: {verdict}."),
    ("When", "How urgent is the intervention?",
     "h_urgency", {}, "h_urgency",
     "Urgency: {urgency}."),
    ("When", "When should the bearing be rechecked?",
     "h_next_check", {}, "h_next_check",
     "Recheck: {recheck}."),
    ("When", "Over how many harmonics does the evidence appear?",
     "h_harmonic_depth", {}, "h_harmonic_depth",
     "Characteristic-frequency evidence is summed over the first {kmax} "
     "harmonics with a {tol} Hz peak tolerance."),
    # Where
    ("Where", "Where is the fault located?",
     "h_fault_location", {}, "h_fault_location",
     "The fault is located in {location}."),
    ("Where", "How does the fault position relate to the load zone?",
     "h_load_zone", {}, "h_load_zone",
     "{zone}"),
    ("Where", "At which position was the vibration measured?",
     "h_sensor_position", {}, "h_sensor_position",
     "The accelerometer was mounted on the drive-end housing at the 12 "
     "o'clock position."),
    ("Where", "In which part of the spectrum does the signature appear?",
     "h_band_where", {}, "h_band_where",
     "The signature is demodulated from the {band} Hz resonance band."),
    # Who
    ("Who", "Who should respond to this report?",
     "h_responder", {}, "h_responder",
     "Responder: {responder} (severity {severity})."),
    ("Who", "How severe is the detected condition?",
     "h_severity", {}, "h_severity",
     "Severity: {severity}."),
    ("Who", "Is escalation to the maintenance planner required?",
     "h_escalation", {}, "h_escalation",
     "Escalation: {escalation}."),
    ("Who", "Who needs to act on a normal report?",
     "h_who_normal", {}, "h_who_normal",
     "For a normal report the operator continues routine monitoring; no "
     "specialist response is required."),
    # How
    ("How", "What action is recommended?",
     "h_action", {}, "h_action",
     "Recommended action: {action}"),
    ("How", "How should the repair be verified?",
     "h_verify_step", {}, "h_verify_step",
     "After replacement, repeat the envelope measurement and confirm the "
     "{freq} prominence falls below {tau}."),
    ("How", "How should monitoring continue afterwards?",
     "h_recheck", {}, "h_recheck",
     "Post-repair monitoring: {recheck}"),
    ("How", "How was the confidence of this verdict established?",
     "h_confidence_note", {}, "h_confidence_note",
     "{note}"),
]

CANONICAL = [("What", "h_fault_component"), ("Why", "h_why_verdict"),
             ("When", "h_intervention"), ("Where", "h_fault_location"),
             ("Who", "h_responder"), ("How", "h_action")]


def emit_bank() -> None:
    questions, kus = [], []
    counters: dict[str, int] = {}
    for cat, text, handler, args, tpl_id, tpl in BANK:
        counters[cat] = counters.get(cat, 0) + 1
        qid = f"Q_{cat}_{sum(counters.values()):03d}"
        questions.append({"category": cat, "text": text,
                          "paraphrases": [f"{text} (variant {counters[cat]})"],
                          "handler": handler, "args": args,
                          "template_id": tpl_id, "id": qid})
        kus.append({"ku_id": qid, "category": cat, "question": text,
                    "paraphrases": questions[-1]["paraphrases"],
                    "T_template_id": tpl_id, "T_template": tpl,
                    "M_handler": handler, "M_args": args,
                    "M_component": "envelope-analysis handler / spec constants"})
    out_q = KNOW / "question_bank" / "question_bank_bearing.json"
    out_q.write_text(json.dumps(
        {"n_questions": len(questions),
         "category_counts": counters, "questions": questions},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (KNOW / "ku_inventory_bearing.json").write_text(json.dumps(
        {"framework": "5W1H question-centered knowledge unit (bearing)",
         "n_kus": len(kus), "category_counts": counters,
         "kus": kus}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"emitted {len(questions)} KUs -> {out_q} + ku_inventory_bearing.json")


# ---------------------------------------------------------------- handlers
def _load_bank() -> dict[str, dict]:
    ku = json.load(open(KNOW / "ku_inventory_bearing.json", encoding="utf-8"))
    return {k["M_handler"] + json.dumps(k["M_args"], sort_keys=True): k
            for k in ku["kus"]}


BANK_BY_HANDLER: dict[str, dict] = {}


def _comp(row: dict) -> tuple[str, str]:
    return row["decision"]["pred"], row["decision"]["tier"]


def _strongest(row: dict, exclude: str) -> tuple[str, float]:
    f = row["features"]
    cands = {k: f[k] for k in ("BPFO", "BPFI", "BALLFAM") if k != exclude}
    k = max(cands, key=cands.get)
    return k, cands[k]


def h_bearing_status(row, ctx):
    pred, tier = _comp(row)
    status = "faulted" if pred != "normal" else "healthy"
    verdict = f"{pred}, {tier} tier" if pred != "normal" else "no fault"
    return {"status": status, "verdict": verdict}


def h_fault_component(row, ctx):
    pred, tier = _comp(row)
    return {"component": "no fault detected" if pred == "normal" else
            {"IR": "inner race (IR)", "OR": "outer race (OR)",
             "B": "rolling element (B)"}[pred],
            "tier": tier + " (normal)" if pred == "normal" else tier}


def h_shaft_speed(row, ctx):
    rpm = row["features"]["rpm"]
    return {"rpm": f"{rpm:.0f}", "fr": f"{rpm / 60:.2f}"}


def h_char_freq(row, ctx, which="BPFO"):
    fr = row["features"]["rpm"] / 60
    return {"fr": f"{fr:.2f}", "mult": M_MULT[which],
            "fc": f"{M_MULT[which] * fr:.1f}"}


def h_why_verdict(row, ctx):
    pred, tier = _comp(row)
    f, t = row["features"], ctx["thresholds"]
    if pred == "normal":
        return {"reason": f"No component's characteristic-frequency "
                f"prominence exceeds its threshold (strongest observed "
                f"{max(f['BPFO'], f['BPFI'], f['BALLFAM']):.1f} vs "
                f"thresholds {t['tau_or']:.0f}/{t['tau_ir']:.0f})."}
    if tier == "rule":
        key = "BPFI" if pred == "IR" else "BPFO"
        others = {"BPFI": ("BPFO", "BALLFAM"), "BPFO": ("BPFI", "BALLFAM")}[key]
        denom = max(f[others[0]], f[others[1]])
        return {"reason": f"Rule-tier evidence: {CHAR_NAME[key]} harmonic "
                f"prominence {f[key]:.0f} exceeds the threshold "
                f"{t['tau_ir' if pred == 'IR' else 'tau_or']:.0f} "
                f"(dominance ratio over competing components "
                f"{f[key] / denom:.1f})."}
    return {"reason": f"Model-tier evidence: the spectral classifier assigns "
            f"this record to {pred} (no characteristic-frequency rule "
            f"exceeded its threshold)."}


def h_why_band(row, ctx):
    f = row["features"]
    return {"band": f"{f['kurt_band_lo']:.0f}", "kurt": f"{f['kurt_band']:.1f}"}


def h_margin(row, ctx):
    pred, tier = _comp(row)
    f = row["features"]
    key = {"IR": "BPFI", "OR": "BPFO", "B": "BALLFAM"}.get(pred, "BPFI")
    win = f[key]
    loser_name, lose = _strongest(row, exclude=key)
    return {"winner": key, "win": f"{win:.0f}", "loser": loser_name,
            "lose": f"{lose:.0f}", "ratio": f"{win / lose:.1f}"}


def h_threshold_info(row, ctx, which="BPFO"):
    t = ctx["thresholds"]
    return {"which": which, "tau": f"{t['tau_ir' if which == 'BPFI' else 'tau_or']:.0f}"}


def h_intervention(row, ctx):
    pred, tier = _comp(row)
    v = ("immediate (rule-tier detection)" if tier == "rule" else
         "prompt inspection within the next maintenance window"
         if pred != "normal" else
         "no intervention; continue routine monitoring")
    return {"verdict": v}


def h_urgency(row, ctx):
    pred, tier = _comp(row)
    u = ("high -- rule-tier detection" if tier == "rule" else
         "elevated -- model-tier detection" if pred != "normal" else "none")
    return {"urgency": u}


def h_next_check(row, ctx):
    pred, _ = _comp(row)
    return {"recheck": "at the next scheduled monitoring cycle" if pred ==
            "normal" else "immediately after the recommended action"}


def h_harmonic_depth(row, ctx):
    return {"kmax": 8, "tol": 0.5}


def h_fault_location(row, ctx):
    pred, _ = _comp(row)
    return {"location": LOC["none" if pred == "normal" else pred]}


def h_load_zone(row, ctx):
    pred, _ = _comp(row)
    return {"zone": "Outer-race faults reported here are referenced to the "
            "load zone; the clock position cannot be confirmed from housing "
            "vibration alone." if pred == "OR" else
            "Not applicable: the load-zone reference applies to outer-race "
            "faults."}


def h_sensor_position(row, ctx):
    return {}


def h_band_where(row, ctx):
    return {"band": f"{row['features']['kurt_band_lo']:.0f}"}


def h_responder(row, ctx):
    pred, tier = _comp(row)
    tier = "none" if pred == "normal" else tier
    return {"responder": RESPONDER[tier], "severity": SEVERITY[tier]}


def h_severity(row, ctx):
    pred, tier = _comp(row)
    tier = "none" if pred == "normal" else tier
    return {"severity": SEVERITY[tier]}


def h_escalation(row, ctx):
    pred, _ = _comp(row)
    return {"escalation": "yes -- involve the maintenance planner" if pred !=
            "normal" else "not required"}


def h_who_normal(row, ctx):
    return {}


def h_action(row, ctx):
    pred, _ = _comp(row)
    return {"action": ACTION["none" if pred == "normal" else pred]}


def h_verify_step(row, ctx):
    pred, _ = _comp(row)
    which = {"IR": "BPFI", "OR": "BPFO", "B": "2xBSF",
             "normal": "BPFO", "none": "BPFO"}[pred]
    tau = ctx["thresholds"]["tau_ir" if which == "BPFI" else "tau_or"]
    return {"freq": which, "tau": f"{tau:.0f}"}


def h_recheck(row, ctx):
    return {"recheck": h_next_check(row, ctx)["recheck"]}


def h_confidence_note(row, ctx):
    pred, tier = _comp(row)
    return {"note": "Rule-tier verdicts rest on characteristic-frequency "
            "evidence with pre-registered thresholds; model-tier verdicts "
            "rest on the spectral classifier ensemble." if pred != "normal"
            else "Normal verdicts require no component prominence above "
            "threshold (conservative gate)."}


HANDLERS = {name: fn for name, fn in list(globals().items())
            if name.startswith("h_") and callable(fn)}


# ---------------------------------------------------------------- machinery
def load_v3() -> tuple[pd.DataFrame, dict]:
    """Reload the frozen v3 feature table + thresholds (+ RF for probs)."""
    df = pd.read_csv(RUNS.parent / "data" / "bearing_cwru" /
                     "bearing_v3_results.csv")
    cal = df[df.load_hp <= 1]
    tau_ir = float(np.sqrt(cal[cal.truth == "IR"].BPFI.min()
                           * cal[cal.truth != "IR"].BPFI.max()))
    om = (cal.truth == "OR") & (cal.defect_in != 0.014)
    tau_or = float(np.sqrt(cal[om].BPFO.min() * cal[~om].BPFO.max()))
    return df, {"tau_ir": tau_ir, "tau_or": tau_or, "tau_dom": 2.4}


def make_row(rec: pd.Series, thresholds: dict, gt: bool = False) -> dict:
    feats = {k: rec[k] for k in (
        "rpm", "BPFO", "BPFI", "BSF", "BALL", "FTF", "BALLFAM",
        "kurt_band", "kurt_band_lo")}
    if gt:
        # GT mode: What-component from the manifest label; the EVIDENCE tier
        # is a signal fact, taken from the deployed v3 verdict (identical to
        # the label on all 40 main files). Using a literal "ground-truth"
        # tier here made h_why_verdict fall through to the model-tier branch
        # and describe rule-tier signals falsely -- judges then penalized
        # CORRECT reports (diagnosed on B1/130.mat 2026-09-09).
        pred = rec.truth
        tier = rec.tier
    else:
        pred, tier = rec.pred, rec.tier
    return {"file": rec.file, "features": feats,
            "decision": {"pred": pred, "tier": tier}}


def canonical_5w1h(row: dict, ctx: dict) -> list[dict]:
    global BANK_BY_HANDLER
    if not BANK_BY_HANDLER:
        BANK_BY_HANDLER = _load_bank()
    out = []
    for category, handler in CANONICAL:
        q = BANK_BY_HANDLER.get(handler + "{}")
        slots = HANDLERS[handler](row, ctx)
        text = q["T_template"].format(**{**{k: "" for k in
                                            _slot_names(q["T_template"])},
                                         **slots})
        out.append({"category": category, "ku_id": q["ku_id"],
                    "question": q["question"], "handler": handler,
                    "slots": slots, "answer": text})
    return out


def _slot_names(tpl: str) -> list[str]:
    import string
    return [f for _, f, _, _ in string.Formatter().parse(tpl) if f]


def backbone(kus: list[dict]) -> str:
    return "\n".join(f"[{k['category']}] {k['answer']}" for k in kus)


def build_gt() -> None:
    df, thresholds = load_v3()
    ctx = {"thresholds": thresholds, "spec": SPEC}
    gt = {}
    for _, rec in df.iterrows():
        row = make_row(rec, thresholds, gt=True)
        kus = canonical_5w1h(row, ctx)
        gt[rec.file] = {"truth": rec.truth, "load_hp": int(rec.load_hp),
                        "defect_in": rec.defect_in, "kus": kus}
    out = RUNS / "bearing_gt.json"
    out.write_text(json.dumps(gt, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"GT for {len(gt)} files -> {out}")


BEARING_BACKGROUND = """BACKGROUND (drive-end bearing, test rig):
Bearing: SKF 6205-2RS JEM deep-groove ball bearing (9 balls, pitch 39.0 mm,
ball 7.94 mm, contact angle 3.134 deg). Vibration is measured by an
accelerometer on the drive-end housing at 12 o'clock, 12 kHz sampling.
Characteristic frequencies (multiples of shaft frequency fr): outer-race
BPFO 3.5848, inner-race BPFI 5.4152, rolling-element BSF 2.3574 / 2xBSF
4.7135, cage FTF 0.3983.
Detection handler (two-tier): the envelope spectrum of a bandpass-filtered
signal is scanned; a rule fires if BPFI or BPFO harmonic prominence exceeds
its pre-registered threshold with dominance over competing components
(rule tier); otherwise a random-forest spectral classifier decides
(model tier). Rule-tier verdicts carry characteristic-frequency evidence;
model-tier verdicts are less certain and trigger hedged recommendations.
"""

REPORT_TASK = (
    "Write a concise predictive-maintenance report for THIS record. In "
    "<=150 words, cover all six aspects: (1) What is the fault/status, "
    "(2) Why it occurs (the diagnostic evidence), (3) When to act / current "
    "stage, (4) Where the problem is located, (5) Who should respond, "
    "(6) How to address it. Use only the provided verified findings; do not "
    "invent numbers."
)


def smoke_b8(provider: str = "openai") -> None:
    import llm_call  # noqa: PLC0415 -- only needed for the B8 smoke
    df, thresholds = load_v3()
    ctx = {"thresholds": thresholds, "spec": SPEC}
    picks = {"130.mat": "OR rule-tier", "197.mat": "OR model-tier",
             "118.mat": "B model-tier", "97.mat": "normal"}
    for fname, label in picks.items():
        rec = df[df.file == fname].iloc[0]
        row = make_row(rec, thresholds)
        kus = canonical_5w1h(row, ctx)
        bb = backbone(kus)
        f = row["features"]
        inst = (f"INSTANCE: record {fname}, shaft speed {f['rpm']:.0f} rpm, "
                f"diagnostic band {f['kurt_band_lo']:.0f} Hz "
                f"(kurtosis {f['kurt_band']:.1f}).")
        prompt = (BEARING_BACKGROUND + "\n" + inst +
                  "\n\nVERIFIED ANALYTICAL FINDINGS (knowledge-unit answers; "
                  "use ONLY these facts):\n" + bb + "\n\nTASK:\n" + REPORT_TASK +
                  "\nThe findings are computed by the analytical components and "
                  "are authoritative; verbalize them into a fluent report "
                  "without changing any value.")
        res = llm_call.chat(provider, [
            {"role": "system",
             "content": "You are a predictive-maintenance report writer."},
            {"role": "user", "content": prompt}], seed=42)
        print(f"\n===== {fname} ({label}) — B8 refined "
              f"[{res['prompt_tokens']}+{res['completion_tokens']} tok, "
              f"{res['latency_s']:.1f}s] =====")
        print(res["text"])


def smoke_b1() -> None:
    df, thresholds = load_v3()
    ctx = {"thresholds": thresholds, "spec": SPEC}
    picks = {"130.mat": "OR rule-tier", "197.mat": "OR model-tier",
             "118.mat": "B model-tier", "97.mat": "normal"}
    for fname, label in picks.items():
        rec = df[df.file == fname].iloc[0]
        row = make_row(rec, thresholds)
        print(f"\n===== {fname} ({label}) — B1 template-only =====")
        print(backbone(canonical_5w1h(row, ctx)))


def main() -> None:
    global BANK_BY_HANDLER
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-bank", action="store_true")
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--smoke-b1", action="store_true")
    ap.add_argument("--smoke-b8", action="store_true")
    args = ap.parse_args()
    if args.emit_bank:
        emit_bank()
    BANK_BY_HANDLER = _load_bank()
    if args.gt:
        build_gt()
    if args.smoke_b1:
        smoke_b1()
    if args.smoke_b8:
        smoke_b8()


if __name__ == "__main__":
    main()
