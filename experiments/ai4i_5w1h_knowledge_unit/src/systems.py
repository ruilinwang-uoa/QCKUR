"""The eight compared systems (B1-B8).

Each system generates ONE maintenance report per instance under the unified task
("produce a PdM report covering What/Why/When/Where/Who/How"). All LLM systems
share the same provider/model, background knowledge and per-instance inputs
(fairness protocol). Result schema:

    {system_id, text, claims:[{category, value}], ku_trace:[...],
     prompt:str, llm_call:dict|None}

B1/B2 are local (no LLM). B3-B8 call the LLM via llm_call.chat. ``seed`` is
threaded to the LLM so multi-seed runs exhibit stochastic variance.
"""
from __future__ import annotations

import json

import physics_rules as pr
import llm_call
import background as bg
import ku_handlers as kh
import config as C

REPORT_TASK = (
    "Write a concise predictive-maintenance report for THIS sample. In <=150 words, "
    "cover all six aspects: (1) What is the fault/status, (2) Why it occurs, "
    "(3) When to act / current stage, (4) Where the problem is located, "
    "(5) Who should respond, (6) How to address it. Use only the provided data and "
    "background; do not invent numbers."
)

# ---- few-shot exemplar cache (B4) ----
_DEV_CACHE: list[dict] = []


def _dev_exemplars(ctx) -> list[dict]:
    if _DEV_CACHE:
        return _DEV_CACHE
    import pandas as pd
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    full = pd.read_parquet(C.DATA_DIR / "ai4i_full_derived.parquet")
    dev = full[~full["UDI"].isin(sample["UDI"])].head(100)
    chosen = []
    if (dev["Machine failure"] == 0).any():
        chosen.append(dev[dev["Machine failure"] == 0].iloc[0])
    for _, r in dev[dev["Machine failure"] == 1].head(2).iterrows():
        chosen.append(r)
    for r in chosen:
        row = r.to_dict()
        # GENERIC exemplar (plain PdM report from the rule-based D2T baseline),
        # NOT the framework's 5W1H-structured answer — keeps few-shot fair.
        rep = b2_traditional_d2t(row, ctx, None)["text"]
        _DEV_CACHE.append({"context": bg.instance_context(row), "report": rep})
    return _DEV_CACHE


# =========================================================================== #
# B1 Template-only (no LLM)
# =========================================================================== #
def b1_template_only(row, ctx, provider, retriever=None, seed=None):
    kus = bg.canonical_5w1h(row, ctx, use_rules=True)  # framework M = rule engine
    return {"system_id": "B1", "text": bg.render_ku_backbone(kus),
            "claims": [{"category": k["category"], "value": k["answer"]} for k in kus],
            "ku_trace": [k["ku_id"] for k in kus], "prompt": "", "llm_call": None}


# =========================================================================== #
# B2 Traditional D2T (no LLM, no KU)
# =========================================================================== #
def b2_traditional_d2t(row, ctx, provider, retriever=None, seed=None):
    parts = []
    # rule-based content determination (hand-crafted rules, no oracle labels)
    rrow = bg._rule_row(row)
    modes = [m for m in ["TWF", "HDF", "PWF", "OSF", "RNF"] if rrow[m] == 1]
    p = pr.power_diagnosis(row)
    ws = pr.wear_stage(row["Tool wear [min]"])
    if not modes:
        parts.append(f"The unit shows no active failure mode. Operating power is {p['power_w']:.0f} W ({p['status']}).")
    else:
        parts.append(f"A {('/'.join(modes))} condition is present on the unit.")
        if "PWF" in modes:
            parts.append(f"Power is {p['power_w']:.0f} W, outside the 3500-9000 W range.")
        if "HDF" in modes:
            parts.append("Temperature difference is low at reduced speed, pointing to cooling.")
        if "OSF" in modes:
            parts.append("Tool-wear and torque load exceed the allowed overstrain.")
        if "TWF" in modes:
            parts.append(f"Tool wear is {row['Tool wear [min]']} min, at the end of life.")
    parts.append(f"Tool-wear stage: {ws['stage']}.")
    if modes:
        m0 = next((x for x in ["HDF", "PWF", "OSF", "TWF", "RNF"] if x in modes), "RNF")
        fa = C.FAULT_ACTION_MAP[m0]
        parts.append(f"Action: {fa['action']}; assign to {fa['responder']} (severity {fa['severity']}).")
    else:
        parts.append("Action: continue routine monitoring (operator).")
    return {"system_id": "B2", "text": " ".join(parts), "claims": [],
            "ku_trace": [], "prompt": "", "llm_call": None}


def _common_prefix(row, ctx):
    return bg.background_text(ctx) + "\nINSTANCE:\n" + bg.instance_context(row) + "\n"


# =========================================================================== #
# B3 Direct GPT-4
# =========================================================================== #
def b3_direct(row, ctx, provider, retriever=None, seed=None):
    prompt = _common_prefix(row, ctx) + "\nTASK:\n" + REPORT_TASK
    res = llm_call.chat(provider, [{"role": "system", "content": "You are a predictive-maintenance analyst."},
                                   {"role": "user", "content": prompt}], seed=seed)
    return {"system_id": "B3", "text": res["text"], "claims": [], "ku_trace": [],
            "prompt": prompt, "llm_call": res}


# =========================================================================== #
# B4 Few-shot GPT-4
# =========================================================================== #
def b4_fewshot(row, ctx, provider, retriever=None, seed=None):
    messages = [{"role": "system", "content": "You are a predictive-maintenance analyst. Follow the style of the examples."}]
    for e in _dev_exemplars(ctx):
        messages.append({"role": "user", "content": "INSTANCE:\n" + e["context"] + "\nTASK:\n" + REPORT_TASK})
        messages.append({"role": "assistant", "content": e["report"]})
    prompt = _common_prefix(row, ctx) + "\nTASK:\n" + REPORT_TASK
    messages.append({"role": "user", "content": prompt})
    res = llm_call.chat(provider, messages, seed=seed)
    return {"system_id": "B4", "text": res["text"], "claims": [], "ku_trace": [],
            "prompt": prompt, "llm_call": res}


# =========================================================================== #
# B5 Tool-using agent (function calling) — INDEPENDENT primitive tools only.
# The agent gets raw analytical primitives and must assemble the report itself;
# it does NOT receive the framework's assembled 5W1H answer (that would leak the
# structure being tested). This keeps B5 a fair "no-KU-structure" baseline.
# =========================================================================== #
_QTY_ENUM = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
             "Torque [Nm]", "Tool wear [min]", "Power [W]", "Temp diff [K]", "Overstrain [min*Nm]"]

_PRIMITIVE_TOOLS = [
    {"type": "function", "function": {
        "name": "get_parameter", "description": "Get the value of one sensor or derived "
        "quantity for the current sample.",
        "parameters": {"type": "object", "properties": {"quantity": {"type": "string", "enum": _QTY_ENUM}},
                       "required": ["quantity"]}}},
    {"type": "function", "function": {
        "name": "get_failure_prediction", "description": "Get the random-forest classifier's "
        "failure prediction and probability for the current sample.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "compute_power", "description": "Compute the mechanical power and whether it "
        "is inside the safe [3500, 9000] W band.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "check_rule", "description": "Check whether a specific failure-mode rule is "
        "triggered for the current sample.",
        "parameters": {"type": "object",
                       "properties": {"mode": {"type": "string", "enum": ["TWF", "HDF", "PWF", "OSF", "RNF"]}},
                       "required": ["mode"]}}},
]


def _exec_primitive(name, args, row, ctx):
    args = args or {}
    if name == "get_parameter":
        q = args.get("quantity")
        if q in row:
            return {"quantity": q, "value": round(float(row[q]), 2)}
        return {"error": f"unknown quantity '{q}'"}
    if name == "get_failure_prediction":
        p = kh._pos(row, ctx)
        return {"predicted": "failure" if ctx["pred_lookup"]["random_forest"]["pred"][p] else "no failure",
                "probability": ctx["pred_lookup"]["random_forest"]["prob"][p]}
    if name == "compute_power":
        pdiag = pr.power_diagnosis(row)
        return {"power_w": pdiag["power_w"], "status": pdiag["status"]}
    if name == "check_rule":
        mode = args.get("mode")
        if mode in ("TWF", "HDF", "PWF", "OSF"):
            trig = pr.predict_modes_from_physics(row)[f"{mode}_pred"]
            return {"mode": mode, "triggered": bool(trig)}
        return {"mode": mode, "triggered": None, "note": "RNF is random, not rule-derivable"}
    return {"error": f"unknown tool '{name}'"}


def _acc(total, res):
    total["prompt_tokens"] += res.get("prompt_tokens", 0)
    total["completion_tokens"] += res.get("completion_tokens", 0)
    total["latency_s"] += res.get("latency_s", 0.0)
    total["cost_usd"] += res.get("cost_usd", 0.0)


def b5_agent(row, ctx, provider, retriever=None, seed=None):
    messages = [
        {"role": "system", "content": "You are a tool-using predictive-maintenance agent. "
         "Inspect the sample with the available tools, then write the report yourself from "
         "what you found."},
        {"role": "user", "content": _common_prefix(row, ctx) + "\nTASK:\n" + REPORT_TASK},
    ]
    total = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "cost_usd": 0.0}
    text, error = "", None
    try:
        res = llm_call.chat(provider, messages, tools=_PRIMITIVE_TOOLS,
                            tool_choice="auto", seed=seed)
        _acc(total, res)
        if res.get("tool_calls"):
            messages.append({"role": "assistant", "content": res["text"], "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["args"]}}
                for tc in res["tool_calls"]]})
            for tc in res["tool_calls"]:
                try:
                    pargs = json.loads(tc["args"]) if tc["args"] else {}
                except Exception:  # noqa: BLE001
                    pargs = {}
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(_exec_primitive(tc["name"], pargs, row, ctx),
                                                       ensure_ascii=False)})
            messages.append({"role": "user", "content":
                "Using the tool results above, write the final maintenance report now. "
                "Do not call any more tools."})
            res2 = llm_call.chat(provider, messages, seed=seed)  # no tools -> final answer
            _acc(total, res2)
            text, error = res2["text"], (None if res2["ok"] else res2["error"])
        else:
            text, error = res["text"], (None if res["ok"] else res["error"])
    except Exception as e:  # noqa: BLE001  (provider may not support tools)
        error = f"{type(e).__name__}: {e}"
    merged = {"ok": error is None, "text": text, "tool_calls": [],
              "prompt_tokens": total["prompt_tokens"], "completion_tokens": total["completion_tokens"],
              "latency_s": round(total["latency_s"], 3), "cost_usd": round(total["cost_usd"], 6),
              "model": llm_call.llm_config.get_provider(provider).model, "error": error}
    return {"system_id": "B5", "text": text, "claims": [], "ku_trace": [],
            "prompt": messages[1]["content"], "llm_call": merged}


# =========================================================================== #
# B6 RAG
# =========================================================================== #
def b6_rag(row, ctx, provider, retriever=None, seed=None):
    query = bg.instance_context(row) + " failure cause action component responder"
    top = retriever.query(query, k=5) if retriever else []
    context = "\n".join(f"[{t['title']}] {t['text']}" for t in top)
    prompt = (bg.background_text(ctx) + "\nINSTANCE:\n" + bg.instance_context(row) +
              "\n\nRETRIEVED KNOWLEDGE (domain corpus; reference only):\n" + context +
              "\n\nTASK:\n" + REPORT_TASK)
    res = llm_call.chat(provider, [{"role": "system", "content": "You are a predictive-maintenance analyst using the retrieved knowledge."},
                                   {"role": "user", "content": prompt}], seed=seed)
    return {"system_id": "B6", "text": res["text"], "claims": [], "ku_trace": [],
            "prompt": prompt, "llm_call": res}


# =========================================================================== #
# B7 RAG + Chain-of-Verification
# =========================================================================== #
def b7_rag_cove(row, ctx, provider, retriever=None, seed=None):
    draft = b6_rag(row, ctx, provider, retriever, seed=seed)
    if not draft["llm_call"]["ok"]:
        draft["system_id"] = "B7"
        return draft
    vq = llm_call.chat(provider, [{"role": "user", "content": (
        "Here is a draft maintenance report. Write 3 short verification questions that "
        "check its key claims, as a JSON object {\"questions\":[...]}.\nDRAFT:\n" + draft["text"])}],
        response_format={"type": "json_object"}, seed=seed)
    # self-verification (faithful CoVe): the model answers its own verification
    # questions from the instance data + background, with NO external oracle.
    ans = llm_call.chat(provider, [{"role": "user", "content": (
        "Answer each verification question using ONLY the instance data and background "
        "knowledge below (no external source).\nQUESTIONS:\n" + (vq["text"] or "{}")
        + "\nINSTANCE:\n" + bg.instance_context(row)
        + "\nBACKGROUND:\n" + bg.background_text(ctx))}], seed=seed)
    rev = llm_call.chat(provider, [{"role": "user", "content": (
        "Revise the draft report to be consistent with the verification answers. Keep <=150 "
        "words and the six aspects.\nDRAFT:\n" + draft["text"] +
        "\nVERIFICATION ANSWERS:\n" + ans["text"])}], seed=seed)
    merged = dict(draft["llm_call"])
    for r in (vq, ans, rev):
        merged["prompt_tokens"] += r.get("prompt_tokens", 0)
        merged["completion_tokens"] += r.get("completion_tokens", 0)
        merged["latency_s"] += r.get("latency_s", 0.0)
        merged["cost_usd"] += r.get("cost_usd", 0.0)
    merged.update({"text": rev["text"], "latency_s": round(merged["latency_s"], 3),
                   "cost_usd": round(merged["cost_usd"], 6)})
    return {"system_id": "B7", "text": rev["text"], "claims": [], "ku_trace": [],
            "prompt": draft["prompt"], "llm_call": merged}


# =========================================================================== #
# B8 Full Framework (5W1H KU backbone + LLM refinement)
# =========================================================================== #
def b8_framework(row, ctx, provider, retriever=None, seed=None):
    kus = bg.canonical_5w1h(row, ctx, use_rules=True)  # framework M = rule engine
    backbone = bg.render_ku_backbone(kus)
    prompt = (bg.background_text(ctx) + "\nINSTANCE:\n" + bg.instance_context(row) +
              "\n\nVERIFIED ANALYTICAL FINDINGS (knowledge-unit answers; use ONLY these facts):\n"
              + backbone + "\n\nTASK:\n" + REPORT_TASK +
              "\nThe findings are computed by the analytical components and are authoritative; "
              "verbalize them into a fluent report without changing any value.")
    res = llm_call.chat(provider, [{"role": "system", "content": "You are a predictive-maintenance report writer."},
                                   {"role": "user", "content": prompt}], seed=seed)
    return {"system_id": "B8", "text": res["text"],
            "claims": [{"category": k["category"], "value": k["answer"]} for k in kus],
            "ku_trace": [k["ku_id"] for k in kus], "prompt": prompt, "llm_call": res}


# =========================================================================== #
# Ablations (RQ5): each removes ONE component from B8 to quantify its contribution.
#   AblT - answer templates  : facts given as raw key:value, LLM verbalizes freely
#   AblM - model handler     : no precomputed facts; LLM derives from sensors
#   AblQ - question matching : facts given as a flat unstructured dump
#   AblP - physical rules    : classifier-only fault signal (no encoded rules)
# =========================================================================== #
def ablation_T(row, ctx, provider, retriever=None, seed=None):
    kus = bg.canonical_5w1h(row, ctx, use_rules=True)
    raw = "\n".join(f"{k['category']}: " + json.dumps(k["slots"], ensure_ascii=False) for k in kus)
    prompt = (bg.background_text(ctx) + "\nINSTANCE:\n" + bg.instance_context(row) +
              "\n\nVERIFIED ANALYTICAL RESULTS (raw key:value; verbalize them freely):\n" + raw +
              "\n\nTASK:\n" + REPORT_TASK)
    res = llm_call.chat(provider, [{"role": "system", "content": "You are a predictive-maintenance report writer."},
                                   {"role": "user", "content": prompt}], seed=seed)
    return {"system_id": "AblT", "text": res["text"], "claims": [], "ku_trace": [],
            "prompt": prompt, "llm_call": res}


def ablation_M(row, ctx, provider, retriever=None, seed=None):
    prompt = (bg.background_text(ctx) + "\nINSTANCE:\n" + bg.instance_context(row) +
              "\n\nTASK:\nWrite the predictive-maintenance report covering all six aspects "
              "(What/Why/When/Where/Who/How). Derive the failure mode, the causes, and ALL "
              "numeric values YOURSELF from the sensor data above; no precomputed analytical "
              "answers are provided.")
    res = llm_call.chat(provider, [{"role": "system", "content": "You are a predictive-maintenance analyst."},
                                   {"role": "user", "content": prompt}], seed=seed)
    return {"system_id": "AblM", "text": res["text"], "claims": [], "ku_trace": [],
            "prompt": prompt, "llm_call": res}


def ablation_Q(row, ctx, provider, retriever=None, seed=None):
    kus = bg.canonical_5w1h(row, ctx, use_rules=True)
    flat = "; ".join(k["answer"] for k in kus)        # facts present, but not aspect-organized
    prompt = (bg.background_text(ctx) + "\nINSTANCE:\n" + bg.instance_context(row) +
              "\n\nVERIFIED FACTS (reference; not organized by aspect):\n" + flat +
              "\n\nTASK:\nProduce a maintenance report for this sample.")
    res = llm_call.chat(provider, [{"role": "system", "content": "You are a predictive-maintenance report writer."},
                                   {"role": "user", "content": prompt}], seed=seed)
    return {"system_id": "AblQ", "text": res["text"], "claims": [], "ku_trace": [],
            "prompt": prompt, "llm_call": res}


def ablation_P(row, ctx, provider, retriever=None, seed=None):
    # classifier-only fault signal; the specific mode/cause is NOT rule-derived
    p = kh._pos(row, ctx)
    clf = (f"Classifier prediction: {'failure' if ctx['pred_lookup']['random_forest']['pred'][p] else 'no failure'} "
           f"(probability {ctx['pred_lookup']['random_forest']['prob'][p]})")
    prompt = (bg.background_text(ctx) + "\nINSTANCE:\n" + bg.instance_context(row) +
              "\n\nANALYTICAL RESULT (classifier only; determine the specific failure mode and cause "
              "yourself from the data):\n" + clf + "\n\nTASK:\n" + REPORT_TASK)
    res = llm_call.chat(provider, [{"role": "system", "content": "You are a predictive-maintenance report writer."},
                                   {"role": "user", "content": prompt}], seed=seed)
    return {"system_id": "AblP", "text": res["text"], "claims": [], "ku_trace": [],
            "prompt": prompt, "llm_call": res}


SYSTEMS = {
    "B1": b1_template_only, "B2": b2_traditional_d2t, "B3": b3_direct,
    "B4": b4_fewshot, "B5": b5_agent, "B6": b6_rag, "B7": b7_rag_cove,
    "B8": b8_framework,
}
ABLATIONS = {"AblT": ablation_T, "AblM": ablation_M, "AblQ": ablation_Q, "AblP": ablation_P}
