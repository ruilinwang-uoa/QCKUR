"""
metropt_campaign.py -- B-system report campaign for the MetroPT
in-service instantiation (attempt 4; preregistration frozen 2026-09-21).

Systems (AI4I-comparable battery, prereg section 4):
  B1 template-only     knowledge bank + handler verdict -> template (no LLM)
  B2 traditional D2T   canned sentences, no 5W1H battery (no LLM)
  B3 direct LLM        background + frozen digest only
  B5 tool-using agent  digest + raw-primitive tools (hourly stats, LPS
                       episodes, duty runs, percentiles); NO verdicts,
                       NO baselines, NO thresholds
  B8 framework         verified analytical findings (handler battery)
                       verbalized by the backbone

All systems receive the SAME fixed REPORT_TASK; baselines never consume
the question stream (WS2 design).

Usage:
  python metropt_campaign.py --run B1 B2            (local, no API)
  python metropt_campaign.py --run B3 B5 B8 --smoke 2
  python metropt_campaign.py --run B3 B5 B8         (full, resume-safe)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import llm_call
from metropt_handlers import handle, load_spec
from metropt_render import (METROPT_BACKGROUND, REPORT_TASK, VEHICLE_DESC,
                            b1_text, b2_sentences, canonical_kus,
                            digest_text, slot_fill)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
PROVIDER = "openai"          # GLM-4-Flash, OpenAI-compatible (as bearing)
SEED = 42

SOURCES = [("metropt3", "features_metropt3.parquet"),
           ("metropt2022", "features_metropt2022.parquet"),
           ("metropt2022B", "features_metropt2022B.parquet")]


def _metrics(res) -> dict:
    if not res:
        return {"ok": False, "prompt_tokens": 0, "completion_tokens": 0,
                "latency_s": 0.0, "cost_usd": 0.0, "error": "no-call"}
    return {"ok": bool(res.get("ok", True)),
            "prompt_tokens": res.get("prompt_tokens", 0),
            "completion_tokens": res.get("completion_tokens", 0),
            "latency_s": res.get("latency_s", 0.0),
            "cost_usd": res.get("cost_usd", 0.0),
            "error": res.get("error")}


def load_windows(smoke: int | None = None):
    spec = load_spec()
    rows = []
    for tag, par in SOURCES:
        f = pd.read_parquet(RUNS / par).copy()
        f["vehicle"] = tag
        if smoke:
            picks = list(f[f.kind == "positive"].index[:smoke]) + \
                    list(f[f.kind == "control"].index[:smoke])
            f = f.loc[sorted(set(picks))]
        rows.append(f)
    return pd.concat(rows), spec


# ------------------------------------------------------------------ B1/B2
def gen_b1(feat, verdict, spec, vehicle):
    slots = slot_fill(verdict, feat, spec, vehicle)
    text = b1_text(slots, verdict["label"])
    kus = canonical_kus(slots)
    return {"system_id": "B1", "text": text,
            "claims": [{"ku_id": k["ku_id"], "category": k["category"],
                        "value": k["answer"]} for k in kus],
            "ku_trace": [k["ku_id"] for k in kus], "prompt": "",
            **_metrics(None)}


def gen_b2(feat, verdict, spec, vehicle):
    return {"system_id": "B2", "text": b2_sentences(verdict, feat, spec,
                                                    vehicle),
            "claims": [], "ku_trace": [], "prompt": "",
            **_metrics(None)}


# ------------------------------------------------------------------ B3
def gen_b3(feat, verdict, spec, vehicle):
    prompt = (METROPT_BACKGROUND + "\n\n" + digest_text(feat, vehicle) +
              "\n\nTASK:\n" + REPORT_TASK)
    res = llm_call.chat(PROVIDER, [
        {"role": "system",
         "content": "You are a predictive-maintenance analyst."},
        {"role": "user", "content": prompt}], seed=SEED, max_tokens=700)
    return {"system_id": "B3", "text": res["text"], "claims": [],
            "ku_trace": [], "prompt": prompt, **_metrics(res)}


# ------------------------------------------------------------------ B5
_TOOLS = [
    {"type": "function", "function": {
        "name": "hourly_table",
        "description": "Hourly telemetry aggregates for the whole 24 h "
        "window (TP2 mean/max, H1, motor-current mean, oil mean/max, LPS "
        "fraction per hour).",
        "parameters": {"type": "object", "properties": {},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "lps_episodes",
        "description": "Episodes of continuous LPS low-pressure-warning "
        "activation (start, end, minutes), longest first.",
        "parameters": {"type": "object", "properties": {},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "duty_runs",
        "description": "Continuous runs of motor current > 4.5 A "
        "(working duty), longest first.",
        "parameters": {"type": "object", "properties": {},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "percentiles",
        "description": "Distribution percentiles (5/50/95/99) of a raw "
        "signal within the window.",
        "parameters": {"type": "object", "properties": {
            "signal": {"type": "string", "enum": ["tp2", "motor_current",
                                                  "oil_temperature"]}},
            "required": ["signal"]}}},
]


def _exec_tool(name: str, args: dict, prim: dict) -> dict:
    if name == "hourly_table":
        return {"hourly": prim["hourly"]}
    if name == "lps_episodes":
        return {"episodes": prim["lps_episodes_min"]}
    if name == "duty_runs":
        return {"runs_over_4p5A_minutes":
                    prim["duty_runs_over_4p5A_min"]}
    if name == "percentiles":
        sig = args.get("signal", "tp2")
        return {"signal": sig, "percentiles": prim["percentiles"][sig]}
    return {"error": f"unknown tool {name}"}


def gen_b5(feat, verdict, spec, vehicle, prim):
    prompt = (METROPT_BACKGROUND + "\n\nWINDOW: 24 h ending "
              f"{feat.get('t_end')} on {VEHICLE_DESC[vehicle]}. You can "
              f"query raw telemetry aggregates with the provided tools "
              f"(they return computed statistics; no analysis or "
              f"thresholds are applied for you).\n\nTASK:\n" + REPORT_TASK)
    msgs = [{"role": "system",
             "content": "You are a predictive-maintenance analyst. Use "
                        "the tools to inspect the telemetry before "
                        "writing the report."},
            {"role": "user", "content": prompt}]
    total = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
             "latency_s": 0.0}
    text, err = "", None
    for _turn in range(6):
        res = llm_call.chat(PROVIDER, msgs, seed=SEED, max_tokens=800,
                            tools=_TOOLS)
        for k in ("prompt_tokens", "completion_tokens", "cost_usd",
                  "latency_s"):
            total[k] += res.get(k, 0) or 0
        if not res.get("ok"):
            err = res.get("error")
            break
        if res["tool_calls"]:
            msgs.append({"role": "assistant", "content": res["text"] or "",
                         "tool_calls": [
                             {"id": tc["id"], "type": "function",
                              "function": {"name": tc["name"],
                                           "arguments": tc["args"]}}
                             for tc in res["tool_calls"]]})
            for tc in res["tool_calls"]:
                try:
                    args = json.loads(tc["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = _exec_tool(tc["name"], args, prim)
                msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(out)[:6000]})
            continue
        text = res["text"]
        break
    return {"system_id": "B5", "text": text, "claims": [], "ku_trace": [],
            "prompt": prompt,
            "ok": bool(text) and not err,
            "prompt_tokens": total["prompt_tokens"],
            "completion_tokens": total["completion_tokens"],
            "latency_s": round(total["latency_s"], 3),
            "cost_usd": round(total["cost_usd"], 5),
            "error": err}


# ------------------------------------------------------------------ B8
def gen_b8(feat, verdict, spec, vehicle):
    slots = slot_fill(verdict, feat, spec, vehicle)
    findings = b1_text(slots, verdict["label"])
    kus = canonical_kus(slots)
    prompt = (METROPT_BACKGROUND + "\n\nWINDOW: 24 h ending "
              f"{feat.get('t_end')} on {VEHICLE_DESC[vehicle]}.\n\n"
              "VERIFIED ANALYTICAL FINDINGS (computed by the knowledge "
              "units and two-tier handler; use ONLY these facts):\n"
              + findings + "\n\nTASK:\n" + REPORT_TASK +
              "\nThe findings are authoritative; verbalize them into a "
              "fluent report without changing any value.")
    res = llm_call.chat(PROVIDER, [
        {"role": "system",
         "content": "You are a predictive-maintenance report writer."},
        {"role": "user", "content": prompt}], seed=SEED, max_tokens=700)
    return {"system_id": "B8", "text": res["text"],
            "claims": [{"ku_id": k["ku_id"], "category": k["category"],
                        "value": k["answer"]} for k in kus],
            "ku_trace": [k["ku_id"] for k in kus], "prompt": prompt,
            **_metrics(res)}


GEN = {"B1": gen_b1, "B2": gen_b2, "B3": gen_b3, "B8": gen_b8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=["B1", "B2"],
                    choices=["B1", "B2", "B3", "B5", "B8"])
    ap.add_argument("--smoke", type=int, default=None,
                    help="first N positives + N controls per vehicle")
    args = ap.parse_args()

    wins, spec = load_windows(args.smoke)
    prims = {}
    for tag, _ in SOURCES:
        p = RUNS / f"primitives_{tag}.json"
        prims[tag] = json.loads(p.read_text(encoding="utf-8")) \
            if p.exists() else {}

    for sys_id in args.run:
        dest = RUNS / f"reports_metropt_{sys_id}.parquet"
        done = set()
        rows = []
        if dest.exists():
            old = pd.read_parquet(dest)
            # resume key = (vehicle, window_id): window_ids REPEAT across
            # vehicles -- a vehicle-blind key silently skips same-named
            # windows of the other vehicles (caught 2026-09-21)
            done = {(r.vehicle, r.window_id) for r in
                    old[(old.ok.astype(str) == "True") |
                        (old.text.fillna("") != "")].itertuples()}
            old = old.drop_duplicates(subset=["vehicle", "window_id"],
                                      keep="last")
            rows = old.to_dict("records")
        n_new, n_fail = 0, 0
        for wid, feat in wins.iterrows():
            vehicle = feat["vehicle"]
            if (vehicle, wid) in done:
                continue
            verdict = handle(feat.to_dict(), vehicle, spec)
            fd = feat.to_dict()
            if sys_id == "B5":
                rec = gen_b5(fd, verdict, spec, vehicle,
                             prims.get(vehicle, {}).get(wid, {}))
            else:
                rec = GEN[sys_id](fd, verdict, spec, vehicle)
            rec.update({"window_id": wid, "vehicle": vehicle,
                        "kind": feat["kind"],
                        "event_id": feat.get("event_id"),
                        "offset_h": feat.get("offset_h"),
                        "verdict_label": verdict["label"],
                        "verdict_tier": verdict["tier"]})
            rows.append(rec)
            n_new += 1
            if not rec.get("text"):
                n_fail += 1
            if n_new % 10 == 0:
                print(f"  {sys_id}: {n_new} generated "
                      f"({n_fail} empty)")
                pd.DataFrame(rows).to_parquet(dest, index=False)
            time.sleep(0.3)
        pd.DataFrame(rows).to_parquet(dest, index=False)
        print(f"{sys_id}: +{n_new} new ({n_fail} empty) -> {dest} "
              f"({len(rows)} rows total)")


if __name__ == "__main__":
    main()
