"""B-system report campaign for the bearing instantiation (Phase 3).

Systems (mirroring the AI4I baselines, on the frozen v3 two-tier handler):
  B1 template-only   backbone, no LLM
  B2 traditional D2T fixed hand-crafted sentences, no 5W1H battery, no LLM
  B3 direct LLM      background + raw instance stats only (no analysis)
  B4 few-shot LLM    3 exemplars from the @3/@12 PROBE files (dev/test split
                     by file identity; the main 40 are never exemplars)
  B5 tool-using agent raw primitives via function calling
                     (prominence/band-kurtosis/expected-frequency tools; the
                     two-tier VERDICT itself is NOT exposed -- the agent must
                     assemble the diagnosis itself, mirroring AI4I B5)
  B8 framework       backbone verbalization (the proposed system)

Output: one resume-safe parquet per system, code/runs/bearing_reports_B*.parquet
Usage:  python code/src/bearing_campaign.py --exemplars   (3 dev B8 calls)
        python code/src/bearing_campaign.py --run B8 B3 B4 B5   (etc.)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

import llm_call
from bearing_knowledge import (BANK_BY_HANDLER, BEARING_BACKGROUND, CANONICAL,
                               HANDLERS, SPEC, _load_bank, backbone,
                               canonical_5w1h, load_v3, make_row)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DATA = ROOT / "data" / "bearing_cwru"
PROVIDER = "openai"          # GLM-4-Flash via OpenAI-compatible endpoint
SEED = 42
REPORT_TASK = (
    "Write a concise predictive-maintenance report for THIS record. In "
    "<=150 words, cover all six aspects: (1) What is the fault/status, "
    "(2) Why it occurs (the diagnostic evidence), (3) When to act / current "
    "stage, (4) Where the problem is located, (5) Who should respond, "
    "(6) How to address it. Use only the provided data and background; do "
    "not invent numbers."
)
COMPONENT_WORD = {"IR": "inner-race", "OR": "outer-race", "B": "ball",
                  "normal": None}

# dev exemplars: @3/@12 probe files, never part of the main 40
DEV_FILES = ["144.mat", "158.mat", "246.mat"]


def raw_instance(rec: pd.Series) -> str:
    """Raw instance context -- sensor facts only, no analysis results."""
    return (f"INSTANCE: vibration record {rec.file} from the drive-end "
            f"accelerometer (12 kHz, ~10 s). Tachometer shaft speed: "
            f"{rec.rpm:.0f} rpm.")


def _metrics(res) -> dict:
    if not res:
        return {"ok": False, "prompt_tokens": 0, "completion_tokens": 0,
                "latency_s": 0.0, "cost_usd": 0.0, "error": "no-call"}
    return {"ok": bool(res.get("ok", True)),
            "prompt_tokens": res.get("prompt_tokens", 0),
            "completion_tokens": res.get("completion_tokens", 0),
            "latency_s": res.get("latency_s", 0.0),
            "cost_usd": res.get("cost_usd", 0.0), "error": res.get("error")}


def _sysmsg(c: str) -> dict:
    return {"role": "system", "content": c}


# ------------------------------------------------------------------ systems
def gen_b1(rec, ctx) -> dict:
    row = make_row(rec, ctx["thresholds"])
    kus = canonical_5w1h(row, ctx)
    return {"system_id": "B1", "instance_id": rec.file,
            "text": backbone(kus),
            "claims": [{"category": k["category"], "value": k["answer"]}
                       for k in kus],
            "ku_trace": [k["ku_id"] for k in kus], "prompt": "",
            **_metrics(None)}


def gen_b2(rec, ctx) -> dict:
    pred, tier = rec.pred, rec.tier
    f = {"BPFO": rec.BPFO, "BPFI": rec.BPFI, "BALLFAM": rec.BALLFAM}
    t = ctx["thresholds"]
    parts = []
    if pred == "normal":
        parts.append("The drive-end bearing shows no fault signature.")
        parts.append(f"Strongest characteristic-frequency prominence is "
                     f"{max(f.values()):.0f}, below all thresholds.")
    else:
        key = "BPFI" if pred == "IR" else "BPFO"
        tau = t["tau_ir"] if pred == "IR" else t["tau_or"]
        parts.append(f"A {COMPONENT_WORD[pred]} fault is present on the "
                     f"drive-end bearing.")
        parts.append(f"{key} prominence {f[key]:.0f} exceeds {tau:.0f}.")
    from bearing_knowledge import ACTION, RESPONDER, SEVERITY
    tier_k = "none" if pred == "normal" else tier
    if pred == "normal":
        parts.append("Action: continue routine monitoring (operator).")
    else:
        parts.append(f"Action: {ACTION[pred]} assign to "
                     f"{RESPONDER[tier]} (severity {SEVERITY[tier]}).")
    return {"system_id": "B2", "instance_id": rec.file,
            "text": " ".join(parts), "claims": [], "ku_trace": [],
            "prompt": "", **_metrics(None)}


def gen_b3(rec, ctx) -> dict:
    prompt = (BEARING_BACKGROUND + "\n" + raw_instance(rec) +
              "\n\nTASK:\n" + REPORT_TASK)
    res = llm_call.chat(PROVIDER, [
        _sysmsg("You are a predictive-maintenance analyst."),
        {"role": "user", "content": prompt}], seed=SEED)
    return {"system_id": "B3", "instance_id": rec.file, "text": res["text"],
            "claims": [], "ku_trace": [], "prompt": prompt, **_metrics(res)}


def gen_b8(rec, ctx) -> dict:
    row = make_row(rec, ctx["thresholds"])
    kus = canonical_5w1h(row, ctx)
    bb = backbone(kus)
    f = row["features"]
    inst = (f"INSTANCE: record {rec.file}, shaft speed {f['rpm']:.0f} rpm, "
            f"diagnostic band {f['kurt_band_lo']:.0f} Hz "
            f"(kurtosis {f['kurt_band']:.1f}).")
    prompt = (BEARING_BACKGROUND + "\n" + inst +
              "\n\nVERIFIED ANALYTICAL FINDINGS (knowledge-unit answers; "
              "use ONLY these facts):\n" + bb + "\n\nTASK:\n" + REPORT_TASK +
              "\nThe findings are computed by the analytical components and "
              "are authoritative; verbalize them into a fluent report "
              "without changing any value.")
    res = llm_call.chat(PROVIDER, [
        _sysmsg("You are a predictive-maintenance report writer."),
        {"role": "user", "content": prompt}], seed=SEED)
    return {"system_id": "B8", "instance_id": rec.file, "text": res["text"],
            "claims": [{"category": k["category"], "value": k["answer"]}
                       for k in kus],
            "ku_trace": [k["ku_id"] for k in kus], "prompt": prompt,
            **_metrics(res)}


# ---- B5: raw primitive tools (verdict NOT exposed) ------------------------
_TOOLS = [
    {"type": "function", "function": {
        "name": "get_shaft_speed",
        "description": "Tachometer shaft speed of the current record.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_envelope_prominence",
        "description": "Harmonic prominence of a fault component's "
        "characteristic frequency on the band-scanned envelope spectrum.",
        "parameters": {"type": "object", "properties": {
            "component": {"type": "string",
                          "enum": ["BPFO", "BPFI", "BSF", "2xBSF", "FTF"]}},
            "required": ["component"]}}},
    {"type": "function", "function": {
        "name": "get_expected_frequency",
        "description": "Expected characteristic frequency in Hz for a "
        "component at the current shaft speed.",
        "parameters": {"type": "object", "properties": {
            "component": {"type": "string",
                          "enum": ["BPFO", "BPFI", "BSF", "2xBSF", "FTF"]}},
            "required": ["component"]}}},
    {"type": "function", "function": {
        "name": "get_band_kurtosis",
        "description": "Maximum band kurtosis over the scanned resonance "
        "bands, and the band's lower edge.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]
_MULTKEY = {"BPFO": "BPFO", "BPFI": "BPFI", "BSF": "BSF", "2xBSF":
            "ball_defect_2xBSF", "FTF": "FTF"}


def _exec_tool(name: str, args: dict, rec: pd.Series) -> dict:
    M = SPEC["defect_freq_multiples_x_fr"]
    fr = rec.rpm / 60.0
    if name == "get_shaft_speed":
        return {"rpm": round(float(rec.rpm), 1), "fr_hz": round(fr, 2)}
    if name == "get_envelope_prominence":
        c = args.get("component", "BPFO")
        key = {"BPFO": "BPFO", "BPFI": "BPFI", "BSF": "BSF",
               "2xBSF": "BALL", "FTF": "FTF"}[c]
        return {"component": c, "prominence": round(float(rec[key]), 1)}
    if name == "get_expected_frequency":
        c = args.get("component", "BPFO")
        return {"component": c, "multiple_x_fr": M[_MULTKEY[c]],
                "hz": round(M[_MULTKEY[c]] * fr, 1)}
    if name == "get_band_kurtosis":
        return {"band_lo_hz": round(float(rec.kurt_band_lo)),
                "kurtosis": round(float(rec.kurt_band), 1)}
    return {"error": f"unknown tool {name}"}


def gen_b5(rec, ctx) -> dict:
    messages = [
        _sysmsg("You are a tool-using predictive-maintenance agent. Inspect "
                "the record with the available tools, then write the report "
                "yourself from what you found."),
        {"role": "user", "content": BEARING_BACKGROUND + "\n" +
         raw_instance(rec) + "\n\nTASK:\n" + REPORT_TASK},
    ]
    prompt = messages[1]["content"]
    res = llm_call.chat(PROVIDER, messages, tools=_TOOLS, tool_choice="auto",
                        seed=SEED)
    tot = {"prompt_tokens": res.get("prompt_tokens", 0),
           "completion_tokens": res.get("completion_tokens", 0),
           "latency_s": res.get("latency_s", 0.0),
           "cost_usd": res.get("cost_usd", 0.0)}
    text, error = res.get("text", ""), res.get("error")
    if res.get("tool_calls"):
        messages.append({"role": "assistant", "content": res["text"],
                         "tool_calls": [{"id": tc["id"], "type": "function",
                                         "function": {"name": tc["name"],
                                                      "arguments": tc["args"]}}
                                        for tc in res["tool_calls"]]})
        for tc in res["tool_calls"]:
            try:
                pargs = json.loads(tc["args"]) if tc["args"] else {}
            except Exception:  # noqa: BLE001
                pargs = {}
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(
                                 _exec_tool(tc["name"], pargs, rec),
                                 ensure_ascii=False)})
        messages.append({"role": "user", "content":
                         "Using the tool results above, write the final "
                         "maintenance report now. Do not call any more "
                         "tools."})
        res2 = llm_call.chat(PROVIDER, messages, seed=SEED)
        for k in tot:
            tot[k] += res2.get(k, 0) if k != "cost_usd" else res2.get(k, 0)
        text, error = res2["text"], res2.get("error")
    return {"system_id": "B5", "instance_id": rec.file, "text": text,
            "claims": [], "ku_trace": [], "prompt": prompt,
            "ok": bool(res.get("ok", True)) and error is None,
            **{k: tot[k] for k in ("prompt_tokens", "completion_tokens",
                                   "latency_s", "cost_usd")},
            "error": error}


# ---- B4: few-shot with dev exemplars --------------------------------------
def dev_exemplars(df_all: pd.DataFrame, thresholds: dict) -> list[dict]:
    cache = RUNS / "bearing_dev_exemplars.json"
    if cache.exists():
        return json.load(open(cache, encoding="utf-8"))
    probe = pd.read_csv(DATA / "bearing_v3_results.csv")
    out = []
    ctx = {"thresholds": thresholds, "spec": SPEC}
    for fname in DEV_FILES:
        rec = _probe_row(probe, fname)
        r = gen_b8(rec, ctx)
        out.append({"context": (f"INSTANCE: record {fname}, shaft speed "
                                f"{rec.rpm:.0f} rpm, diagnostic band "
                                f"{rec.kurt_band_lo:.0f} Hz."),
                    "report": r["text"]})
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    return out


def _probe_row(probe_df: pd.DataFrame, fname: str) -> pd.Series:
    """Synthesize a probe-file row (features via bearing_v3.features)."""
    import bearing_v3
    meta = {"load_hp": 0, "condition": "fault", "component": "OR@3",
            "defect_size_in": 0.0}
    f = bearing_v3.features(fname.split(".")[0], meta)
    # reuse the frozen decision logic on the probe features
    cal = probe_df[probe_df.load_hp <= 1]
    import numpy as np
    tau_ir = float(np.sqrt(cal[cal.truth == "IR"].BPFI.min()
                           * cal[cal.truth != "IR"].BPFI.max()))
    om = (cal.truth == "OR") & (cal.defect_in != 0.014)
    tau_or = float(np.sqrt(cal[om].BPFO.min() * cal[~om].BPFO.max()))
    pred, tier = "normal", "model"
    if f["BPFI"] >= tau_ir and f["BPFI"] / max(f["BPFO"], f["BALLFAM"]) >= 2.4:
        pred, tier = "IR", "rule"
    elif f["BPFO"] >= tau_or:
        pred, tier = "OR", "rule"
    else:
        from sklearn.ensemble import RandomForestClassifier
        FEATS = ["rms", "kurt_raw", "skew_raw", "en0", "en1", "en2", "en3",
                 "en4", "en5", "BPFO", "BPFI", "BSF", "BALL", "FTF",
                 "BALLFAM", "kurt_band"]
        clf = RandomForestClassifier(n_estimators=500, random_state=42,
                                     class_weight="balanced")
        clf.fit(cal[FEATS], cal.truth)
        pred = clf.predict(pd.DataFrame([{k: f[k] for k in FEATS}]))[0]
        tier = "model"
    base = {"file": fname, "truth": "OR", "load_hp": 0, "defect_in": 0.007,
            "pred": pred, "tier": tier, **f}
    return pd.Series(base)


def gen_b4(rec, ctx, exemplars: list[dict]) -> dict:
    messages = [_sysmsg("You are a predictive-maintenance analyst. Follow "
                        "the style of the examples.")]
    for e in exemplars:
        messages.append({"role": "user", "content":
                         BEARING_BACKGROUND + "\n" + e["context"] +
                         "\n\nTASK:\n" + REPORT_TASK})
        messages.append({"role": "assistant", "content": e["report"]})
    prompt = (BEARING_BACKGROUND + "\n" + raw_instance(rec) +
              "\n\nTASK:\n" + REPORT_TASK)
    messages.append({"role": "user", "content": prompt})
    res = llm_call.chat(PROVIDER, messages, seed=SEED)
    return {"system_id": "B4", "instance_id": rec.file, "text": res["text"],
            "claims": [], "ku_trace": [], "prompt": prompt, **_metrics(res)}


GEN = {"B1": gen_b1, "B2": gen_b2, "B3": gen_b3, "B8": gen_b8}


def run(systems: list[str]) -> None:
    global BANK_BY_HANDLER
    BANK_BY_HANDLER = _load_bank()
    df, thresholds = load_v3()
    ctx = {"thresholds": thresholds, "spec": SPEC}
    exemplars = None
    for sys_id in systems:
        out = RUNS / f"bearing_reports_{sys_id}.parquet"
        done = set()
        if out.exists():
            done = set(pd.read_parquet(out).instance_id)
        rows, n_calls = [], 0
        for _, rec in df.iterrows():
            if rec.file in done:
                continue
            if sys_id == "B4":
                if exemplars is None:
                    exemplars = dev_exemplars(df, thresholds)
                r = gen_b4(rec, ctx, exemplars)
            elif sys_id == "B5":
                r = gen_b5(rec, ctx)
            else:
                r = GEN[sys_id](rec, ctx)
            rows.append(r)
            n_calls += 0 if sys_id in ("B1", "B2") else (2 if sys_id == "B5"
                                                         else 1)
            if r["text"]:
                print(f"  {sys_id} {rec.file} ok "
                      f"({r.get('completion_tokens', 0)} tok)")
            else:
                print(f"  {sys_id} {rec.file} EMPTY/error: "
                      f"{r.get('error')}")
        if rows:
            new = pd.DataFrame(rows)
            full = new if not out.exists() else pd.concat(
                [pd.read_parquet(out), new], ignore_index=True)
            full.to_parquet(out, index=False)
            print(f"{sys_id}: +{len(rows)} rows -> {out.name} "
                  f"(total {len(full)})")
        else:
            print(f"{sys_id}: nothing to do ({len(done)} done)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", default=[])
    ap.add_argument("--exemplars", action="store_true")
    args = ap.parse_args()
    if args.exemplars:
        global BANK_BY_HANDLER
        BANK_BY_HANDLER = _load_bank()
        df, thresholds = load_v3()
        dev_exemplars(df, thresholds)
        print("dev exemplars ready")
    if args.run:
        run(args.run)


if __name__ == "__main__":
    main()
