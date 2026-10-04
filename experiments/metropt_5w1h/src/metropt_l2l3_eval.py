"""
metropt_l2l3_eval.py -- Layer-2/Layer-3 judging for the MetroPT campaign.

SAME judge prompts, JSON schemas, layer formulas and primary judge
(DeepSeek-V4-Flash, max_tokens 3000, seed 42) as the AI4I/bearing
campaigns -- protocol-identical three-layer evaluation.

VERIFIED FACTS block = telemetry-grounded GT with lead-time honesty:
the documented future event is given as CONTEXT explicitly marked NOT
observable from this window; a fault claim against a no-indication
window therefore contradicts the facts.

Resume key = (system, vehicle, window_id) -- window_ids repeat across
vehicles.

Usage:
  python metropt_l2l3_eval.py --run [--workers 4]
  python metropt_l2l3_eval.py --smoke 3
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import llm_call
from metropt_handlers import handle, load_spec

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
SYSTEMS = ["B1", "B2", "B3", "B3R", "B5", "B8", "B8S7", "B3RS7"]
PROVIDER = "deepseek"
SEED = 42
# DS chain-of-thought budget lesson (AI4I/bearing): the judge's emitted
# reasoning consumes the completion budget before the JSON. MetroPT L2
# needs more than bearing's 3000 (successes hug the cap: median 1921,
# max 2996, 21/276 within 100 of it; failures return ok=True with EMPTY
# text) -> L2 escalated to 5000 (same documented-escalation precedent as
# AI4I's 1500->3000). L3 stays at 3000 (512/525 ok, no cap pressure).
MAXTOK = {"l2": 5000, "l3": 3000}

L2_PROMPT = """You are a strict consistency checker for predictive-maintenance reports.
Given a REPORT and the VERIFIED FACTS (ground truth), extract the report's
factual claims and verify each against the facts.

For every claim return: a 5W1H category (what/why/when/where/who/how), the
claim text, and a verdict:
- "consistent": the claim matches the verified facts
- "contradicts": the claim conflicts with the verified facts (wrong value/mode/cause)
- "unsupported": the claim cannot be checked against the verified facts
Also mark coverage: for each aspect 1 if the report addresses it, else 0.

Return STRICT JSON only:
{"claims":[{"category":"...","statement":"...","verdict":"consistent|contradicts|unsupported"}],
 "coverage":{"what":0,"why":0,"when":0,"where":0,"who":0,"how":0}}
"""

L3_RUBRIC = """You are a strict evaluator of predictive-maintenance reports.
Score the report on five dimensions, each 1-5 (5 best). Use ONLY the provided
VERIFIED FACTS as ground truth.

- faithfulness: are all stated facts (fault mode, parameter values, causes,
  thresholds) consistent with the VERIFIED FACTS? Penalize any invented or
  wrong number/mode.
- completeness: does it give the information needed to understand the window?
- coherence: is it well-organized, fluent, internally consistent?
- actionability: does it give a concrete, correct maintenance action?
- coverage: for EACH of the six aspects, 1 if the report addresses it
  meaningfully, else 0:
    what (fault/status), why (cause), when (stage/urgency),
    where (component/location), who (responder/severity), how (action).

Return STRICT JSON only:
{"faithfulness":1-5,"completeness":1-5,"coherence":1-5,"actionability":1-5,
 "coverage":{"what":0/1,"why":0/1,"when":0/1,"where":0/1,"who":0/1,"how":0/1}}
"""


def _chat_json(system_prompt: str, user: str, layer: str = "l2",
               seed: int | None = None) -> dict:
    res, last_err = None, None
    for attempt in range(3):
        res = llm_call.chat(PROVIDER, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            seed=SEED if seed is None else seed,
            max_tokens=MAXTOK[layer])
        if res.get("ok") and res.get("text"):
            try:
                return {"ok": True, "j": json.loads(res["text"]),
                        "res": res}
            except Exception:  # noqa: BLE001
                last_err = "json-parse"
        else:
            last_err = str(res.get("error"))[:120]
        time.sleep(4 * (attempt + 1))
    return {"ok": False, "j": {}, "res": res or {}, "err": last_err}


def score_l2(text: str, facts: str, inst: str) -> dict:
    if not text or not text.strip():
        return {"ok": False, "L2": 0.0, "consistency": 0.0,
                "contradiction": 0.0, "coverage": 0.0, "n_claims": 0}
    r = _chat_json(L2_PROMPT, f"INSTANCE:\n{inst}\n\nVERIFIED FACTS:\n{facts}"
                   f"\n\nREPORT:\n{text}\n\nExtract and verify the claims; "
                   "return the JSON.", layer="l2")
    claims = r["j"].get("claims", []) or []
    n = len(claims)
    cons = (sum(1 for c in claims if c.get("verdict") == "consistent") / n
            if n else 0.0)
    contra = (sum(1 for c in claims if c.get("verdict") == "contradicts") / n
              if n else 0.0)
    cov = r["j"].get("coverage") or {}
    if not isinstance(cov, dict):
        cov = {}
    coverage = sum(int(cov.get(k, 0)) for k in
                   ["what", "why", "when", "where", "who", "how"]) / 6.0
    return {"ok": r["ok"], "consistency": round(cons, 3),
            "contradiction": round(contra, 3),
            "coverage": round(coverage, 3), "n_claims": n,
            "L2": round(0.50 * cons + 0.25 * coverage
                        + 0.25 * (1 - contra), 3),
            "judge_tokens": r["res"].get("completion_tokens", 0)}


def score_l3(text: str, facts: str, inst: str) -> dict:
    if not text or not text.strip():
        return {"ok": False, "faithfulness": 0, "completeness": 0,
                "coherence": 0, "actionability": 0, "coverage":
                    {k: 0 for k in "what why when where who how".split()},
                "L3": 0.75}
    r = _chat_json(L3_RUBRIC, f"INSTANCE:\n{inst}\n\nVERIFIED FACTS (ground "
                   f"truth):\n{facts}\n\nREPORT TO EVALUATE:\n{text}\n\n"
                   "Return the JSON scores.", layer="l3")
    j = r["j"]
    cov = j.get("coverage") or {}
    if not isinstance(cov, dict):
        cov = {}
    n_cov = sum(int(cov.get(k, 0)) for k in
                ["what", "why", "when", "where", "who", "how"])
    cov5 = n_cov / 6.0 * 5.0
    dims = {d: float(j.get(d, 0)) for d in ("faithfulness", "completeness",
                                            "coherence", "actionability")}
    dims["coverage"] = cov5
    L3 = (0.30 * dims["faithfulness"] + 0.15 * dims["completeness"]
          + 0.25 * cov5 + 0.15 * dims["coherence"]
          + 0.15 * dims["actionability"])
    dims.update({"ok": r["ok"], "L3": round(L3, 3), "cov_count": n_cov,
                 "judge_tokens": r["res"].get("completion_tokens", 0)})
    return dims


# ------------------------------------------------------------------ facts
def build_facts(g: dict, feat: dict, verdict: dict, vehicle: str,
                spec: dict) -> tuple[str, str]:
    """(facts block, instance line) with lead-time honesty.

    Protocol note (aligned with AI4I, where the facts block = the full
    KU battery): the block carries (a) telemetry-grounded GT, (b) the
    documented future event explicitly marked NOT observable, and (c)
    the knowledge bank's frozen verified outputs for this window (the
    same battery B1/B8 verbalize). Without (c) the judge marks all
    bank-derived statements unsupported, penalizing exactly the systems
    that carry the knowledge layer."""
    from metropt_render import canonical_kus, slot_fill
    v = spec["per_vehicle"][vehicle]
    vis = g["visible"]
    ev = g.get("event")
    lines = ["What: " + g["expected"]["what"]]
    if vis["air"]:
        lines.append("Why: air-leak signature in telemetry: near-continuous "
                     "working duty without the pressure cutoff, and/or "
                     "sustained LPS low-pressure activation")
    elif vis["oil"]:
        lines.append("Why: oil temperature elevated above this vehicle's "
                     "baseline band")
    elif vis["thermal"]:
        lines.append("Why: abnormal oil-temperature warming dynamics "
                     "(overload pattern)")
    else:
        lines.append("Why: measurements inside this vehicle's baseline band")
    lines.append(
        f"Key values: TP2 p95 {feat.get('tp2_p95')} bar "
        f"({(feat.get('tp2_frac_ge9_5') or 0)*100:.1f}% of samples "
        f">= 9.5 bar); working duty {(feat.get('mc_on_frac') or 0)*100:.1f}% "
        f"(longest run {feat.get('mc_on_maxrun_min')} min); LPS active "
        f"{(feat.get('lps_frac') or 0)*100:.2f}% (longest "
        f"{feat.get('lps_maxrun_min')} min); oil max "
        f"{feat.get('oil_max')} degC vs baseline p99 {v['oil_max_p99']} "
        f"(elevation {g['oil_elev_c']} degC, window warming "
        f"{feat.get('oil_trend_c')} degC)")
    if ev:
        lines.append(
            f"Event context (NOT observable from this window's telemetry): "
            f"a documented {ev['mode']} event ({ev['severity_class']}, "
            f"disposition {ev['disposition']}) ended {ev['lead_h']:.0f} h "
            f"after this window")
    else:
        if g.get("ooc_candidate"):
            lines.append("Control context: no documented event; nominal "
                         "expected, but an unexplained signature is present "
                         "(out-of-bank candidate)")
        else:
            lines.append("Control context: no documented event; nominal "
                         "operation expected")
    lines.append("Action ground truth: " + g["expected"]["action_class"])
    slots = slot_fill(verdict, feat, spec, vehicle)
    lines.append("Verified bank outputs for this window (frozen knowledge "
                 "bank; authoritative where applicable):")
    for k in canonical_kus(slots):
        lines.append(f"{k['ku_id']} ({k['category']}): {k['answer']}")
    facts = "\n".join(lines)
    inst = (f"{vehicle} window {g['window_id']} (24 h ending {g['t_end']}), "
            f"data coverage {(feat.get('coverage') or 0)*100:.0f}%")
    return facts, inst


_FEAT: dict[str, pd.DataFrame] = {}


def feat_row(tag: str, wid: str) -> dict:
    if tag not in _FEAT:
        par = {"metropt3": "features_metropt3.parquet",
               "metropt2022": "features_metropt2022.parquet",
               "metropt2022B": "features_metropt2022B.parquet"}[tag]
        _FEAT[tag] = pd.read_parquet(RUNS / par)
    return _FEAT[tag].loc[wid].to_dict()


def run_layer(layer: str, workers: int = 4, smoke: int | None = None) -> None:
    spec = load_spec()
    gt = {f"{g['vehicle']}::{g['window_id']}": g for g in json.load(
        open(RUNS / "metropt_gt.json", encoding="utf-8"))["windows"]}
    # window_ids repeat across vehicles -> key facts lookup by
    # (vehicle, window_id)
    out = RUNS / f"metropt_{layer}.parquet"
    done = set()
    if out.exists():
        old = pd.read_parquet(out)
        # only successful rows count as done; ok=False (zero-claim /
        # json-parse floors) retries on rerun (Kimi/bearing sweep pattern)
        ok_mask = old.ok.astype(str) == "True"
        done = {(r.system, r.vehicle, r.window_id) for r in
                old[ok_mask].itertuples()}
    tasks = []
    for sys_id in SYSTEMS:
        rep = pd.read_parquet(RUNS / f"reports_metropt_{sys_id}.parquet")
        if smoke:
            rep = rep.groupby("vehicle", group_keys=False).apply(
                lambda d: pd.concat([d[d.kind == "positive"].head(smoke),
                                     d[d.kind == "control"].head(smoke)]))
        for _, r in rep.iterrows():
            if (sys_id, r.vehicle, r.window_id) in done:
                continue
            g = gt[f"{r.vehicle}::{r.window_id}"]
            feat = feat_row(r.vehicle, r.window_id)
            verdict = handle(feat, r.vehicle, spec)
            facts, inst = build_facts(g, feat, verdict, r.vehicle, spec)
            tasks.append((sys_id, r.vehicle, r.window_id, str(r.text),
                          facts, inst))
    print(f"{layer}: {len(tasks)} to judge ({len(done)} done)")
    if not tasks:
        return
    lock, rows = threading.Lock(), []

    def _save(new: pd.DataFrame) -> None:
        """Append + dedupe by key, preferring ok=True rows (retries of
        previously-failed keys must not leave stale ok=False copies)."""
        full = new if not out.exists() else pd.concat(
            [pd.read_parquet(out), new], ignore_index=True)
        full["_ok"] = (full.ok.astype(str) == "True").astype(int)
        full = (full.sort_values("_ok").drop_duplicates(
            subset=["system", "vehicle", "window_id"], keep="last")
            .drop(columns="_ok").reset_index(drop=True))
        full.to_parquet(out, index=False)

    def work(t):
        sys_id, veh, wid, text, facts, inst = t
        s = (score_l2(text, facts, inst) if layer == "l2"
             else score_l3(text, facts, inst))
        return {"system": sys_id, "vehicle": veh, "window_id": wid, **s}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        persisted = 0
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                t = futs[fut]
                row = {"system": t[0], "vehicle": t[1], "window_id": t[2],
                       "ok": False, "error": str(e)}
            with lock:
                rows.append(row)
                if i % 20 == 0 or i == len(tasks):
                    new = pd.DataFrame(rows[persisted:])
                    _save(new)
                    persisted = len(rows)
                    print(f"  {layer}: {i}/{len(tasks)} "
                          f"(parquet {len(pd.read_parquet(out))} rows)")
    new = pd.DataFrame(rows[persisted:])
    if len(new):
        _save(new)
    n_ok = (pd.read_parquet(out).ok.astype(str) == "True").sum()
    print(f"{layer}: complete -> {out.name} "
          f"({len(pd.read_parquet(out))} rows, {n_ok} ok)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.run or args.smoke:
        run_layer("l2", args.workers, args.smoke)
        run_layer("l3", args.workers, args.smoke)


if __name__ == "__main__":
    main()
