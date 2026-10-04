"""Layer-2 / Layer-3 judging for the bearing campaign (Phase 3).

Uses the SAME judge prompts, JSON schemas, layer formulas, and primary judge
(DeepSeek-V4-Flash, max_tokens 1500, seed 42) as the AI4I campaign, so the
bearing results are protocol-identical (paper: one three-layer protocol).

Facts block  = bearing_gt.json canonical 5W1H ground-truth answers.
Instance ctx = record id + shaft speed + load.

Resume-safe: code/runs/bearing_l2.parquet / bearing_l3.parquet keyed on
(system, instance_id); the done set is read back before appending (the
paraphrase-run checkpoint lesson).

Usage:
  python code/src/bearing_l2l3_eval.py --run          # both layers, 4 workers
  python code/src/bearing_l2l3_eval.py --summary      # full 3-layer table
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import llm_call
from bearing_knowledge import load_v3

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SYSTEMS = ["B1", "B2", "B3", "B4", "B5", "B8"]
PROVIDER = "deepseek"      # primary judge, as in the AI4I campaign
SEED = 42
# Campaign judging budget = 3000: DS-V4-Flash's emitted chain-of-thought
# consumes the completion budget before the JSON on verbose free-form
# reports (verified: 1500-cap calls finish with empty text). Same remedy as
# the AI4I GLM-5.3 third judge; applied uniformly to ALL systems here.
MAXTOK = 3000

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
- completeness: does it give the information needed to understand the sample?
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


def _chat_json(system_prompt: str, user: str) -> dict:
    import time as _t
    res, last_err = None, None
    for attempt in range(3):
        res = llm_call.chat(PROVIDER, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user}],
            response_format={"type": "json_object"}, seed=SEED,
            max_tokens=MAXTOK)
        if res.get("ok") and res.get("text"):
            try:
                return {"ok": True, "j": json.loads(res["text"]), "res": res}
            except Exception:  # noqa: BLE001
                last_err = "json-parse"
        else:
            last_err = str(res.get("error"))[:120]
        _t.sleep(4 * (attempt + 1))
    return {"ok": False, "j": {}, "res": res or {},
            "err": last_err}


def score_l2(text: str, facts: str, inst: str) -> dict:
    if not text or not text.strip():
        return {"ok": False, "L2": 0.0, "consistency": 0.0,
                "contradiction": 0.0, "coverage": 0.0, "n_claims": 0}
    r = _chat_json(L2_PROMPT, f"INSTANCE:\n{inst}\n\nVERIFIED FACTS:\n{facts}"
                   f"\n\nREPORT:\n{text}\n\nExtract and verify the claims; "
                   "return the JSON.")
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
            "contradiction": round(contra, 3), "coverage": round(coverage, 3),
            "n_claims": n,
            "L2": round(0.50 * cons + 0.25 * coverage + 0.25 * (1 - contra), 3),
            "judge_tokens": r["res"].get("completion_tokens", 0)}


def score_l3(text: str, facts: str, inst: str) -> dict:
    if not text or not text.strip():
        return {"ok": False, "faithfulness": 0, "completeness": 0,
                "coherence": 0, "actionability": 0,
                "coverage": {k: 0 for k in "what why when where who how".split()},
                "L3": 0.75}
    r = _chat_json(L3_RUBRIC, f"INSTANCE:\n{inst}\n\nVERIFIED FACTS (ground "
                   f"truth):\n{facts}\n\nREPORT TO EVALUATE:\n{text}\n\n"
                   "Return the JSON scores.")
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


def run_layer(layer: str, workers: int = 4) -> None:
    gt = json.load(open(RUNS / "bearing_gt.json", encoding="utf-8"))
    v3, _ = load_v3()
    out = RUNS / f"bearing_{layer}.parquet"
    done = set()
    if out.exists():
        done = set(map(tuple, pd.read_parquet(out)[["system", "instance_id"]]
                       .to_records(index=False)))
    tasks = []
    for sys_id in SYSTEMS:
        rep = pd.read_parquet(RUNS / f"bearing_reports_{sys_id}.parquet")
        for _, r in rep.iterrows():
            if (sys_id, r.instance_id) in done:
                continue
            g = gt[r.instance_id]
            facts = "\n".join(f"{k['category']}: {k['answer']}" for k in g["kus"])
            rec = v3[v3.file == r.instance_id].iloc[0]
            inst = (f"record {r.instance_id}, shaft speed {rec.rpm:.0f} rpm, "
                    f"load {g['load_hp']} hp, verified component: {g['truth']}")
            tasks.append((sys_id, r.instance_id, str(r.text), facts, inst))
    print(f"{layer}: {len(tasks)} to judge ({len(done)} done)")
    if not tasks:
        return
    lock, rows = threading.Lock(), []

    def work(t):
        sys_id, iid, text, facts, inst = t
        s = score_l2(text, facts, inst) if layer == "l2" else score_l3(
            text, facts, inst)
        return {"system": sys_id, "instance_id": iid, **s}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        persisted = 0  # rows of THIS run already written (append-dup guard)
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                t = futs[fut]
                row = {"system": t[0], "instance_id": t[1], "ok": False,
                       "error": str(e)}
            with lock:
                rows.append(row)
                if i % 20 == 0 or i == len(tasks):
                    new = pd.DataFrame(rows[persisted:])
                    full = new if not out.exists() else pd.concat(
                        [pd.read_parquet(out), new], ignore_index=True)
                    full.to_parquet(out, index=False)
                    persisted = len(rows)
                    print(f"  {layer}: {i}/{len(tasks)} "
                          f"(parquet {len(full)} rows)")
    new = pd.DataFrame(rows[persisted:])
    if len(new):
        full = new if not out.exists() else pd.concat(
            [pd.read_parquet(out), new], ignore_index=True)
        full.to_parquet(out, index=False)
    print(f"{layer}: complete -> {out.name} "
          f"({len(pd.read_parquet(out))} rows)")


def summary() -> None:
    l1 = pd.read_csv(RUNS / "bearing_l1.csv")
    l2 = pd.read_parquet(RUNS / "bearing_l2.parquet")
    l3 = pd.read_parquet(RUNS / "bearing_l3.parquet")
    v3, _ = load_v3()

    def l1_of(rec):
        return (0.30 * (rec.numeric if pd.notna(rec.numeric) else 0)
                + 0.30 * (rec.rule if pd.notna(rec.rule) else 0)
                + 0.40 * (rec.classification if pd.notna(rec.classification)
                          else 0))

    l1["L1"] = l1.apply(l1_of, axis=1)
    m = (l1.groupby("system")
         .agg(L1=("L1", "mean"), n=("instance_id", "count")).reset_index())
    m = m.merge(l2.groupby("system").agg(L2=("L2", "mean")).reset_index(),
                on="system")
    m = m.merge(l3.groupby("system").agg(L3=("L3", "mean"),
                                         cov=("cov_count", "mean"),
                                         faith=("faithfulness", "mean"),
                                         act=("actionability", "mean")),
                on="system")
    m["Final"] = 0.35 * 100 * m.L1 + 0.30 * 100 * m.L2 + 0.35 * 20 * m.L3
    order = {s: i for i, s in enumerate(SYSTEMS)}
    m = m.sort_values("system", key=lambda s: s.map(order))
    print(m.round(3).to_string(index=False))
    m.to_csv(RUNS / "bearing_main_summary.csv", index=False)
    print("-> code/runs/bearing_main_summary.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.run:
        run_layer("l2", args.workers)
        run_layer("l3", args.workers)
    if args.summary:
        summary()


if __name__ == "__main__":
    main()
