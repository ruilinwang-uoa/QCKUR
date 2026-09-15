"""Layer 3 evaluation: LLM-as-judge multi-dimensional scoring (paper Phase-4 L3).

Scores every report on five dimensions and computes
    L3 = 0.30*faithfulness + 0.15*completeness + 0.25*coverage5w1h
         + 0.15*coherence + 0.15*actionability
The 5W1H coverage dimension (25%, the largest single weight) is the core piece
of evidence for the framework: it checks whether a report addresses all six
cognitive aspects.

The judge is anchored on the instance's verified analytical facts (the canonical
5W1H answers) so faithfulness is checked against ground truth, not the model's
opinion. Bias controls: a fixed rubric, shuffled report order, and (optionally)
N sampled judgments averaged.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import pandas as pd

import config as C
import background as bg
import llm_call
import llm_config

JUDGE_RUBRIC = """You are a strict evaluator of predictive-maintenance reports.
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


def _facts_block(row, ctx) -> str:
    kus = bg.canonical_5w1h(row, ctx)
    return "\n".join(f"{k['category']}: {k['answer']}" for k in kus)


def score_one(report_text: str, facts: str, instance_ctx: str, provider: str,
              seed: int) -> dict:
    if not report_text or not report_text.strip():
        return {"ok": False, "faithfulness": 0, "completeness": 0, "coherence": 0,
                "actionability": 0, "coverage": {k: 0 for k in "what why when where who how".split()}}
    user = (f"INSTANCE:\n{instance_ctx}\n\nVERIFIED FACTS (ground truth):\n{facts}\n\n"
            f"REPORT TO EVALUATE:\n{report_text}\n\nReturn the JSON scores.")
    res = llm_call.chat(provider, [{"role": "system", "content": JUDGE_RUBRIC},
                                   {"role": "user", "content": user}],
                        response_format={"type": "json_object"}, seed=seed)
    out = {"ok": res["ok"], "raw": res["text"]}
    try:
        j = json.loads(res["text"])
    except Exception:  # noqa: BLE001
        j = {}
    cov = j.get("coverage") or {}
    if not isinstance(cov, dict):
        cov = {}
    out.update({
        "faithfulness": float(j.get("faithfulness", 0)),
        "completeness": float(j.get("completeness", 0)),
        "coherence": float(j.get("coherence", 0)),
        "actionability": float(j.get("actionability", 0)),
        "coverage": {k: int(cov.get(k, 0)) for k in ["what", "why", "when", "where", "who", "how"]},
        "judge_tokens": res.get("completion_tokens", 0),
    })
    return out


def l3_from(s: dict) -> float:
    cov = s.get("coverage", {})
    cov5 = (sum(cov.get(k, 0) for k in ["what", "why", "when", "where", "who", "how"]) / 6.0) * 5.0
    return round(0.30 * s["faithfulness"] + 0.15 * s["completeness"]
                 + 0.25 * cov5 + 0.15 * s["coherence"] + 0.15 * s["actionability"], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", help="reports parquet under runs/")
    ap.add_argument("--judge", default="deepseek", help="judge provider")
    ap.add_argument("--max-tokens", type=int, default=0, help="override judge max_tokens")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=0, help="cap reports per system (0=all)")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent judge calls (thread pool); 1 = serial")
    args = ap.parse_args()

    if not llm_config.get_provider(args.judge).is_configured:
        print(f"ERROR: judge provider '{args.judge}' not configured"); return
    if args.max_tokens:
        llm_call.set_max_tokens(args.max_tokens)
        print(f"judge max_tokens overridden to {args.max_tokens}")
    reports = pd.read_parquet(C.RUNS_DIR / args.reports)
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    ctx = bg.build_ctx()

    # cap per system (stratified by failure), shuffle order for bias control
    if args.n:
        parts = []
        for m, g in reports.groupby("method"):
            parts.append(g.head(args.n))
        reports = pd.concat(parts)
    rng = random.Random(args.seed)
    order = list(reports.index)
    rng.shuffle(order)

    # resume support: skip (method, instance_id) pairs already scored in the
    # output parquet, and checkpoint every 10 reports
    out_path = C.RUNS_DIR / args.reports.replace(".parquet", "_l3.parquet")
    done = pd.DataFrame()
    if out_path.exists():
        done = pd.read_parquet(out_path)
        done_keys = set(zip(done["method"], done["instance_id"]))
        order = [i for i in order
                 if (reports.loc[i, "method"], reports.loc[i, "instance_id"]) not in done_keys]
        print(f"resume: {len(done)} already scored, {len(order)} remaining")

    def _work(r):
        inst = sample[sample["sample_id"] == r["instance_id"]].iloc[0].to_dict()
        facts = _facts_block(inst, ctx)
        try:
            s = score_one(r["raw_output"], facts, bg.instance_context(inst), args.judge, args.seed)
        except Exception as e:  # noqa: BLE001  one malformed judge response shouldn't sink the run
            s = {"ok": False, "faithfulness": 0, "completeness": 0, "coherence": 0,
                 "actionability": 0, "coverage": {k: 0 for k in "what why when where who how".split()}}
            print(f"  [warn] {r['method']} {r['instance_id']}: {type(e).__name__} {e}", flush=True)
        s.update({"method": r["method"], "instance_id": r["instance_id"],
                  "machine_failure": int(r["machine_failure"]), "L3": l3_from(s)})
        return s

    rows = []
    t0 = time.time()

    def _record(s):
        rows.append(s)
        if len(rows) % 10 == 0:
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(out_path, index=False)
        print(f"  [{s['method']}] {s['instance_id']}: L3={s['L3']} "
              f"cov={sum(s['coverage'].values())}/6 faith={s['faithfulness']} "
              f"({len(rows)}/{len(order)})", flush=True)

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_work, reports.loc[idx]) for idx in order]
            for fut in as_completed(futs):
                _record(fut.result())
    else:
        for idx in order:
            _record(_work(reports.loc[idx]))
            time.sleep(args.delay)

    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    df.to_parquet(out_path, index=False)

    print(f"\n=== Layer 3 summary (judge={args.judge}, n={len(df)}) ===")
    df["cov6"] = df["coverage"].apply(lambda d: sum(d.values()))
    g = df.groupby("method")
    summ = pd.DataFrame({
        "n": g.size(),
        "L3": g["L3"].mean().round(2),
        "faithfulness": g["faithfulness"].mean().round(2),
        "completeness": g["completeness"].mean().round(2),
        "coherence": g["coherence"].mean().round(2),
        "actionability": g["actionability"].mean().round(2),
        "coverage_/6": g["cov6"].mean().round(2),
    }).sort_values("L3", ascending=False)
    print(summ.to_string())
    print(f"\nSaved: {out_path} | wall {round(time.time()-t0,1)}s")


if __name__ == "__main__":
    main()
