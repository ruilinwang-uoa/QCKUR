"""Layer 2 evaluation: QA closed-loop / internal consistency (paper Phase-4 L2).

For each report the judge extracts its factual claims, tags each with a 5W1H
category, and verifies it against the instance's VERIFIED FACTS (the canonical
5W1H answers). From the verdicts:
  * consistency  = consistent / total claims
  * contradiction = contradicts / total claims
  * coverage     = 5W1H aspects addressed / 6
  L2 = 0.50*consistency + 0.25*coverage + 0.25*(1 - contradiction)   (in 0-1)

Cross-judge (the OTHER model judges) avoids self-bias, matching Layer 3.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

import config as C
import background as bg
import llm_call
import llm_config

PROMPT = """You are a strict consistency checker for predictive-maintenance reports.
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


def score_one(text, facts, inst_ctx, provider, seed):
    if not text or not text.strip():
        return {"ok": False, "L2": 0.0, "consistency": 0.0, "contradiction": 0.0,
                "coverage": 0.0, "n_claims": 0}
    user = (f"INSTANCE:\n{inst_ctx}\n\nVERIFIED FACTS:\n{facts}\n\nREPORT:\n{text}\n\n"
            "Extract and verify the claims; return the JSON.")
    res = llm_call.chat(provider, [{"role": "system", "content": PROMPT},
                                   {"role": "user", "content": user}],
                        response_format={"type": "json_object"}, seed=seed)
    out = {"ok": res["ok"]}
    try:
        j = json.loads(res["text"])
    except Exception:  # noqa: BLE001
        j = {}
    claims = j.get("claims", []) or []
    n = len(claims)
    if n:
        cons = sum(1 for c in claims if c.get("verdict") == "consistent") / n
        contra = sum(1 for c in claims if c.get("verdict") == "contradicts") / n
    else:
        cons, contra = 0.0, 0.0
    cov = j.get("coverage") or {}
    if not isinstance(cov, dict):
        cov = {}
    coverage = sum(int(cov.get(k, 0)) for k in ["what", "why", "when", "where", "who", "how"]) / 6.0
    out.update({"consistency": round(cons, 3), "contradiction": round(contra, 3),
                "coverage": round(coverage, 3), "n_claims": n,
                "L2": round(0.50 * cons + 0.25 * coverage + 0.25 * (1 - contra), 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", help="reports parquet under runs/")
    ap.add_argument("--judge", default="deepseek")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent judge calls (thread pool); 1 = serial")
    args = ap.parse_args()
    if not llm_config.get_provider(args.judge).is_configured:
        print(f"ERROR: judge '{args.judge}' not configured"); return
    if args.max_tokens:
        llm_call.set_max_tokens(args.max_tokens)

    rep = pd.read_parquet(C.RUNS_DIR / args.reports)
    if args.n:
        rep = pd.concat([g.head(args.n) for _, g in rep.groupby("method")])
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    ctx = bg.build_ctx()

    # resume support: skip (method, instance_id) pairs already scored in the
    # output parquet, and checkpoint every 10 reports so a long judging run
    # survives interruptions
    out = C.RUNS_DIR / args.reports.replace(".parquet", "_l2.parquet")
    done = pd.DataFrame()
    if out.exists():
        done = pd.read_parquet(out)
        done_keys = set(zip(done["method"], done["instance_id"]))
        rep = rep[~pd.Series([(m, i) in done_keys for m, i in zip(rep["method"], rep["instance_id"])],
                             index=rep.index)]
        print(f"resume: {len(done)} already scored, {len(rep)} remaining")

    def _work(r):
        inst = sample[sample["sample_id"] == r["instance_id"]].iloc[0].to_dict()
        facts = "\n".join(f"{k['category']}: {k['answer']}" for k in bg.canonical_5w1h(inst, ctx))
        try:
            s = score_one(r["raw_output"], facts, bg.instance_context(inst), args.judge, args.seed)
        except Exception as e:  # noqa: BLE001  one malformed judge response shouldn't sink the run
            s = {"ok": False, "L2": 0.0, "consistency": 0.0, "contradiction": 0.0,
                 "coverage": 0.0, "n_claims": 0}
            print(f"  [warn] {r['method']} {r['instance_id']}: {type(e).__name__} {e}", flush=True)
        s.update({"method": r["method"], "instance_id": r["instance_id"]})
        return s

    rows = []
    t0 = time.time()

    def _record(s):
        rows.append(s)
        if len(rows) % 10 == 0:
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(out, index=False)
        print(f"  [{s['method']}] {s['instance_id']}: L2={s['L2']} "
              f"cons={s['consistency']} contra={s['contradiction']} cov={s['coverage']} "
              f"claims={s['n_claims']} ({len(rows)}/{len(rep)})", flush=True)

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        recs = [r for _, r in rep.iterrows()]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_work, r) for r in recs]
            for fut in as_completed(futs):
                _record(fut.result())
    else:
        for _, r in rep.iterrows():
            _record(_work(r))
            time.sleep(args.delay)

    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"\n=== Layer 2 summary (judge={args.judge}, n={len(df)}) ===")
    g = df.groupby("method").agg(L2=("L2", "mean"), cons=("consistency", "mean"),
                                 contra=("contradiction", "mean"), cov=("coverage", "mean"),
                                 claims=("n_claims", "mean")).round(3)
    g = g.reindex([m for m in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"] if m in g.index])
    print(g.to_string())
    print(f"\nSaved: {out} | wall {round(time.time()-t0,1)}s")


if __name__ == "__main__":
    main()
