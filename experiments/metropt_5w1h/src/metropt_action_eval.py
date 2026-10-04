"""
metropt_action_eval.py -- decision-usefulness scoring: a dedicated blind
action classifier per report (the L1 keyword matcher proved too brittle
at 3-7% -- "Inspect at the next window" fell through every pattern).

The classifier sees ONLY the report text (+ telemetry digest context,
no GT) and assigns one action class from the pre-registered taxonomy:
  remove_or_urgent_inspection | schedule_inspection |
  monitor_and_verify | no_action_or_monitor
Matched against the GT action class (window visibility x failure class
x disposition, frozen in metropt_gt.json).

Usage: python metropt_action_eval.py [--workers 4]
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

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
SYSTEMS = ["B1", "B2", "B3", "B5", "B8"]
PROVIDER = "deepseek"
SEED = 42
CLASSES = ["remove_or_urgent_inspection", "schedule_inspection",
           "monitor_and_verify", "no_action_or_monitor"]

PROMPT = """You classify the RECOMMENDED ACTION in a predictive-maintenance report.
Choose exactly one class:
- "remove_or_urgent_inspection": remove the vehicle/unit from service now,
  or perform an urgent inspection within hours
- "schedule_inspection": schedule an inspection within the next
  maintenance window (~24-48 h)
- "monitor_and_verify": keep monitoring / verify on site at the next
  opportunity, no firm maintenance scheduling yet
- "no_action_or_monitor": routine operation, no specific action requested
Return STRICT JSON: {"action": "<class>", "confidence": 0-1}
"""


def classify(text: str, digest: str) -> dict:
    if not text or not text.strip():
        return {"ok": False, "action": None}
    res, err = None, None
    for attempt in range(3):
        res = llm_call.chat(PROVIDER, [
            {"role": "system", "content": PROMPT},
            {"role": "user",
             "content": f"TELEMETRY CONTEXT:\n{digest}\n\nREPORT:\n"
                        f"{text}\n\nReturn the JSON."}],
            response_format={"type": "json_object"}, seed=SEED,
            max_tokens=1200)
        if res.get("ok") and res.get("text"):
            try:
                j = json.loads(res["text"])
                act = j.get("action")
                if act in CLASSES:
                    return {"ok": True, "action": act,
                            "confidence": j.get("confidence"),
                            "judge_tokens": res.get("completion_tokens", 0)}
                err = "bad-class"
            except Exception:  # noqa: BLE001
                err = "json-parse"
        else:
            err = str(res.get("error"))[:80]
        time.sleep(2 * (attempt + 1))
    return {"ok": False, "action": None, "error": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from metropt_l2l3_eval import feat_row
    gt = {f"{g['vehicle']}::{g['window_id']}": g for g in json.load(
        open(RUNS / "metropt_gt.json", encoding="utf-8"))["windows"]}
    out = RUNS / "metropt_actions.parquet"
    done = set()
    if out.exists():
        old = pd.read_parquet(out)
        done = {(r.system, r.vehicle, r.window_id) for r in
                old[old.ok.astype(str) == "True"].itertuples()}
        old = old.drop_duplicates(subset=["system", "vehicle", "window_id"],
                                  keep="last")
        rows = old.to_dict("records")
    else:
        rows = []
    tasks = []
    for s in SYSTEMS:
        rep = pd.read_parquet(RUNS / f"reports_metropt_{s}.parquet")
        for _, r in rep.iterrows():
            if (s, r.vehicle, r.window_id) in done:
                continue
            feat = feat_row(r.vehicle, r.window_id)
            tasks.append((s, r.vehicle, r.window_id, str(r.text), feat))
    print(f"action classify: {len(tasks)} to do ({len(done)} done)")
    if not tasks:
        return
    from metropt_render import digest_text
    lock = threading.Lock()

    def work(t):
        s, veh, wid, text, feat = t
        c = classify(text, digest_text(feat, veh))
        g = gt[f"{veh}::{wid}"]
        return {"system": s, "vehicle": veh, "window_id": wid,
                "action_claim": c.get("action"),
                "gt_action": g["expected"]["action_class"],
                "match": int(c.get("action")
                             == g["expected"]["action_class"]),
                "ok": c.get("ok", False)}

    persisted = len(rows)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            with lock:
                rows.append(row)
                if i % 25 == 0 or i == len(tasks):
                    new = pd.DataFrame(rows[persisted:])
                    full = new if not out.exists() else pd.concat(
                        [pd.read_parquet(out), new], ignore_index=True)
                    full.to_parquet(out, index=False)
                    persisted = len(rows)
                    print(f"  {i}/{len(tasks)}")
    d = pd.read_parquet(out)
    d = d.drop_duplicates(subset=["system", "vehicle", "window_id"],
                          keep="last")
    d.to_parquet(out, index=False)
    ok = d[d.ok.astype(str) == "True"]
    print("=== decision-usefulness (blind action classifier vs GT) ===")
    print(ok.groupby("system").match.agg(["count", "sum", "mean"])
          .round(3).to_string())
    xt = pd.crosstab(ok.gt_action, ok.action_claim)
    print("\nGT x claimed confusion:")
    print(xt.to_string())


if __name__ == "__main__":
    main()
