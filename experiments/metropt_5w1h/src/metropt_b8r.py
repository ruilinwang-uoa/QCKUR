"""
metropt_b8r.py -- B8R repair-loop ablation: generation -> DETERMINISTIC
checker (L1 rules only; no judge in the loop, anti-circularity red line)
-> targeted repair -> re-check (max 2 repair rounds).

Tests whether the architecture's native self-check recovers the
verbalization residual (numeric errors + true softenings), i.e. whether
the bound can be tightened end-to-end. Judge layers (L2/L3) are applied
AFTERWARDS by the standard evaluation, never inside generation.

Usage: python metropt_b8r.py [--workers 4] [--smoke N]
Output: runs/metropt3/reports_metropt_B8R.parquet
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
from metropt_campaign import PROVIDER, SEED, load_windows
from metropt_handlers import handle, load_spec
from metropt_l1 import l1_check
from metropt_render import (METROPT_BACKGROUND, REPORT_TASK, VEHICLE_DESC,
                            b1_text, slot_fill)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"

REPAIR_TASK = (
    "Your report was checked by a deterministic verifier against the "
    "verified facts. FIX ONLY the issues listed below; keep every other "
    "sentence unchanged. Return the corrected full report only.")


def _l1_complaints(c: dict) -> list[str]:
    out = []
    for bad in c["numeric"]["bad"]:
        out.append(f"- The number '{bad['value']}' in \"{bad['context']}\" "
                   "is not verifiable against the verified facts; correct "
                   "it to the matching fact value or remove the claim.")
    cc = c["classification"]
    if not cc["ok"] and not cc["ambiguous"]:
        out.append(f"- The telemetry-implied state is '{cc['expected']}' "
                   f"but the report's opening claims '{cc['claimed']}'; "
                   "state the correct current state.")
    for iss in c["rule"]["issues"]:
        out.append(f"- Rule-consistency: {iss}.")
    return out


def gen_b8r(feat, verdict, spec, vehicle, max_rounds: int = 2) -> dict:
    slots = slot_fill(verdict, feat, spec, vehicle)
    findings = b1_text(slots, verdict["label"])
    base_prompt = (METROPT_BACKGROUND + "\n\nWINDOW: 24 h ending "
                   f"{feat.get('t_end')} on {VEHICLE_DESC[vehicle]}.\n\n"
                   "VERIFIED ANALYTICAL FINDINGS (computed by the knowledge "
                   "units and two-tier handler; use ONLY these facts):\n"
                   + findings + "\n\nTASK:\n" + REPORT_TASK +
                   "\nThe findings are authoritative; verbalize them into a "
                   "fluent report without changing any value.")
    res = llm_call.chat(PROVIDER, [
        {"role": "system",
         "content": "You are a predictive-maintenance report writer."},
        {"role": "user", "content": base_prompt}], seed=SEED, max_tokens=700)
    text = res["text"]
    total_tok = {"p": res.get("prompt_tokens", 0),
                 "c": res.get("completion_tokens", 0)}
    cost = res.get("cost_usd", 0.0)
    rounds, complaints_log = 0, []
    c = l1_check(text, feat, verdict, spec, vehicle)
    while (not c["pass"]) and rounds < max_rounds:
        rounds += 1
        complaints = _l1_complaints(c)
        complaints_log.append(complaints)
        if not complaints:
            break
        rep = llm_call.chat(PROVIDER, [
            {"role": "system",
             "content": "You are a predictive-maintenance report writer."},
            {"role": "user",
             "content": base_prompt + "\n\nYOUR DRAFT:\n" + text +
             "\n\nVERIFIER ISSUES:\n" + "\n".join(complaints) +
             "\n\n" + REPAIR_TASK}],
            seed=SEED + rounds, max_tokens=700)
        if not rep.get("ok") or not rep.get("text"):
            break
        text = rep["text"]
        total_tok["p"] += rep.get("prompt_tokens", 0)
        total_tok["c"] += rep.get("completion_tokens", 0)
        cost += rep.get("cost_usd", 0.0)
        c = l1_check(text, feat, verdict, spec, vehicle)
    return {"text": text, "n_repair_rounds": rounds,
            "final_pass": c["pass"],
            "final_n_bad": c["numeric"]["n_bad"],
            "complaints_log": json.dumps(complaints_log)[:2000],
            "prompt_tokens": total_tok["p"],
            "completion_tokens": total_tok["c"],
            "cost_usd": round(cost, 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=None)
    args = ap.parse_args()
    wins, spec = load_windows(args.smoke)
    out = RUNS / "reports_metropt_B8R.parquet"
    done, rows = set(), []
    if out.exists():
        old = pd.read_parquet(out)
        done = {(r.vehicle, r.window_id) for r in
                old[(old.ok.astype(str) == "True") |
                    (old.text.fillna("") != "")].itertuples()}
        old = old.drop_duplicates(subset=["vehicle", "window_id"],
                                  keep="last")
        rows = old.to_dict("records")
    tasks = [(wid, f.to_dict()) for wid, f in wins.iterrows()
             if (f["vehicle"], wid) not in done]
    print(f"B8R: {len(tasks)} to generate ({len(done)} done)")
    if not tasks:
        return
    lock = threading.Lock()
    persisted = len(rows)

    def work2(t):
        wid, feat = t
        vehicle = feat["vehicle"]
        verdict = handle(feat, vehicle, spec)
        r = gen_b8r(feat, verdict, spec, vehicle)
        text = r.pop("text")
        r.update({"system_id": "B8R", "window_id": wid,
                  "vehicle": vehicle, "kind": feat["kind"],
                  "event_id": feat.get("event_id"),
                  "offset_h": feat.get("offset_h"),
                  "verdict_label": verdict["label"],
                  "verdict_tier": verdict["tier"],
                  "ok": bool(text), "text": text})
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work2, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            with lock:
                rows.append(row)
                if i % 10 == 0 or i == len(tasks):
                    new = pd.DataFrame(rows[persisted:])
                    full = new if not out.exists() else pd.concat(
                        [pd.read_parquet(out), new], ignore_index=True)
                    full = full.drop_duplicates(
                        subset=["vehicle", "window_id"], keep="last")
                    full.to_parquet(out, index=False)
                    persisted = len(rows)
                    print(f"  {i}/{len(tasks)}")
    d = pd.read_parquet(out).drop_duplicates(subset=["vehicle", "window_id"],
                                             keep="last")
    d.to_parquet(out, index=False)
    n_rep = (d.n_repair_rounds > 0).sum()
    print(f"B8R: {len(d)} rows; repair triggered on {n_rep}; "
          f"final L1 pass {d.final_pass.sum() if 'final_pass' in d else '?'}; "
          f"avg rounds {d.n_repair_rounds.mean():.2f}")


if __name__ == "__main__":
    main()
