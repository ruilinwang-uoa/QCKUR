"""Out-of-coverage (OOC) rejection test for the question-matching mechanism.

The AI4I instantiation maps each report aspect one-to-one to a predefined
knowledge unit, so positive matching is 1.00 by construction and the negative
class is never exercised.  This test runs the framework's question-matching
step (as specified in the paper: the LLM must return the exact text of one
abstract question from the bank, or the token NO MATCH) on:

  * 40 in-coverage questions: paraphrases of bank questions taken verbatim
    from the question bank's own paraphrase sets (one per sampled question,
    stratified across the six 5W1H categories);
  * 40 out-of-coverage questions: realistic maintenance questions that the
    per-instance analytics bank does not cover (cost, vendors, scheduling,
    fleet comparison, history, staffing, regulation, ...).

Reports: positive matching accuracy (exact-text match to the intended bank
question), wrong-match rate, and OOC rejection rate.  Provider: the primary
generation model, temperature 0.3, seed 42 (the framework defaults).
"""
from __future__ import annotations

import argparse
import json
import random
import time

import config as C
import llm_call
import llm_config

MATCH_SYSTEM = """You are a semantic matching assistant for an analytics question bank.
Given a user question and the question bank, decide which single abstract
question in the bank the user question semantically matches.  Return ONLY the
exact text of that bank question, copied verbatim, or the token NO MATCH if no
bank question matches.  Output nothing else."""

# 40 out-of-coverage maintenance questions (outside the per-instance analytics
# bank: cost, procurement, scheduling, fleet, history, staffing, regulation...)
OOC_QUESTIONS = [
    "How much will this unplanned stop cost us in lost output?",
    "What is the annual maintenance budget for this production line?",
    "Which supplier should we order replacement cutting tools from?",
    "Are spare spindles currently in stock in the warehouse?",
    "When is the next planned maintenance window for this machine?",
    "How does this machine's failure rate compare to the other machines in the plant?",
    "What is the OEE of line 3 this quarter?",
    "How many downtime events were recorded across the fleet last month?",
    "Which operator was on shift when the fault occurred?",
    "Who is certified to repair the spindle drive on this model?",
    "Do we need to file an incident report for this event?",
    "What does the ISO 55001 standard require for our asset registers?",
    "Should we buy the extended warranty for this machine?",
    "What was the total energy bill for the workshop last year?",
    "How do we renegotiate the service contract with the OEM?",
    "What is the carbon footprint of running this machine for a day?",
    "Which production order should we prioritize after this stoppage?",
    "How do we adjust the ERP maintenance module to log this event?",
    "What training does a new technician need before operating this machine?",
    "Is there a firmware update available for the machine controller?",
    "How do weather conditions in the plant affect sensor readings?",
    "Can we claim insurance for this failure?",
    "Which department pays for the replacement parts?",
    "How do we benchmark our maintenance program against industry peers?",
    "What is the expected resale value of this machine next year?",
    "How many operators does a shift on this line require?",
    "Which KPI dashboard shows overall plant availability?",
    "How do we schedule production around next week's maintenance?",
    "What caused the quality defects reported by the inspection team?",
    "How is the maintenance team's overtime tracked and paid?",
    "Which machines are due for statutory safety inspection this year?",
    "How do we reduce the number of false alarms from the monitoring system?",
    "What is the corporate policy on repairing versus replacing old equipment?",
    "How do we integrate this machine's data with the MES?",
    "Which consultant should we hire for the lean maintenance rollout?",
    "What was last year's total spend on cutting fluid?",
    "How do customers rate the surface finish of recent batches?",
    "What is the delivery lead time for a new machine of this type?",
    "How do we document this event for the ISO audit trail?",
    "Which shift pattern minimizes fatigue-related handling errors?",
]


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--n", type=int, default=40, help="questions per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    if not llm_config.get_provider(args.provider).is_configured:
        print(f"ERROR: provider '{args.provider}' not configured"); return

    bank = json.loads((C.QUESTION_BANK_DIR / "question_bank_5w1h.json")
                      .read_text(encoding="utf-8"))["questions"]
    texts = {norm(q["text"]) for q in bank}

    rng = random.Random(args.seed)
    # in-coverage: one paraphrase per question, stratified by category
    by_cat: dict[str, list[dict]] = {}
    for q in bank:
        by_cat.setdefault(q["category"], []).append(q)
    in_cov = []
    per_cat = max(1, args.n // len(by_cat))
    for cat, qs in by_cat.items():
        for q in rng.sample(qs, min(per_cat, len(qs))):
            if q.get("paraphrases"):
                p = rng.choice(q["paraphrases"])
                in_cov.append({"question": p, "target": q["text"], "category": cat})
    rng.shuffle(in_cov)
    in_cov = in_cov[:args.n]
    ooc = [{"question": q, "target": None, "category": "OOC"}
           for q in OOC_QUESTIONS[:args.n]]

    bank_block = "\n".join(f"- {q['text']}" for q in bank)

    results = []
    t0 = time.time()
    for i, item in enumerate(in_cov + ooc, 1):
        user = (f"MODEL CONTEXT: predictive maintenance of a milling machine "
                f"(AI4I 2020 data: temperature, rotational speed, torque, tool "
                f"wear; failure modes TWF/HDF/PWF/OSF/RNF; logistic regression "
                f"/ random forest / gradient boosting models).\n\nQUESTION BANK:\n"
                f"{bank_block}\n\nUSER QUESTION: {item['question']}\n\n"
                f"Return the exact matching bank question text, or NO MATCH.")
        try:
            res = llm_call.chat(args.provider,
                                [{"role": "system", "content": MATCH_SYSTEM},
                                 {"role": "user", "content": user}],
                                seed=args.seed, temperature=0.3)
            raw = (res.get("text") or "").strip()
        except Exception as e:  # noqa: BLE001
            raw = f"__ERROR__ {e}"
        nraw = norm(raw)
        if nraw == "no match":
            pred = "NO_MATCH"
        elif nraw in texts:
            pred = raw
        else:
            pred = f"INVALID::{raw[:80]}"
        results.append({**item, "raw": raw[:200], "pred": pred})
        print(f"  [{i}/{len(in_cov)+len(ooc)}] {item['category']}: "
              f"{'OK' if item['target'] else 'ooc'} -> "
              f"{pred if pred == 'NO_MATCH' else ('match' if pred.startswith('INVALID') is False else pred)}",
              flush=True)
        time.sleep(args.delay)

    # metrics
    pos_correct = sum(1 for r in results[:len(in_cov)]
                      if norm(r["pred"]) == norm(r["target"]))
    pos_wrong = sum(1 for r in results[:len(in_cov)]
                    if r["pred"] != "NO_MATCH" and not r["pred"].startswith("INVALID")
                    and norm(r["pred"]) != norm(r["target"]))
    pos_reject = sum(1 for r in results[:len(in_cov)] if r["pred"] == "NO_MATCH")
    pos_invalid = sum(1 for r in results[:len(in_cov)]
                      if r["pred"].startswith("INVALID"))
    ooc_reject = sum(1 for r in results[len(in_cov):] if r["pred"] == "NO_MATCH")
    ooc_accept = sum(1 for r in results[len(in_cov):]
                     if r["pred"] != "NO_MATCH" and not r["pred"].startswith("INVALID"))
    ooc_invalid = sum(1 for r in results[len(in_cov):]
                      if r["pred"].startswith("INVALID"))

    report = {
        "provider": args.provider, "seed": args.seed,
        "n_in_coverage": len(in_cov), "n_ooc": len(ooc),
        "positive_accuracy": round(pos_correct / len(in_cov), 4),
        "positive_wrong_match": round(pos_wrong / len(in_cov), 4),
        "positive_rejected": round(pos_reject / len(in_cov), 4),
        "positive_invalid_output": round(pos_invalid / len(in_cov), 4),
        "ooc_rejection_rate": round(ooc_reject / len(ooc), 4),
        "ooc_wrongly_accepted": round(ooc_accept / len(ooc), 4),
        "ooc_invalid_output": round(ooc_invalid / len(ooc), 4),
        "wall_s": round(time.time() - t0, 1),
        "details": results,
    }
    out = C.RUNS_DIR / "ooc_test_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== OOC test summary ===")
    for k in ["positive_accuracy", "positive_wrong_match", "positive_rejected",
              "positive_invalid_output", "ooc_rejection_rate",
              "ooc_wrongly_accepted", "ooc_invalid_output"]:
        print(f"  {k}: {report[k]}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
