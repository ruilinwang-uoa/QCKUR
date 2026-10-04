"""One-shot live test of the Kimi (SiliconFlow) judge configuration.

1. Plain chat call  -> connectivity, auth, latency
2. L3 rubric path   -> response_format json_object + seed + coverage-dict parsing
3. L2 rubric path   -> claims-extraction JSON schema
Uses the same code path as the judging scripts (llm_call.chat / score_one).
"""
import json
import time

import llm_call

PROVIDER = "kimi"

print("=== 1. plain connectivity call ===")
t0 = time.time()
r = llm_call.chat(PROVIDER, [{"role": "user", "content": "Reply with the single word: ready"}],
                  max_tokens=100)
print(f"ok={r['ok']} model={r['model']} latency={r['latency_s']}s "
      f"tokens={r['prompt_tokens']}+{r['completion_tokens']}")
print(f"text={r['text'][:80]!r} error={r['error']}")
if not r["ok"]:
    raise SystemExit("connectivity FAILED - see error above")

print("\n=== 2. L3 rubric path (response_format + seed) ===")
import layer3_eval

facts = ("what: no fault mode detected; active modes: 0\n"
         "why: all operating variables within safe envelopes\n"
         "when: routine monitoring; no intervention needed\n"
         "where: cutting tool shows tool wear 206 min (band 200-240)\n"
         "who: operator; severity none\n"
         "how: continue routine monitoring")
ctx = ("AI4I 2020 sample s0038 (UDI 1088, variant H): air temp 296.9 K, process temp "
       "307.8 K, rotational speed 1549 rpm, torque 35.8 Nm, tool wear 206 min.")
report = ("1. Fault/Status: No fault mode detected. 2. Cause: All operating variables "
          "are within safe limits. 3. Action Required: No intervention; continue "
          "routine monitoring. 4. Problem Location: Cutting tool (wear 206 min). "
          "5. Responder: Operator, severity none. 6. Recommended Action: Continue "
          "routine monitoring.")
t0 = time.time()
s = layer3_eval.score_one(report, facts, ctx, PROVIDER, seed=42)
print(f"L3 = {layer3_eval.l3_from(s)} | ok={s['ok']} judge_tokens={s.get('judge_tokens')}")
print(f"scores: faith={s['faithfulness']} compl={s['completeness']} coh={s['coherence']} "
      f"act={s['actionability']} cov={s['coverage']}")
if not s["ok"] or s["faithfulness"] == 0:
    print(f"RAW OUTPUT (first 400 chars): {s.get('raw', '')[:400]!r}")

print("\n=== 3. L2 rubric path ===")
import layer2_eval

s2 = layer2_eval.score_one(report, facts, ctx, PROVIDER, seed=42)
print(f"L2 = {s2['L2']:.3f} | ok={s2['ok']} n_claims={s2['n_claims']} "
      f"cons={s2['consistency']:.2f} contra={s2['contradiction']:.2f}")
if not s2["ok"] or s2["n_claims"] == 0:
    print("L2 returned no claims - inspect the judge output manually if repeated.")

print("\nALL TESTS DONE")
