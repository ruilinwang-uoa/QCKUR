"""Re-judge the zero-claim L2 rows with a format-robust extraction instruction.

The AI4I L2 judge extracted 0 claims from 18/100 B8 and 8/100 B2 reports
(numbered-markdown formatting), flooring those instances at L2 = 0.25. This
script re-judges exactly those 26 reports under the ORIGINAL campaign
protocol (same L2 prompt, DeepSeek-V4-Flash judge, seed 42, max_tokens 1500,
same facts/instance construction via background.canonical_5w1h) plus one
extraction-robustness addendum, and reports the measured (not bracketed)
effect on B8/B2 L2.

Output: code/runs/l2_zero_claim_rejudge.csv + console summary.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_call  # noqa: E402
import background as bg  # noqa: E402
import config as C  # noqa: E402
from layer2_eval import PROMPT  # noqa: E402

ADDENDUM = ("\nNOTE: the report may use markdown bold headers, numbered "
            "sections, or bracketed aspect tags. Extract claims from EVERY "
            "section regardless of its formatting; a section with substantive "
            "content yields at least one claim.")

PROVIDER, SEED, MAXTOK = "deepseek", 42, 3000
# 3000 required: at 1500 the judge's chain of thought consumed the budget
# before the JSON on these reports (verified 2026-09-09: first pass at 1500
# recovered 5/26 perfectly and left 21 with empty verdicts). Same remedy as
# the bearing campaign; disclosed in the paper.


def main() -> None:
    l2 = pd.read_parquet(C.RUNS_DIR / "reports_main_glm_n100_B1B2B8_v3_l2.parquet")
    zero = l2[l2.n_claims == 0][["method", "instance_id"]]
    rep = pd.read_parquet(C.RUNS_DIR / "reports_main_glm_n100_B1B2B8_v3.parquet")
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    ctx = bg.build_ctx()
    print(f"{len(zero)} zero-claim rows to re-judge")

    rows = []
    for _, z in zero.iterrows():
        r = rep[(rep.method == z.method)
                & (rep.instance_id == z.instance_id)].iloc[0]
        inst = sample[sample.sample_id == z.instance_id].iloc[0].to_dict()
        facts = "\n".join(f"{k['category']}: {k['answer']}"
                          for k in bg.canonical_5w1h(inst, ctx))
        user = (f"INSTANCE:\n{bg.instance_context(inst)}\n\nVERIFIED FACTS:\n"
                f"{facts}\n\nREPORT:\n{r['raw_output']}\n\nExtract and verify "
                "the claims; return the JSON.")
        res = llm_call.chat(PROVIDER, [
            {"role": "system", "content": PROMPT + ADDENDUM},
            {"role": "user", "content": user}],
            response_format={"type": "json_object"}, seed=SEED,
            max_tokens=MAXTOK)
        import json
        try:
            j = json.loads(res["text"])
            claims = j.get("claims", []) or []
            n = len(claims)
            cons = (sum(1 for c in claims if c.get("verdict") == "consistent")
                    / n if n else 0.0)
            contra = (sum(1 for c in claims
                          if c.get("verdict") == "contradicts") / n if n else 0.0)
            cov = j.get("coverage") or {}
            coverage = (sum(int(cov.get(k, 0)) for k in
                            ["what", "why", "when", "where", "who", "how"]) / 6.0
                        if isinstance(cov, dict) else 0.0)
            L2 = round(0.50 * cons + 0.25 * coverage + 0.25 * (1 - contra), 3)
        except Exception:  # noqa: BLE001
            n, L2 = 0, 0.25
            cons = contra = coverage = 0.0
        rows.append({"method": z.method, "instance_id": z.instance_id,
                     "n_claims_new": n, "L2_new": L2,
                     "consistency_new": round(cons, 3),
                     "contradiction_new": round(contra, 3),
                     "coverage_new": round(coverage, 3)})
        print(f"  {z.method} {z.instance_id}: claims={n} L2={L2}", flush=True)
        time.sleep(0.3)

    out = pd.DataFrame(rows)
    out.to_csv(C.RUNS_DIR / "l2_zero_claim_rejudge.csv", index=False)
    print("\nsummary:")
    for m in ["B2", "B8"]:
        d = l2[l2.method == m]
        nz = len(d[d.n_claims == 0])
        old = d.L2.mean()
        new_rows = out[out.method == m]
        fixed = new_rows.L2_new.mean() if len(new_rows) else 0.0
        measured = (old * (100 - nz) + fixed * nz) / 100
        print(f"  {m}: zero-claim {nz}, old L2 {old:.3f}, "
              f"rejudged mean {fixed:.3f}, measured L2 {measured:.3f}")
    print("-> code/runs/l2_zero_claim_rejudge.csv")


if __name__ == "__main__":
    main()
