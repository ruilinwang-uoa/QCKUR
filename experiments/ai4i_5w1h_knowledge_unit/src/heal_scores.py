"""Heal empty-judge rows in Layer-2/Layer-3 score parquets.

The DeepSeek reasoning endpoint intermittently returns ok=True with EMPTY
content (finish consumed by chain-of-thought or server variance); those rows
parse as all-zero.  This script finds such rows (n_claims==0 for L2;
faithfulness==0 with empty raw for L3), re-judges them with up to 4 attempts,
and patches the rows in place.

Usage: python heal_scores.py <stem>   e.g. reports_main_glm_n100_B6dense
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import config as C
import background as bg
import layer2_eval
import layer3_eval
import llm_call
import llm_config

MAX_TOKENS = 3000
ATTEMPTS = 4


def _ctx_sample():
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    return sample, bg.build_ctx()


def heal_l2(stem: str) -> None:
    path = C.RUNS_DIR / f"{stem}_l2.parquet"
    d = pd.read_parquet(path)
    rep = pd.read_parquet(C.RUNS_DIR / f"{stem}.parquet").set_index(
        ["method", "instance_id"])
    sample, ctx = _ctx_sample()
    llm_call.set_max_tokens(MAX_TOKENS)
    bad = d.index[d["n_claims"] == 0].tolist()
    print(f"L2 {stem}: {len(bad)} empty rows of {len(d)}")
    fixed = 0
    for i in bad:
        r = d.loc[i]
        row = rep.loc[(r["method"], r["instance_id"])]
        inst = sample[sample["sample_id"] == r["instance_id"]].iloc[0].to_dict()
        facts = "\n".join(f"{k['category']}: {k['answer']}"
                          for k in bg.canonical_5w1h(inst, ctx))
        s = None
        for a in range(ATTEMPTS):
            s = layer2_eval.score_one(row["raw_output"], facts,
                                      bg.instance_context(inst), "deepseek", 42)
            if s.get("n_claims", 0) > 0:
                break
            time.sleep(1.5)
        if s and s.get("n_claims", 0) > 0:
            for k in ["L2", "consistency", "contradiction", "coverage", "n_claims"]:
                d.loc[i, k] = s[k]
            fixed += 1
        print(f"  {r['instance_id']}: n_claims={s.get('n_claims') if s else None}",
              flush=True)
    d.to_parquet(path, index=False)
    print(f"L2 healed {fixed}/{len(bad)}; mean L2 now {d['L2'].mean():.3f}")


def heal_l3(stem: str) -> None:
    path = C.RUNS_DIR / f"{stem}_l3.parquet"
    d = pd.read_parquet(path)
    rep = pd.read_parquet(C.RUNS_DIR / f"{stem}.parquet").set_index(
        ["method", "instance_id"])
    sample, ctx = _ctx_sample()
    llm_call.set_max_tokens(MAX_TOKENS)
    raw_empty = d["raw"].fillna("").str.strip() == ""
    bad = d.index[(d["faithfulness"] == 0) & raw_empty].tolist()
    print(f"L3 {stem}: {len(bad)} empty rows of {len(d)}")
    fixed = 0
    for i in bad:
        r = d.loc[i]
        row = rep.loc[(r["method"], r["instance_id"])]
        inst = sample[sample["sample_id"] == r["instance_id"]].iloc[0].to_dict()
        facts = layer3_eval._facts_block(inst, ctx)
        s = None
        for a in range(ATTEMPTS):
            s = layer3_eval.score_one(row["raw_output"], facts,
                                      bg.instance_context(inst), "deepseek", 42)
            if s.get("ok") and str(s.get("raw", "")).strip():
                break
            time.sleep(1.5)
        if s and s.get("ok") and str(s.get("raw", "")).strip():
            for k in ["faithfulness", "completeness", "coherence", "actionability"]:
                d.loc[i, k] = s[k]
            d.loc[i, "raw"] = s.get("raw", "")
            d.loc[i, "L3"] = layer3_eval.l3_from(s)
            fixed += 1
        print(f"  {r['instance_id']}: faith={s.get('faithfulness') if s else None}",
              flush=True)
    d.to_parquet(path, index=False)
    print(f"L3 healed {fixed}/{len(bad)}; mean L3 now {d['L3'].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    args = ap.parse_args()
    if not llm_config.get_provider("deepseek").is_configured:
        print("deepseek not configured"); return
    heal_l2(args.stem)
    heal_l3(args.stem)


if __name__ == "__main__":
    main()
