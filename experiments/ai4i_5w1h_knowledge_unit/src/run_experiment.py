"""Unified experiment runner (Phase 3, main experiment).

Generates one maintenance report per (system, instance, seed) under the standard
log schema and writes runs/reports.parquet + runs/run_log.csv. Two modes:

  smoke  : a few representative instances x all 8 systems x 1 seed (default;
           validates the pipeline end-to-end with real LLM calls)
  full   : n_instances x systems x seeds (the 315-sample campaign)

Usage:
  python run_experiment.py smoke                       # ~8 instances, all systems
  python run_experiment.py full --n 315 --systems all --seeds 42
  python run_experiment.py full --n 315 --systems B3,B8 --seeds 42,1,7,2024,123
  python run_experiment.py smoke --provider deepseek   # cross-LLM check
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

import config as C
import background as bg
import llm_config
import llm_call
from corpus import get_corpus_and_retriever
from systems import SYSTEMS as _BASE_SYSTEMS, ABLATIONS
SYSTEMS = {**_BASE_SYSTEMS, **ABLATIONS}


def _pick_smoke_instances(sample: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """A small set covering every fault mode present in the 315-sample."""
    picks = []
    if (sample["Machine failure"] == 0).any():
        picks.append(sample[sample["Machine failure"] == 0].iloc[0])
    for mode in ["HDF", "PWF", "OSF", "TWF", "RNF"]:
        sub = sample[sample[mode] == 1]
        if len(sub):
            picks.append(sub.iloc[0])
    out = pd.DataFrame(picks[:n]).drop_duplicates(subset=["UDI"])
    return out.reset_index(drop=True)


def _stratified_subset(sample: pd.DataFrame, n: int) -> pd.DataFrame:
    """All faulted instances first, then healthy, to guarantee fault coverage.

    With ~3.5% failures, head(n) would rarely include a fault; this keeps every
    faulted sample (<=11 in the 315-set) and fills the rest with healthy ones.
    """
    faulted = sample[sample["Machine failure"] == 1]
    healthy = sample[sample["Machine failure"] == 0]
    n_f = min(len(faulted), n)
    n_h = max(0, n - n_f)
    out = pd.concat([faulted.head(n_f), healthy.head(n_h)]).sort_values("sample_id")
    return out.reset_index(drop=True)


def _to_log_row(result, seed, instance_row, provider, exp_id):
    lc = result.get("llm_call") or {}
    text = result.get("text", "")
    is_local = lc == {}
    ifr = (not is_local) and llm_call.is_ifr(lc) if lc else False
    if is_local:
        ifr = not text.strip()
    modes = [m for m in ["TWF", "HDF", "PWF", "OSF", "RNF"] if instance_row[m] == 1]
    return {
        "run_id": f"{result['system_id']}_{instance_row['sample_id']}_s{seed}",
        "exp_id": exp_id,
        "method": result["system_id"],
        "llm_version": (lc.get("model") or "local") if not is_local else "local",
        "provider": provider,
        "seed": seed,
        "instance_id": instance_row["sample_id"],
        "udi": int(instance_row["UDI"]),
        "machine_failure": int(instance_row["Machine failure"]),
        "active_modes": "+".join(modes) if modes else "none",
        "latency_s": lc.get("latency_s", 0.0),
        "prompt_tokens": lc.get("prompt_tokens", 0),
        "completion_tokens": lc.get("completion_tokens", 0),
        "api_cost": lc.get("cost_usd", 0.0),
        "raw_output": text,
        "parsed_answer": text,
        "ifr_flag": bool(ifr),
        "claims": json.dumps(result.get("claims", []), ensure_ascii=False),
        "ku_trace": json.dumps(result.get("ku_trace", []), ensure_ascii=False),
        "llm_error": lc.get("error"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def run(instances: pd.DataFrame, systems: list[str], seeds: list[int],
        provider: str, retriever, ctx, exp_id: str, delay: float = 1.0,
        out_path=None, resume: bool = True, ckpt_every: int = 5) -> pd.DataFrame:
    """Generate reports with incremental checkpointing + resume.

    If ``out_path`` exists and ``resume``, already-finished (method, instance,
    seed) triples are skipped and their rows kept. The checkpoint parquet is
    rewritten every ``ckpt_every`` new records so an interrupt never loses more
    than a handful of calls.
    """
    rows: list[dict] = []
    done: set[tuple] = set()
    if out_path and out_path.exists() and resume:
        existing = pd.read_parquet(out_path)
        rows = existing.to_dict("records")
        done = {(r["method"], r["instance_id"], r["seed"]) for r in rows}
        print(f"Resuming: {len(done)} records already in {out_path.name}")

    t_start = time.time()
    n_new = 0
    for seed in seeds:
        for _, inst in instances.iterrows():
            row = inst.to_dict()
            for sid in systems:
                if (sid, inst["sample_id"], seed) in done:
                    continue
                fn = SYSTEMS[sid]
                try:
                    res = fn(row, ctx, provider, retriever, seed=seed)
                except Exception as e:  # noqa: BLE001
                    res = {"system_id": sid, "text": "", "claims": [], "ku_trace": [],
                           "prompt": "", "llm_call": {"ok": False, "text": "",
                           "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
                           "cost_usd": 0.0, "model": "", "error": f"{type(e).__name__}: {e}"}}
                rows.append(_to_log_row(res, seed, inst, provider, exp_id))
                n_new += 1
                if res.get("llm_call"):
                    print(f"  [{sid}] {inst['sample_id']} s{seed}: "
                          f"ok={res['llm_call'].get('ok')} "
                          f"tok={res['llm_call'].get('prompt_tokens',0)}+"
                          f"{res['llm_call'].get('completion_tokens',0)} "
                          f"t={res['llm_call'].get('latency_s',0)}s "
                          f"ifr={rows[-1]['ifr_flag']} (total {len(rows)})", flush=True)
                    if delay:
                        time.sleep(delay)
                else:
                    print(f"  [{sid}] {inst['sample_id']} s{seed}: (local) "
                          f"ifr={rows[-1]['ifr_flag']} (total {len(rows)})", flush=True)
                if out_path and n_new % ckpt_every == 0:
                    pd.DataFrame(rows).to_parquet(out_path, index=False)

    df = pd.DataFrame(rows)
    df["wall_time_s"] = round(time.time() - t_start, 1)
    if out_path:
        df.to_parquet(out_path, index=False)
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n=== Summary ===")
    g = df.groupby("method")
    summ = pd.DataFrame({
        "n": g.size(),
        "ifr_rate": g["ifr_flag"].mean().round(3),
        "avg_latency_s": g["latency_s"].mean().round(2),
        "avg_prompt_tok": g["prompt_tokens"].mean().round(0),
        "avg_compl_tok": g["completion_tokens"].mean().round(0),
        "total_cost_usd": g["api_cost"].sum().round(4),
    })
    print(summ.to_string())
    print(f"\nTotal API cost (USD): {df['api_cost'].sum():.4f} | "
          f"total tokens: {int(df['prompt_tokens'].sum())}+{int(df['completion_tokens'].sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "full"])
    ap.add_argument("--n", type=int, default=8, help="instances (smoke default 8)")
    ap.add_argument("--systems", default="all", help="comma list or 'all'")
    ap.add_argument("--seeds", default="42", help="comma list of seeds")
    ap.add_argument("--provider", default="openai", choices=["openai", "deepseek"])
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds to sleep after each LLM call (rate-limit pacing)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="override max_tokens (raise >=2000 for reasoning models)")
    ap.add_argument("--no-resume", action="store_true",
                    help="start fresh even if the output checkpoint exists")
    ap.add_argument("--out", default=None, help="output parquet path under runs/")
    args = ap.parse_args()

    if not llm_config.get_provider(args.provider).is_configured:
        print(f"ERROR: provider '{args.provider}' not configured in .env")
        return
    if args.max_tokens:
        llm_call.set_max_tokens(args.max_tokens)
        print(f"max_tokens overridden to {args.max_tokens}")
    systems = list(SYSTEMS) if args.systems == "all" else args.systems.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    ctx = bg.build_ctx()
    _, retriever = get_corpus_and_retriever()

    if args.mode == "smoke":
        instances = _pick_smoke_instances(sample, n=args.n)
        exp_id = "smoke"
    else:
        instances = _stratified_subset(sample, args.n)
        exp_id = f"ai4i_{args.provider}"

    print(f"Mode={args.mode} | provider={args.provider} ({llm_config.get_provider(args.provider).version_string}) "
          f"| systems={systems} | seeds={seeds} | instances={len(instances)} | delay={args.delay}s")

    out = args.out or f"reports_{args.mode}_{args.provider}.parquet"
    out_path = C.RUNS_DIR / out
    df = run(instances, systems, seeds, args.provider, retriever, ctx, exp_id,
             delay=args.delay, out_path=out_path, resume=not args.no_resume)
    summarize(df)

    df.to_parquet(out_path, index=False)
    df.drop(columns=["raw_output", "parsed_answer", "claims", "ku_trace", "prompt"]
            if "prompt" in df else ["raw_output", "parsed_answer", "claims", "ku_trace"]
            ).to_csv(C.RUNS_DIR / out.replace(".parquet", ".csv"), index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
