"""Progress checker for the Kimi judging run (read-only, safe to run anytime).

Prints, for each layer: scored/800, ok-rate, per-system counts, and the
latest judge call from the log. Usage:
    python check_kimi_progress.py
"""
from pathlib import Path

import pandas as pd

import config as C

ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]


def check(layer: str) -> None:
    p = C.RUNS_DIR / f"reports_main_glm_n100_kimi_{layer}.parquet"
    log = C.LOGS_DIR / f"kimi_{layer}.log"
    print(f"=== Layer {layer.upper()} ===")
    if not p.exists():
        print("  not started yet")
    else:
        d = pd.read_parquet(p)
        n_ok = int(d["ok"].sum())
        print(f"  scored: {len(d)}/800   ok: {n_ok}/{len(d)}"
              + (f"   FAILED: {len(d) - n_ok}  <-- investigate logs" if n_ok < len(d) else ""))
        counts = d.groupby("method").size()
        print("  " + "  ".join(f"{m}:{counts.get(m, 0)}" for m in ORDER))
        score_col = "L3" if layer == "l3" else "L2"
        recent = d.tail(3)[["method", "instance_id", "ok", score_col]]
        print("  last scored: " + "; ".join(
            f"{r.method}/{r.instance_id}={getattr(r, score_col)}" for r in recent.itertuples()))
    if log.exists():
        lines = [ln for ln in log.read_text(encoding="utf-8", errors="ignore").splitlines()
                 if ln.strip().startswith("[") or "warn" in ln.lower()]
        if lines:
            print(f"  log tail: {lines[-1][:100]}")
    print()


for lay in ("l2", "l3"):
    check(lay)
done = all((C.RUNS_DIR / f"reports_main_glm_n100_kimi_{l}.parquet").exists()
           and len(pd.read_parquet(C.RUNS_DIR / f"reports_main_glm_n100_kimi_{l}.parquet")) >= 800
           for l in ("l2", "l3"))
print("analysis ready to run" if done
      else "(when both layers reach 800/800, the bat runs analyze_kimi_judge.py automatically)")
