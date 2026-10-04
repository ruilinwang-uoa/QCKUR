"""Merge the per-system multi-seed generation slices into one parquet per seed.

Inputs : runs/reports_main_glm_n100_{B3,B4,B5,B6,B7}_seed{S}.parquet
Output : runs/reports_main_glm_n100_B3toB7_seed{S}.parquet  (+ .csv)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parents[1] / "runs"
SYSTEMS = ["B3", "B4", "B5", "B6", "B7"]

def main(seed: int) -> None:
    parts = []
    for m in SYSTEMS:
        p = RUNS / f"reports_main_glm_n100_{m}_seed{seed}.parquet"
        d = pd.read_parquet(p)
        d = d[(d["method"] == m) & (d["seed"] == seed)]
        parts.append(d)
        print(f"  {m} s{seed}: {len(d)} reports")
    out = pd.concat(parts, ignore_index=True)
    assert len(out) == 500, f"expected 500 reports, got {len(out)}"
    path = RUNS / f"reports_main_glm_n100_B3toB7_seed{seed}.parquet"
    out.to_parquet(path, index=False)
    out.drop(columns=[c for c in ["raw_output", "parsed_answer", "claims", "ku_trace", "prompt"]
                      if c in out.columns]).to_csv(
        RUNS / f"reports_main_glm_n100_B3toB7_seed{seed}.csv", index=False)
    print(f"Saved {path} ({len(out)} reports)")

if __name__ == "__main__":
    main(int(sys.argv[1]))
