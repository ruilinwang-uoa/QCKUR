"""
metropt_smoke.py -- run the two-tier handler over all campaign windows
and report the verdict distribution + per-positive detail.

Usage: python metropt_smoke.py [--verbose]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from metropt_handlers import handle, load_spec

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
SOURCES = [("metropt3", "features_metropt3.parquet"),
           ("metropt2022", "features_metropt2022.parquet"),
           ("metropt2022B", "features_metropt2022B.parquet")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    spec = load_spec()
    all_rows = []
    for tag, par in SOURCES:
        feats = pd.read_parquet(f"{RUNS}/{par}")
        for wid, row in feats.iterrows():
            v = handle(row.to_dict(), tag, spec)
            all_rows.append({
                "window_id": wid, "vehicle": tag,
                "kind": row["kind"],
                "event_id": row.get("event_id"),
                "offset_h": row.get("offset_h"),
                "label": v["label"], "tier": v["tier"],
                "conf": v["confidence"],
                "oob": ",".join(v["oob_flags"]) or None,
                "oil_elev_c": v["evidence"].get("oil_elev_c"),
            })
    df = pd.DataFrame(all_rows)
    print("=== verdict x kind ===")
    print(pd.crosstab(df.label, df.kind).to_string())
    print()
    print("=== positives detail ===")
    pos = df[df.kind == "positive"].sort_values(
        ["vehicle", "event_id", "offset_h"])
    print(pos.to_string(index=False))
    print()
    ctl_sig = df[(df.kind == "control") & (df.label != "healthy")]
    print(f"=== controls with non-healthy verdicts: {len(ctl_sig)} ===")
    if len(ctl_sig):
        print(ctl_sig.to_string(index=False))
    if args.verbose:
        print("=== all windows ===")
        print(df.to_string(index=False))
    df.to_csv(f"{RUNS}/handler_smoke_verdicts.csv", index=False)
    print(f"\nverdicts -> {RUNS}/handler_smoke_verdicts.csv")


if __name__ == "__main__":
    main()
