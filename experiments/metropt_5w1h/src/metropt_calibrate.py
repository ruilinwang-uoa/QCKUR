"""
metropt_calibrate.py -- per-vehicle healthy baselines + rule-threshold
false-alarm calibration for the MetroPT case.

Samples N random healthy windows (same guards as the window builder:
ends drawn from the healthy pool = >= 72 h from every event region
[start-72h, max(end, maintenance)+post-excl], >= 24 h apart, coverage
>= 0.8), extracts the frozen digest features, and writes
runs/metropt3/baseline_{tag}.json with robust stats per feature.

The rule-tier thresholds in the knowledge spec are then chosen so that
NO healthy window fires them (zero false alarms on the calibration
sample), with a stated safety margin. Detection-side justification of
the rules comes from the documented failure signatures (W2 label
forensics), not from the healthy sample.

Usage:
  python metropt_calibrate.py --csv CSV --manifest DATA_MANIFEST_JSON \
      --tag metropt2022 [--n 150]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metropt_features import ALL_COLS, window_features

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "runs" / "metropt3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--manifest", required=True,
                    help="data manifest with failure_events")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--post-exclude-h", type=float, default=24.0)
    args = ap.parse_args()

    events = json.loads(
        Path(args.manifest).read_text(encoding="utf-8"))["failure_events"]
    regions = [(pd.Timestamp(e["start"]) - pd.Timedelta(hours=72),
                pd.Timestamp(e.get("maintenance") or e["end"])
                + pd.Timedelta(hours=args.post_exclude_h))
               for e in events]

    print(f"loading {args.csv} ...")
    header = pd.read_csv(args.csv, nrows=0).columns.tolist()
    use = [c for c in ALL_COLS if c in header]
    df = pd.read_csv(args.csv, usecols=use)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed",
                                     dayfirst=True)
    df = df.set_index("timestamp").sort_index()
    ts = df.index.to_series().reset_index(drop=True)

    daily = ts.dt.date.value_counts()
    med_daily = float(daily.median())
    expected = int(24.0 * med_daily / 24.0)

    healthy = pd.Series(True, index=ts.index)
    for lo, hi in regions:
        healthy &= ~((ts >= lo) & (ts <= hi))
    pool = ts[healthy]
    print(f"healthy pool: {len(pool)} / {len(ts)} samples "
          f"({100.0 * len(pool) / len(ts):.1f}%)")

    rng = np.random.default_rng(args.seed)
    feats, chosen = [], []
    tries = 0
    while len(feats) < args.n and tries < args.n * 40:
        tries += 1
        t1 = pd.Timestamp(pool.iloc[int(rng.integers(len(pool)))])
        t0 = t1 - pd.Timedelta(hours=24)
        if t0 < ts.min() or t1 > ts.max():
            continue
        if any(abs((pd.Timestamp(c) - t1).total_seconds()) < 86400.0
               for c in chosen):
            continue  # keep sampled windows >= 24h apart
        w = df.loc[t0:t1]
        f = window_features(w, expected)
        if f.get("coverage", 0) is None or f.get("coverage", 0) < 0.8:
            continue
        if any(lo <= t1 <= hi or lo <= t0 <= hi for lo, hi in regions):
            continue
        chosen.append(str(t1))
        f["t_end"] = str(t1)
        feats.append(f)

    F = pd.DataFrame(feats)
    # per-window baseline features -- the spec builder runs the composite
    # rule false-alarm check on these (joint check, not marginals)
    F.to_parquet(OUTDIR / f"baseline_windows_{args.tag}.parquet")
    num = F.select_dtypes(include=[np.number])
    stats = {
        "tag": args.tag,
        "n_windows": len(F),
        "seed": args.seed,
        "coverage_note": ("healthy pool = >=72h from every event region "
                          "[start-72h, max(end,maintenance)+"
                          f"{args.post_exclude_h}h]; windows >=24h apart, "
                          "coverage >= 0.8"),
        "features": {},
    }
    for col in num.columns:
        v = num[col].dropna()
        if len(v) == 0:
            continue
        stats["features"][col] = {
            "n": int(len(v)), "median": round(float(v.median()), 4),
            "p01": round(float(v.quantile(0.01)), 4),
            "p99": round(float(v.quantile(0.99)), 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
        }
    dest = OUTDIR / f"baseline_{args.tag}.json"
    dest.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"baseline ({len(F)} windows) -> {dest}")
    # quick false-alarm-relevant printout
    for k in ["mc_on_frac", "mc_on_maxrun_min", "tp2_frac_ge9_5",
              "lowpress_highload_min", "lps_frac", "lps_maxrun_min",
              "tp2_p95", "oil_max", "oil_trend_c", "mc_maxrun5_min",
              "oil_level_frac"]:
        if k in stats["features"]:
            s = stats["features"][k]
            print(f"  {k}: median {s['median']}  p99 {s['p99']}  "
                  f"max {s['max']}")


if __name__ == "__main__":
    main()
