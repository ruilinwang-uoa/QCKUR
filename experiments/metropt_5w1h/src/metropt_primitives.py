"""
metropt_primitives.py -- raw-primitive menu for the B5 tool agent (no
verdicts, no baselines, no thresholds -- only computed aggregates the
agent could in principle compute from raw telemetry).

Per window: 24-row hourly table (means/max for key signals), LPS episode
list, duty-run list, oil hourly max trajectory.

Usage: python metropt_primitives.py --csv CSV --windows MANIFEST --tag TAG
Output: runs/metropt3/primitives_{tag}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "runs" / "metropt3"

COLS = ["timestamp", "TP2", "H1", "DV_pressure", "Motor_current",
        "Oil_temperature", "LPS", "COMP", "Towers", "Oil_level",
        "Flowmeter"]


def episodes(mask: pd.Series, limit: int = 10) -> list[dict]:
    m = mask.to_numpy()
    if not m.any():
        return []
    gap = np.r_[False, np.diff(mask.index.asi8) > 600e9]
    prev = np.r_[False, m[:-1]]
    starts = np.flatnonzero(m & ~(np.where(gap, False, prev)))
    nxt = np.r_[m[1:], False]
    gap_next = np.r_[gap[1:], False]
    ends = np.flatnonzero(m & ~(nxt & ~gap_next))
    out = []
    for s, e in zip(starts, ends):
        dur = (mask.index[e] - mask.index[s]).total_seconds() / 60.0
        if dur >= 1:
            out.append({"start": str(mask.index[s]),
                        "end": str(mask.index[e]),
                        "minutes": round(dur, 1)})
    out.sort(key=lambda x: -x["minutes"])
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    man = json.loads(Path(args.windows).read_text(encoding="utf-8"))
    header = pd.read_csv(args.csv, nrows=0).columns.tolist()
    use = [c for c in COLS if c in header]
    print(f"loading {args.csv} ...")
    df = pd.read_csv(args.csv, usecols=use)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed",
                                     dayfirst=True)
    df = df.set_index("timestamp").sort_index()

    menu: dict[str, dict] = {}
    for w in man["windows"]:
        t0, t1 = pd.Timestamp(w["t_start"]), pd.Timestamp(w["t_end"])
        win = df.loc[t0:t1]
        hourly = (win.resample("1h")
                  .agg({"TP2": ["mean", "max"], "H1": "mean",
                        "DV_pressure": "mean", "Motor_current": "mean",
                        "Oil_temperature": ["mean", "max"],
                        "LPS": "mean", "COMP": "mean", "Towers": "mean"})
                  .round(2))
        hour_rows = []
        for ts, row in hourly.iterrows():
            hour_rows.append({
                "hour_ending": str(ts + pd.Timedelta(hours=1)),
                "tp2_mean": row[("TP2", "mean")],
                "tp2_max": row[("TP2", "max")],
                "h1_mean": row[("H1", "mean")],
                "motor_current_mean": row[("Motor_current", "mean")],
                "oil_mean": row[("Oil_temperature", "mean")],
                "oil_max": row[("Oil_temperature", "max")],
                "lps_fraction": row[("LPS", "mean")],
            })
        menu[w["window_id"]] = {
            "window": [w["t_start"], w["t_end"]],
            "hourly": hour_rows,
            "lps_episodes_min": episodes(win["LPS"] > 0.5),
            "duty_runs_over_4p5A_min": episodes(win["Motor_current"] > 4.5),
            "percentiles": {
                "tp2": {q: round(float(win["TP2"].quantile(float(q))), 2)
                        for q in ("0.05", "0.5", "0.95", "0.99")},
                "motor_current": {
                    q: round(float(win["Motor_current"]
                                   .quantile(float(q))), 2)
                    for q in ("0.05", "0.5", "0.95", "0.99")},
                "oil_temperature": {
                    q: round(float(win["Oil_temperature"]
                                   .quantile(float(q))), 1)
                    for q in ("0.05", "0.5", "0.95", "0.99")},
            },
        }
        print(f"  {w['window_id']}: {len(win)} rows, "
              f"{len(menu[w['window_id']]['lps_episodes_min'])} LPS episodes")

    dest = OUTDIR / f"primitives_{args.tag}.json"
    dest.write_text(json.dumps(menu, indent=1), encoding="utf-8")
    print(f"primitives -> {dest} ({len(menu)} windows)")


if __name__ == "__main__":
    main()
