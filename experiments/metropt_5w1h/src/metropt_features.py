"""
metropt_features.py -- frozen window->digest feature extractor for the
MetroPT industrial case (QCKUR paper, attempt 4).

The digest is THE shared window representation for all systems:
  * B1/B2/B8: digest + question bank + two-tier handler verdicts
  * B3 (direct prompting): digest only
  * B5 (tool agent): digest + raw-primitive tools (no verdicts)
It is also the data-side truth for L1 numeric checks.

Usage:
  python metropt_features.py --csv CSV --windows WINDOW_MANIFEST_JSON \
      --tag metropt2022 [--out OUT.parquet]

Output: one row per window_id with the features below (NaN where a signal
is absent in that epoch).

Feature spec (v1.0, frozen with the knowledge spec):
  coverage        n_rows / expected rows at the file's observed density
  gaps_n          count of intra-window gaps > 10 min
  gap_max_h       longest intra-window gap (hours)
  tp2_mean/p95/max
  tp2_frac_ge9_5  fraction of samples with TP2 >= 9.5 bar (near cutoff)
  lowpress_highload_min  longest run (minutes) of TP2 < 9.0 bar while
                  Motor_current > 4.0 A  (compressor running, pressure
                  not building -- acute air-leak signature)
  h1_mean, h1_min         reservoir-loop pressure stats
  dv_frac_load    fraction with DV_pressure > 0.5 (tower discharge duty;
                  operator rule: ~0 under load)
  mc_mean, mc_frac_off / _offload / _load / _start
                  motor-current state fractions (official bands:
                  <1.5 off, 1.5-5.5 off-loaded, 5.5-8.5 under load,
                  >8.5 starting)
  mc_maxrun5_min  longest run (minutes) with Motor_current > 5 A
  oil_max, oil_mean, oil_p95
  oil_trend_c     oil-temp max over the last 6 h minus max over the
                  first 6 h of the window (slow-degradation direction)
  lps_frac, lps_first_before_end_h
                  LPS active fraction; hours before window end of the
                  first LPS trigger (NaN if none)
  oil_level_frac  fraction with Oil_level == 1 (oil below expected)
  comp_frac, towers_frac, pswitch_frac, caudal_frac
  flow_mean, flow_max      (2022 epochs only)
  duty_frac       fraction with gpsSpeed > 0.5 m/s (2022 only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "runs" / "metropt3"

ALL_COLS = ["timestamp", "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
            "Oil_temperature", "Flowmeter", "Motor_current", "COMP",
            "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch",
            "Oil_level", "Caudal_impulses", "gpsSpeed"]


def longest_run_min(mask: pd.Series, index: pd.DatetimeIndex) -> float:
    """Longest consecutive True run in minutes, vectorized. A sampling gap
    > 10 min breaks a run (conservative: elapsed time across a telemetry
    hole is not credited). Single-sample runs count as 0 minutes."""
    m = mask.to_numpy()
    if not m.any():
        return 0.0
    gap = np.r_[False, np.diff(index.asi8) > 600e9]  # ns, >10 min
    prev = np.r_[False, m[:-1]]
    prev_eff = np.where(gap, False, prev)          # continuity into sample i
    starts = np.flatnonzero(m & ~prev_eff)
    nxt = np.r_[m[1:], False]
    gap_next = np.r_[gap[1:], False]               # gap between i and i+1
    ends = np.flatnonzero(m & ~(nxt & ~gap_next))
    durs = (index[ends] - index[starts]).total_seconds() / 60.0
    return float(durs.max())


def window_features(w: pd.DataFrame, expected_rows: int) -> dict:
    f: dict = {}
    n = len(w)
    f["n_rows"] = n
    f["coverage"] = round(n / expected_rows, 4) if expected_rows else np.nan
    if n == 0:
        return f
    idx = w.index
    dt = idx.to_series().diff().dt.total_seconds()
    gaps = dt[dt > 600]
    f["gaps_n"] = int(len(gaps))
    f["gap_max_h"] = round(float(gaps.max()) / 3600.0, 3) if len(gaps) else 0.0

    tp2 = w["TP2"]
    f["tp2_mean"] = round(float(tp2.mean()), 3)
    f["tp2_p95"] = round(float(tp2.quantile(0.95)), 3)
    f["tp2_max"] = round(float(tp2.max()), 3)
    f["tp2_frac_ge9_5"] = round(float((tp2 >= 9.5).mean()), 4)

    mc = w["Motor_current"]
    lowload = (tp2 < 9.0) & (mc > 4.0)
    f["lowpress_highload_min"] = round(
        longest_run_min(lowload, idx), 2)
    f["mc_mean"] = round(float(mc.mean()), 3)
    f["mc_frac_off"] = round(float((mc < 1.5).mean()), 4)
    f["mc_frac_offload"] = round(float(mc.between(1.5, 5.5).mean()), 4)
    f["mc_frac_load"] = round(float(mc.between(5.5, 8.5).mean()), 4)
    f["mc_frac_start"] = round(float((mc > 8.5).mean()), 4)
    f["mc_maxrun5_min"] = round(longest_run_min(mc > 5.0, idx), 2)
    # duty physics: "working" = either offload-run or under load
    # (official states 4 A / 7 A); a leak shows near-continuous working
    # duty while pressure never reaches the cutoff
    f["mc_on_frac"] = round(float((mc > 4.5).mean()), 4)
    f["mc_on_maxrun_min"] = round(longest_run_min(mc > 4.5, idx), 2)

    h1 = w["H1"]
    f["h1_mean"] = round(float(h1.mean()), 3)
    f["h1_min"] = round(float(h1.min()), 3)

    dv = w["DV_pressure"]
    f["dv_frac_load"] = round(float((dv > 0.5).mean()), 4)

    oil = w["Oil_temperature"]
    f["oil_max"] = round(float(oil.max()), 2)
    f["oil_mean"] = round(float(oil.mean()), 2)
    f["oil_p95"] = round(float(oil.quantile(0.95)), 2)
    if n > 20:
        six_h = pd.Timedelta(hours=6)
        first = oil[oil.index <= idx[0] + six_h]
        last = oil[oil.index >= idx[-1] - six_h]
        f["oil_trend_c"] = round(float(last.max() - first.max()), 2)
    else:
        f["oil_trend_c"] = np.nan

    lps = w["LPS"]
    f["lps_frac"] = round(float((lps > 0.5).mean()), 4)
    f["lps_maxrun_min"] = round(longest_run_min(lps > 0.5, idx), 2)
    on = w.index[lps.to_numpy() > 0.5]
    if len(on):
        f["lps_first_before_end_h"] = round(
            float((idx[-1] - on[0]).total_seconds() / 3600.0), 3)
    else:
        f["lps_first_before_end_h"] = np.nan

    for key, col in [("oil_level_frac", "Oil_level"), ("comp_frac", "COMP"),
                     ("towers_frac", "Towers"),
                     ("pswitch_frac", "Pressure_switch"),
                     ("caudal_frac", "Caudal_impulses")]:
        s = w[col]
        f[key] = round(float((s > 0.5).mean()), 4)

    if "Flowmeter" in w.columns and w["Flowmeter"].notna().any():
        fl = w["Flowmeter"]
        f["flow_mean"] = round(float(fl.mean()), 2)
        f["flow_max"] = round(float(fl.max()), 2)
    else:
        f["flow_mean"] = f["flow_max"] = np.nan

    if "gpsSpeed" in w.columns and w["gpsSpeed"].notna().any():
        f["duty_frac"] = round(float((w["gpsSpeed"] > 0.5).mean()), 4)
    else:
        f["duty_frac"] = np.nan
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--windows", required=True,
                    help="window manifest json (window_manifest[_tag].json)")
    ap.add_argument("--tag", default="metropt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(args.windows).read_text(encoding="utf-8"))
    expected = int(man["params"]["expected_rows_per_window"])
    windows = man["windows"]

    print(f"loading {args.csv} ...")
    header = pd.read_csv(args.csv, nrows=0).columns.tolist()
    use = [c for c in ALL_COLS if c in header]
    df = pd.read_csv(args.csv, usecols=use)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed",
                                     dayfirst=True)
    df = df.set_index("timestamp").sort_index()

    rows = []
    for wspec in windows:
        t0 = pd.Timestamp(wspec["t_start"])
        t1 = pd.Timestamp(wspec["t_end"])
        w = df.loc[t0:t1]
        f = window_features(w, expected)
        f["window_id"] = wspec["window_id"]
        f["kind"] = wspec["kind"]
        f["event_id"] = wspec.get("event_id")
        f["offset_h"] = wspec.get("offset_h")
        f["t_start"], f["t_end"] = wspec["t_start"], wspec["t_end"]
        rows.append(f)
        print(f"  {wspec['window_id']}: {len(w)} rows")

    out = pd.DataFrame(rows).set_index("window_id")
    dest = Path(args.out) if args.out else (
        OUTDIR / f"features_{args.tag}.parquet")
    out.to_parquet(dest)
    print(f"features -> {dest}  ({len(out)} windows)")


if __name__ == "__main__":
    main()
