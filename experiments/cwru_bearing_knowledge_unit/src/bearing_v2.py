"""CWRU rule engine v2: band-scanned envelope (per-component max over bands).

v1 picked ONE max-kurtosis band; diagnostics (bearing_diag.py) showed the
picker often lands on a flat band for ball faults and OR 0.014, while the
raw envelope is unusable for gating (normal files out-score real faults).
v2 therefore evaluates the envelope spectrum of EVERY candidate band and
takes, per fault component, the maximum prominence across bands -- the band
choice is optimized for the decision metric itself.

Components: BPFO / BPFI / BALLFAM = max(BSF, 2xBSF), harmonics k <= 8.
Normal gate: r_max < tau, tau calibrated on loads 0/1 (geometric midpoint).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, filtfilt, hilbert

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bearing_cwru"
SPEC = json.load(open(ROOT / "knowledge" / "maps" / "physics_rules_spec_bearing.json",
                      encoding="utf-8"))
FS = 12000
RPM_TABLE = SPEC["cwru_12k_drive_end"]["shaft_speed_rpm_by_load"]
M = SPEC["defect_freq_multiples_x_fr"]

BAND_LO_HZ = 400
BAND_WIDTH_HZ = 400
BAND_STEP_HZ = 200
BAND_HI_HZ = 5800
KMAX = 8
TOL_HZ = 0.5
WIN_HZ = 8.0


def band_edges() -> list[tuple[float, float]]:
    nyq = FS / 2
    edges, lo = [], BAND_LO_HZ
    while lo + BAND_WIDTH_HZ <= min(BAND_HI_HZ, nyq - 10):
        edges.append((lo, lo + BAND_WIDTH_HZ))
        lo += BAND_STEP_HZ
    return edges


def prominence(freqs: np.ndarray, amp: np.ndarray, f_c: float) -> float:
    total = 0.0
    for k in range(1, KMAX + 1):
        f0 = k * f_c
        if f0 >= freqs[-1] - WIN_HZ:
            break
        band = (freqs >= f0 - WIN_HZ) & (freqs <= f0 + WIN_HZ)
        near = (freqs >= f0 - TOL_HZ) & (freqs <= f0 + TOL_HZ)
        floor = np.median(amp[band])
        total += float(amp[near].max() / floor) if floor > 0 else 0.0
    return total


def analyse(num: str, meta: dict) -> dict:
    m = scipy.io.loadmat(DATA / f"{num}.mat")
    x = m.get(f"X{num}_DE_time")
    if x is None:
        x = m.get(f"X{int(num):03d}_DE_time")
    x = x.ravel().astype(float)
    rpm_key = f"X{num}RPM"
    rpm = float(m[rpm_key].ravel()[0]) if rpm_key in m else None
    if rpm is None:
        rpm = RPM_TABLE[str(meta["load_hp"])]
    fr = rpm / 60.0

    best = {"BPFO": 0.0, "BPFI": 0.0, "BSF": 0.0, "BALL": 0.0}
    best_band = {}
    nyq = FS / 2
    n = len(x)
    win = np.hanning(n)
    for lo, hi in band_edges():
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        xb = filtfilt(b, a, x)
        env = np.abs(hilbert(xb))
        env = env - env.mean()
        amp = np.abs(np.fft.rfft(env * win)) * 2 / np.sum(win)
        freqs = np.fft.rfftfreq(n, d=1.0 / FS)
        for name, mult in [("BPFO", M["BPFO"]), ("BPFI", M["BPFI"]),
                           ("BSF", M["BSF"]), ("BALL", M["ball_defect_2xBSF"])]:
            r = prominence(freqs, amp, mult * fr)
            if r > best[name]:
                best[name] = r
                best_band[name] = lo
    best["BALLFAM"] = max(best["BSF"], best["BALL"])
    return {"rpm": rpm, **best, "band_BPFO": best_band.get("BPFO"),
            "band_BPFI": best_band.get("BPFI"), "band_BALL": best_band.get("BALL")}


def main() -> None:
    manifest = json.load(open(DATA / "cwru_manifest.json", encoding="utf-8"))
    rows = []
    for name, meta in manifest.items():
        num = name.split(".")[0]
        res = analyse(num, meta)
        truth = ("normal" if meta["condition"] == "normal"
                 else meta["component"].split("@")[0])
        rows.append({"file": name, "truth": truth, "load_hp": meta["load_hp"],
                     "defect_in": meta["defect_size_in"], **res})
    df = pd.DataFrame(rows)
    df["r_max"] = df[["BPFO", "BPFI", "BALLFAM"]].max(axis=1)

    cal = df[df.load_hp <= 1]
    normal_max = cal[cal.truth == "normal"].r_max.max()
    fault_min = cal[cal.truth != "normal"].r_max.min()
    tau = float(np.sqrt(normal_max * fault_min))
    print(f"calibration (load 0/1): max normal r={normal_max:.1f}, "
          f"min fault r={fault_min:.1f}, tau={tau:.1f}"
          + ("" if normal_max < fault_min else "  [WARNING: overlap]"))

    comp = {"BPFO": "OR", "BPFI": "IR", "BALLFAM": "B"}
    df["pred"] = np.where(df.r_max < tau, "normal",
                          df[["BPFO", "BPFI", "BALLFAM"]].idxmax(axis=1).map(comp))
    df["ok"] = df.pred == df.truth

    for _, r in df.sort_values(["truth", "defect_in", "load_hp"]).iterrows():
        print(f"  {r['file']:>8} {r.truth:>6} d={r.defect_in:<6} L{r.load_hp} "
              f"{r.BPFO:6.1f} @{r.band_BPFO:.0f} | {r.BPFI:6.1f} @{r.band_BPFI:.0f} "
              f"| {r.BALLFAM:6.1f} @{r.band_BALL:.0f} -> {r.pred:>6} "
              f"{'Y' if r.ok else 'N'}")

    print("\nconfusion matrix (rows=true, cols=pred):")
    print(pd.crosstab(df.truth, df.pred).to_string())
    print(f"\naccuracy: {df.ok.mean():.3f} ({df.ok.sum()}/{len(df)})")
    held = df[df.load_hp >= 2]
    print(f"held-out half (loads 2/3): {held.ok.sum()}/{len(held)}")

    by_group = df[df.truth != "normal"].groupby(
        ["truth", "defect_in"]).ok.agg(["sum", "count"])
    print("\nby fault group:")
    print(by_group.to_string())

    df.to_csv(DATA / "bearing_v2_results.csv", index=False)
    print("results -> data/bearing_cwru/bearing_v2_results.csv")


if __name__ == "__main__":
    main()
