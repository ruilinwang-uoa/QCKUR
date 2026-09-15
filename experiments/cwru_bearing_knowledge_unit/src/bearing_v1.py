"""CWRU rule engine v1: bandpass envelope + normal gate + full-40 evaluation.

v0 (bearing_prototype.py) established that BPFO/BPFI/BSF-family prominence on
the raw Hilbert envelope separates the three fault classes 7/7, but the
normal file's ball-family prominence overlapped the ball-fault range, so
normality could not be gated by relative comparison alone.

v1 adds:
  1. Spectral-kurtosis band selection: scan fixed 600-Hz bands over
     0.3-5.7 kHz, band-pass each (Butterworth-4, filtfilt), pick the band
     with maximum signal kurtosis, and take the envelope of that band only.
     This suppresses broadband normal content and sharpens impulsive
     fault modulation.
  2. A normal gate: r = max(BPFO, BPFI, BALLFAM) prominence; classify as
     normal iff r < tau. tau is calibrated on the load-0/1 half
     (geometric midpoint of max-normal and min-fault) and then applied to
     all 40 files, so the load-2/3 half is untouched by calibration.
  3. Full-40 confusion matrix.

Outputs: console table + code/data/bearing_cwru/bearing_v1_results.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bearing_cwru"
SPEC = json.load(open(ROOT / "knowledge" / "maps" / "physics_rules_spec_bearing.json",
                      encoding="utf-8"))
FS = SPEC["cwru_12k_drive_end"]["sampling_hz"]
RPM_TABLE = SPEC["cwru_12k_drive_end"]["shaft_speed_rpm_by_load"]
M = SPEC["defect_freq_multiples_x_fr"]

BAND_LO_HZ = 400
BAND_WIDTH_HZ = 400
BAND_STEP_HZ = 200
BAND_HI_HZ = 5800
KMAX = 10
TOL_HZ = 0.5
WIN_HZ = 8.0


def load_file(num: str) -> tuple[np.ndarray, float]:
    m = scipy.io.loadmat(DATA / f"{num}.mat")
    x = m.get(f"X{num}_DE_time")
    if x is None:
        x = m.get(f"X{int(num):03d}_DE_time")
    x = x.ravel().astype(float)
    rpm_key = f"X{num}RPM"
    rpm = float(m[rpm_key].ravel()[0]) if rpm_key in m else None
    return x, rpm


def band_edges() -> list[tuple[float, float]]:
    nyq = FS / 2
    edges = []
    lo = BAND_LO_HZ
    while lo + BAND_WIDTH_HZ <= min(BAND_HI_HZ, nyq - 10):
        edges.append((lo, lo + BAND_WIDTH_HZ))
        lo += BAND_STEP_HZ
    return edges


def select_band(x: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Return (lo, hi, kurtosis, bandpassed signal) of the max-kurtosis band."""
    nyq = FS / 2
    best = (None, None, -np.inf, None)
    for lo, hi in band_edges():
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        xb = filtfilt(b, a, x)
        k = float(kurtosis(xb, fisher=True))
        if k > best[2]:
            best = (lo, hi, k, xb)
    return best


def envelope_spectrum(xb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    env = np.abs(hilbert(xb))
    env = env - env.mean()
    n = len(env)
    win = np.hanning(n)
    amp = np.abs(np.fft.rfft(env * win)) * 2 / np.sum(win)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    return freqs, amp


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
    x, rpm = load_file(num)
    if rpm is None:
        rpm = RPM_TABLE[str(meta["load_hp"])]
    fr = rpm / 60.0
    lo, hi, kurt, xb = select_band(x)
    freqs, amp = envelope_spectrum(xb)
    r = {
        "BPFO": prominence(freqs, amp, M["BPFO"] * fr),
        "BPFI": prominence(freqs, amp, M["BPFI"] * fr),
        "BSF": prominence(freqs, amp, M["BSF"] * fr),
        "BALL": prominence(freqs, amp, M["ball_defect_2xBSF"] * fr),
    }
    r["BALLFAM"] = max(r["BSF"], r["BALL"])
    return {"rpm": rpm, "band_lo": lo, "band_kurt": kurt, **r}


def main() -> None:
    manifest = json.load(open(DATA / "cwru_manifest.json", encoding="utf-8"))
    rows = []
    for name, meta in manifest.items():
        num = name.split(".")[0]
        res = analyse(num, meta)
        truth = ("normal" if meta["condition"] == "normal"
                 else meta["component"].split("@")[0])
        rows.append({"file": name, "num": num,
                     "truth": truth, "load_hp": meta["load_hp"],
                     "defect_in": meta["defect_size_in"], **res})
    df = pd.DataFrame(rows)
    df["r_max"] = df[["BPFO", "BPFI", "BALLFAM"]].max(axis=1)

    # ---- normal-gate calibration on load 0/1 only ----
    cal = df[df.load_hp <= 1]
    normal_max = cal[cal.truth == "normal"].r_max.max()
    fault_min = cal[cal.truth != "normal"].r_max.min()
    tau = float(np.sqrt(normal_max * fault_min))
    print(f"calibration (load 0/1): max normal r={normal_max:.1f}, "
          f"min fault r={fault_min:.1f}, tau={tau:.1f}")
    if normal_max >= fault_min:
        print("WARNING: calibration subset does not separate normal/fault")

    comp = {"BPFO": "OR", "BPFI": "IR", "BALLFAM": "B"}
    df["pred"] = np.where(
        df.r_max < tau, "normal",
        df[["BPFO", "BPFI", "BALLFAM"]].idxmax(axis=1).map(comp))
    df["ok"] = df.pred == df.truth

    print("\nper-file:")
    for _, r in df.sort_values(["truth", "load_hp", "defect_in"]).iterrows():
        print(f"  {r['file']:>8} {r.truth:>6} d={r.defect_in:<6} L{r.load_hp} "
              f"band {r.band_lo:.0f}Hz k={r.band_kurt:5.1f} "
              f"{r.BPFO:6.1f} {r.BPFI:6.1f} {r.BALLFAM:6.1f} "
              f"-> {r.pred:>6} {'Y' if r.ok else 'N'}")

    cm = pd.crosstab(df.truth, df.pred)
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(cm.to_string())
    print(f"\naccuracy: {df.ok.mean():.3f} ({df.ok.sum()}/{len(df)})")

    # calibration-free half (loads 2/3): how the frozen tau fares
    held = df[df.load_hp >= 2]
    print(f"held-out half (loads 2/3): {held.ok.sum()}/{len(held)}")

    df.to_csv(DATA / "bearing_v1_results.csv", index=False)
    print(f"results -> data/bearing_cwru/bearing_v1_results.csv")


if __name__ == "__main__":
    main()
