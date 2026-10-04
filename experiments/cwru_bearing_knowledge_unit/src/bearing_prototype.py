"""Envelope-spectrum rule prototype for the CWRU second instantiation (Phase 3).

Validates that characteristic-frequency rules are decidable on the real data:
for a probe set spanning all four classes, computes the Hilbert envelope
spectrum, evaluates harmonic prominence around BPFO / BPFI / 2xBSF / FTF
(multiples from physics_rules_spec_bearing.json, shaft speed from the
per-file RPM channel), and applies the v0 decision rule
(argmax harmonic prominence -> component).

This is the bearing-domain analogue of AI4I's rule engine feasibility check:
if the rules cannot separate the classes here, the instantiation design
(bandpass choice, thresholds) must change before any KU/question-bank work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bearing_cwru"
SPEC = json.load(open(ROOT / "knowledge" / "maps" / "physics_rules_spec_bearing.json",
                      encoding="utf-8"))
FS = SPEC["cwru_12k_drive_end"]["sampling_hz"]
RPM_TABLE = SPEC["cwru_12k_drive_end"]["shaft_speed_rpm_by_load"]
M = SPEC["defect_freq_multiples_x_fr"]

PROBE = ["97", "105", "118", "130", "169", "222", "234", "237"]


def load_file(num: str) -> tuple[np.ndarray, float]:
    m = scipy.io.loadmat(DATA / f"{num}.mat")
    x = m.get(f"X{num}_DE_time")
    if x is None:
        x = m.get(f"X{int(num):03d}_DE_time")
    x = x.ravel().astype(float)
    rpm_key = f"X{num}RPM"
    rpm = float(m[rpm_key].ravel()[0]) if rpm_key in m else None
    return x, rpm


def envelope_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean()
    env = np.abs(hilbert(x))
    env = env - env.mean()
    n = len(env)
    win = np.hanning(n)
    amp = np.abs(np.fft.rfft(env * win)) * 2 / np.sum(win)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    return freqs, amp


def harmonic_prominence(freqs: np.ndarray, amp: np.ndarray, f_c: float,
                        kmax: int = 6, tol_hz: float = 0.5,
                        win_hz: float = 8.0) -> tuple[float, list]:
    """Peak amplitude at k*f_c relative to the LOCAL median floor in a +-win_hz
    band around each harmonic (robust to the 1/f envelope-spectrum shape),
    summed over k = 1..kmax."""
    total, peaks = 0.0, []
    for k in range(1, kmax + 1):
        f0 = k * f_c
        if f0 >= freqs[-1] - win_hz:
            break
        band = (freqs >= f0 - win_hz) & (freqs <= f0 + win_hz)
        near = (freqs >= f0 - tol_hz) & (freqs <= f0 + tol_hz)
        floor = np.median(amp[band])
        a = amp[near].max()
        ratio = float(a / floor) if floor > 0 else 0.0
        peaks.append(round(ratio, 1))
        total += ratio
    return total, peaks


def main() -> None:
    manifest = json.load(open(DATA / "cwru_manifest.json", encoding="utf-8"))
    print(f"{'file':>7} {'true':>6} {'rpm':>7} "
          f"{'BPFO':>7} {'BPFI':>7} {'BALL':>7} {'FTF':>6}  pred  ok")
    for num in PROBE:
        x, rpm = load_file(num)
        meta = manifest[f"{num}.mat"]
        if rpm is None:
            rpm = RPM_TABLE[str(meta["load_hp"])]
        fr = rpm / 60.0
        freqs, amp = envelope_spectrum(x)
        r = {}
        for name, mult in [("BPFO", M["BPFO"]), ("BPFI", M["BPFI"]),
                           ("BSF", M["BSF"]),
                           ("BALL", M["ball_defect_2xBSF"]), ("FTF", M["FTF"])]:
            r[name], _ = harmonic_prominence(freqs, amp, mult * fr, kmax=10)
        r["BALLFAM"] = r["BSF"] + r["BALL"]  # ball evidence across BSF family
        comp = {"BPFO": "OR", "BPFI": "IR", "BALLFAM": "B"}
        pred_key = max(comp, key=lambda k: r[k])
        pred = comp[pred_key]
        truth = ("none" if meta["condition"] == "normal"
                 else meta["component"].split("@")[0])
        ok = "Y" if pred == truth else ("~" if truth == "none" else "N")
        print(f"{num:>7} {truth:>6} {rpm:7.1f} "
              f"{r['BPFO']:7.1f} {r['BPFI']:7.1f} {r['BALLFAM']:7.1f} "
              f"{r['FTF']:6.1f}  {pred:>4}  {ok}")


if __name__ == "__main__":
    main()
