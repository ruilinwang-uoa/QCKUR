"""Targeted diagnostics for the two hard groups (read-only analysis).

(a) OR 0.014" @6 (197-200): is BPFO evidence present in the RAW envelope
    (no bandpass) at higher harmonics, and does the kurtosis-selected band
    actually discard the modulation?
(b) Ball faults (118-121, 185-188, 222-225): sideband-aware ball score --
    textbook ball signature is harmonics of BSF / 2xBSF flanked by +-FTF
    sidebands (cage modulation). Score = prominence at k*f +- j*FTF.
(c) Normals under the same scores (gate reference).
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
FS = 12000
RPM_TABLE = SPEC["cwru_12k_drive_end"]["shaft_speed_rpm_by_load"]
M = SPEC["defect_freq_multiples_x_fr"]


def load_env(num: str) -> tuple[np.ndarray, float]:
    m = scipy.io.loadmat(DATA / f"{num}.mat")
    x = m.get(f"X{num}_DE_time")
    if x is None:
        x = m.get(f"X{int(num):03d}_DE_time")
    x = x.ravel().astype(float)
    rpm_key = f"X{num}RPM"
    rpm = float(m[rpm_key].ravel()[0]) if rpm_key in m else None
    env = np.abs(hilbert(x - x.mean()))
    env = env - env.mean()
    n = len(env)
    win = np.hanning(n)
    amp = np.abs(np.fft.rfft(env * win)) * 2 / np.sum(win)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    return freqs, amp, rpm


def peak(freqs, amp, f0, tol=0.5, win=8.0):
    band = (freqs >= f0 - win) & (freqs <= f0 + win)
    near = (freqs >= f0 - tol) & (freqs <= f0 + tol)
    floor = np.median(amp[band])
    return float(amp[near].max() / floor) if floor > 0 else 0.0


def harm(freqs, amp, f_c, kmax, **kw):
    return sum(peak(freqs, amp, k * f_c, **kw) for k in range(1, kmax + 1))


def sideband_score(freqs, amp, f_c, ftf, kmax=8, jmax=1):
    """Sum prominence over k*f_c + j*FTF for |j|<=jmax."""
    s = 0.0
    for k in range(1, kmax + 1):
        for j in range(-jmax, jmax + 1):
            f0 = k * f_c + j * ftf
            if 2 < f0 < freqs[-1] - 10:
                s += peak(freqs, amp, f0)
    return s


def main() -> None:
    groups = {
        "OR014": (["197", "198", "199", "200"], "OR"),
        "B": (["118", "119", "120", "121", "185", "186", "187", "188",
               "222", "223", "224", "225"], "B"),
        "normal": (["97", "98", "99", "100"], "normal"),
    }
    for gname, (nums, _) in groups.items():
        print(f"== {gname} ==")
        for num in nums:
            m = scipy.io.loadmat(DATA / f"{num}.mat")  # noqa: F841 (label only)
            freqs, amp, rpm = load_env(num)
            manifest = json.load(open(DATA / "cwru_manifest.json", encoding="utf-8"))
            meta = manifest[f"{num}.mat"]
            if rpm is None:
                rpm = RPM_TABLE[str(meta["load_hp"])]
            fr = rpm / 60.0
            ftf = M["FTF"] * fr
            bpfo_raw = harm(freqs, amp, M["BPFO"] * fr, 20)
            bpfo_lo = harm(freqs, amp, M["BPFO"] * fr, 5)
            sb_bsf = sideband_score(freqs, amp, M["BSF"] * fr, ftf)
            sb_2bsf = sideband_score(freqs, amp, M["ball_defect_2xBSF"] * fr, ftf)
            plain_bsf = harm(freqs, amp, M["BSF"] * fr, 8)
            plain_2bsf = harm(freqs, amp, M["ball_defect_2xBSF"] * fr, 8)
            print(f"  {num} L{meta['load_hp']} d={meta['defect_size_in']}: "
                  f"BPFO(k<=5)={bpfo_lo:6.1f} BPFO(k<=20)={bpfo_raw:7.1f} | "
                  f"BSFplain={plain_bsf:6.1f} BSFsb={sb_bsf:7.1f} | "
                  f"2BSFplain={plain_2bsf:6.1f} 2BSFsb={sb_2bsf:7.1f}")


if __name__ == "__main__":
    main()
