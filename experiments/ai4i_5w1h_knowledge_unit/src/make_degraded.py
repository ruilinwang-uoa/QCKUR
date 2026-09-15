"""Synthetic-anchor degradation set for protocol-validity testing (no humans).

Programmatically degrades "gold" reports (B1 template-only and B8 framework,
rule-engine v3 run) at four graded severity levels; if the three-layer
protocol measures the constructs it claims, scores must fall monotonically
with severity in the expected layers:

  D0  original report                                (baseline)
  D1  delete two 5W1H sections (When, Where)         -> coverage/completeness
  D2  corrupt numbers >=10 by x1.3                   -> numeric tolerance, faithfulness
  D3  corrupt the fault mode / inject a false fault  -> classification, L2 consistency

Sample: 30 instances (11 faulted + 19 healthy) x 2 systems x 4 levels = 240
reports, written as 8 pseudo-systems (B1_D0..B8_D3) so the standard
layer1/2/3 scoring scripts run unchanged.
"""
from __future__ import annotations

import re

import pandas as pd

import config as C

SRC = "reports_main_glm_n100_B1B2B8_v3"
N_FAULT, N_HEALTH = 11, 19

_MODE_MAP = {"TWF": "PWF", "PWF": "HDF", "HDF": "OSF", "OSF": "TWF", "RNF": "TWF"}
_NAME_MAP = [
    ("tool wear failure", "power failure"),
    ("power failure", "heat dissipation failure"),
    ("heat dissipation failure", "overstrain failure"),
    ("overstrain failure", "tool wear failure"),
    ("random failure", "tool wear failure"),
]
_NEGATIONS = [
    ("no fault mode detected", "active fault mode: PWF (power failure)"),
    ("no failure", "PWF (power failure) condition present"),
    ("no fault", "PWF fault present"),
    ("operating normally", "operating under a power failure (PWF)"),
    ("without a fault", "with an active PWF fault"),
    ("no anomaly detected", "anomaly detected: PWF"),
    ("healthy", "faulted (PWF)"),
]


def degrade_d1(text: str) -> str:
    """Delete the When and Where 5W1H sections (bracket or numbered style)."""
    # style A: [What] ... [Why] ... (B1)
    parts = re.split(r"(?=\[(?:What|Why|When|Where|Who|How)\])", text)
    if len(parts) >= 4:
        keep = [p for p in parts
                if not re.match(r"\[(?:When|Where)\]", p.strip())]
        if len(keep) < len(parts):
            return "".join(keep)
    # style B: numbered bold sections (B8): **1. xxx:** ...
    secs = re.split(r"(?=\.?\s*\*\*\d\.\s)", text)
    if len(secs) >= 4:
        keep = secs[:1] + [s for s in secs[1:]
                           if not re.match(r"\.?\s*\*\*[45]\.", s)]
        return "".join(keep)
    # fallback: drop the last two paragraphs
    paras = [p for p in text.split("\n\n") if p.strip()]
    return "\n\n".join(paras[:-2]) if len(paras) > 3 else text


def degrade_d2(text: str) -> str:
    def rep(m: re.Match) -> str:
        v = float(m.group())
        if 10.0 <= v <= 15000.0:
            nv = v * 1.3
            return str(int(round(nv))) if "." not in m.group() else f"{nv:.1f}"
        return m.group()
    return re.sub(r"\d+\.?\d*", rep, text)


def degrade_d3(text: str) -> str:
    out = text
    for mode, wrong in _MODE_MAP.items():
        out = re.sub(rf"\b{mode}\b", wrong, out)
    for name, wrong in _NAME_MAP:
        out = re.sub(re.escape(name), wrong, out, flags=re.I)
    for neg, claim in _NEGATIONS:
        out = re.sub(re.escape(neg), claim, out, flags=re.I)
    if out == text:  # nothing matched (e.g., terse healthy report)
        out = ("**1. Fault/Status:** Active fault mode: PWF (power failure). "
               + out)
    return out


DEGRADERS = {"D0": lambda t: t, "D1": degrade_d1, "D2": degrade_d2, "D3": degrade_d3}


def main() -> None:
    rep = pd.read_parquet(C.RUNS_DIR / f"{SRC}.parquet")
    sample = pd.read_parquet(C.DATA_DIR / "sample_315.parquet")
    ids = (list(sample[sample["Machine failure"] == 1]["sample_id"][:N_FAULT])
           + list(sample[sample["Machine failure"] == 0]["sample_id"][:N_HEALTH]))
    rows, fb = [], 0
    for sysname in ["B1", "B8"]:
        sub = rep[rep["method"] == sysname].set_index("instance_id")
        for iid in ids:
            r = sub.loc[iid]
            for lvl, fn in DEGRADERS.items():
                deg = fn(r["raw_output"])
                if lvl == "D1" and deg == r["raw_output"]:
                    fb += 1
                rows.append({
                    "method": f"{sysname}_{lvl}", "instance_id": iid,
                    "raw_output": deg, "machine_failure": int(r["machine_failure"]),
                    "seed": r.get("seed", 42),
                })
    df = pd.DataFrame(rows)
    out = C.RUNS_DIR / "reports_degraded.parquet"
    df.to_parquet(out, index=False)
    print(f"degraded set: {len(df)} reports -> {out.name}")
    print(df.groupby("method").size().to_string())
    print(f"D1 fallback-to-truncation rate: {fb}/{2*len(ids)}")


if __name__ == "__main__":
    main()
