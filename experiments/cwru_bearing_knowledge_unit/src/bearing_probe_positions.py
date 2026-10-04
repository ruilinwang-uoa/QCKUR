"""Robustness probe: run the frozen v3 two-tier decision on the 16 excluded
OR @3/@12 clock-position files (never used in calibration or training).

Downloads the files from the official CWRU site on first run. Expected label
for all 16 is OR; this measures position generalization of the rule tier and
of the RandomForest model handler, both of which saw only @6 outer-race
faults during development. Load assignments follow the official page table:
0.007"@3 = 144-147 (L0-3); 0.007"@12 = 156(L0), 158-160 (L1-3);
0.021"@3 = 246-249 (L0-3); 0.021"@12 = 258-261 (L0-3).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bearing_v3 import DATA, FEATS, features  # noqa: E402

OFFICIAL = "https://engineering.case.edu/sites/default/files/{}"

LOADS = {}
for nums in [(144, 145, 146, 147), (246, 247, 248, 249), (258, 259, 260, 261)]:
    for n, load in zip(nums, (0, 1, 2, 3)):
        LOADS[f"{n}.mat"] = load
LOADS["156.mat"], LOADS["158.mat"], LOADS["159.mat"], LOADS["160.mat"] = 0, 1, 2, 3

PROBE = {**{f"{n}.mat": "OR@3" for n in (144, 145, 146, 147, 246, 247, 248, 249)},
         **{f"{n}.mat": "OR@12" for n in (156, 158, 159, 160, 258, 259, 260, 261)}}


def main() -> None:
    manifest = json.load(open(DATA / "cwru_manifest.json", encoding="utf-8"))
    for name in PROBE:
        dest = DATA / name
        if not dest.exists():
            urllib.request.urlretrieve(OFFICIAL.format(name), dest)
            print(f"downloaded {name}")

    # rebuild the frozen v3 decision (thresholds from loads 0/1 + RF + guard)
    rows = []
    for name, meta in manifest.items():
        num = name.split(".")[0]
        f = features(num, meta)
        truth = ("normal" if meta["condition"] == "normal"
                 else meta["component"].split("@")[0])
        rows.append({"file": name, "truth": truth, "load_hp": meta["load_hp"],
                     "defect_in": meta["defect_size_in"], **f})
    df = pd.DataFrame(rows)
    cal = df[df.load_hp <= 1]
    tau_ir = float(np.sqrt(cal[cal.truth == "IR"].BPFI.min()
                           * cal[cal.truth != "IR"].BPFI.max()))
    or_mask = (cal.truth == "OR") & (cal.defect_in != 0.014)
    tau_or = float(np.sqrt(cal[or_mask].BPFO.min() * cal[~or_mask].BPFO.max()))
    tau_dom = 2.4  # frozen 2026-09-08 (geometric midpoint 3.8 / 1.6)
    clf = RandomForestClassifier(n_estimators=500, random_state=42,
                                 class_weight="balanced")
    clf.fit(cal[FEATS], cal.truth)

    print("\nprobe (unseen OR clock positions):")
    ok = {}
    for name, label in sorted(PROBE.items(), key=lambda kv: int(kv[0][:3])):
        num = name.split(".")[0]
        meta = {"load_hp": LOADS[name], "condition": "fault",
                "component": label, "defect_size_in": 0.0}
        f = features(num, meta)
        if f["BPFI"] >= tau_ir and f["BPFI"] / max(f["BPFO"], f["BALLFAM"]) >= tau_dom:
            pred, tier = "IR", "rule"
        elif f["BPFO"] >= tau_or:
            pred, tier = "OR", "rule"
        else:
            X = pd.DataFrame([{k: f[k] for k in FEATS}])[FEATS]
            pred, tier = clf.predict(X)[0], "model"
        good = pred == "OR"
        ok.setdefault(label, []).append(good)
        print(f"  {name:>8} {label}: BPFI={f['BPFI']:6.1f} BPFO={f['BPFO']:6.1f} "
              f"-> {pred:>6} [{tier}] {'Y' if good else 'N'}")
    for label, v in ok.items():
        print(f"{label}: {sum(v)}/{len(v)}")


if __name__ == "__main__":
    main()
