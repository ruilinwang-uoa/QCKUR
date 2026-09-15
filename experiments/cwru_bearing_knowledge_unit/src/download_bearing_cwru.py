"""Download the curated CWRU bearing-fault subset for Phase 3 (second dataset).

Verified 2026-09-08:
  - Official CWRU fault files are no longer directly reachable (404 on the
    historical /sites/default/files/{fault}.mat pattern); the Zenodo mirror
    (records 10986655 Pt.1 + 10987113 Pt.2, DOI 10.5281/zenodo.10987112
    concept) is complete and scriptable.
  - Smoke-tested: 292.mat downloads (200, ~2.9 MB) and loads with
    scipy.io.loadmat (X*_DE_time / X*_FE_time / X*_BA_time / X*RPM channels).

Curated subset: the classic 12 kHz drive-end diagnosis benchmark --
Normal + IR/B/OR x {0.007, 0.014, 0.021} inches x loads {0,1,2,3} hp
(OR centered at 6 o'clock), 40 files, ~120 MB total.

Usage (paper2code conda python):
    python code/src/download_bearing_cwru.py            # download all
    python code/src/download_bearing_cwru.py --verify   # manifest + keys only

Each file's condition label lives in MANIFEST; the GT builder for Phase 3
should read this manifest, not re-derive labels from file names.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import scipy.io

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bearing_cwru"

# Download sources, tried in order: the official CWRU site (authoritative;
# verified alive 2026-09-08 -- only the historical letter-named URL pattern
# is dead, numbered files like 105.mat/237.mat resolve directly) and the
# Zenodo mirror (records verified 2024-04-17 deposits, alive 2026-09-08).
OFFICIAL = "https://engineering.case.edu/sites/default/files/{}"
RECORDS = [10986655, 10987113]  # Pt.1 (12k DE + normal), Pt.2 (12k FE)

# file number -> (condition, component, defect_size_in, load_hp)
# Standard 12 kHz drive-end benchmark; numbers are the canonical CWRU IDs.
MANIFEST = {}
for nums, (cond, comp, size) in {
    (97, 98, 99, 100): ("normal", "none", 0.0),
    (105, 106, 107, 108): ("fault", "IR", 0.007),
    (169, 170, 171, 172): ("fault", "IR", 0.014),
    (209, 210, 211, 212): ("fault", "IR", 0.021),
    (118, 119, 120, 121): ("fault", "B", 0.007),
    (185, 186, 187, 188): ("fault", "B", 0.014),
    (222, 223, 224, 225): ("fault", "B", 0.021),
    (130, 131, 132, 133): ("fault", "OR@6", 0.007),
    (197, 198, 199, 200): ("fault", "OR@6", 0.014),
    (234, 235, 236, 237): ("fault", "OR@6", 0.021),
}.items():
    for num, load in zip(nums, (0, 1, 2, 3)):
        MANIFEST[f"{num}.mat"] = {
            "condition": cond, "component": comp,
            "defect_size_in": size, "load_hp": load,
            "sampling_hz": 12000, "channel": "DE",
        }


def file_url(name: str) -> str | None:
    candidates = [OFFICIAL.format(name)] + [
        f"https://zenodo.org/api/records/{rec}/files/{name}/content"
        for rec in RECORDS
    ]
    for url in candidates:
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 200:
                    return url
        except Exception:
            continue
    return None


def verify(path: Path, num: str) -> bool:
    m = scipy.io.loadmat(path)
    # channel keys are zero-padded to three digits (X097_DE_time, X105_DE_time)
    return (f"X{num}_DE_time" in m
            or f"X{int(num):03d}_DE_time" in m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="only check local files, no downloads")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "cwru_manifest.json"
    manifest_path.write_text(json.dumps(MANIFEST, indent=1), encoding="utf-8")
    print(f"manifest ({len(MANIFEST)} files) -> {manifest_path}")

    ok = missing = failed = 0
    for name in sorted(MANIFEST, key=lambda s: int(s.split(".")[0])):
        dest = OUT / name
        if dest.exists() and verify(dest, name.split(".")[0]):
            ok += 1
            continue
        if args.verify:
            print(f"  MISSING {name}")
            missing += 1
            continue
        url = file_url(name)
        if url is None:
            print(f"  NOT FOUND on mirror: {name}")
            missing += 1
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            if verify(dest, name.split(".")[0]):
                ok += 1
                print(f"  ok {name}")
            else:
                failed += 1
                print(f"  BAD CHANNELS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {e}")
    print(f"done: ok={ok} missing={missing} failed={failed}")
    sys.exit(0 if (missing == 0 and failed == 0) else 1)


if __name__ == "__main__":
    main()
