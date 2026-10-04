"""Download the MetroPT-3 dataset for the industrial in-service case (attempt 4).

Verified 2026-09-20:
  - UCI Machine Learning Repository, dataset ID 791 ("MetroPT-3 Dataset",
    donated 2023-03-21). Direct archive:
    https://archive.ics.uci.edu/static/public/791/metopt+3.zip  (~208 MB)
  - License: CC BY 4.0. Citation: Davari et al. 2021 (dataset paper),
    Veloso et al. 2022 Scientific Data (family paper, MetroPT-1/2/3).
  - NOTE: "MetroPT-5"/"MetroPT-6" do NOT exist (verified 2026-09-20) --
    the series is MetroPT-1/2/3 only; MetroPT-3 is the UCI-hosted one.
  - Contents: MetroPT3(AirCompressor).csv (~208 MB) + Data Description PDF.
    ~1.52M rows, 2020-02..2020-08, ONE train's Air Production Unit compressor,
    15 signals: 7 analog (TP2, TP3, H1, DV_pressure, Reservoirs,
    Motor_current, Oil_temperature) + 8 digital (COMP, DV electric, Towers,
    MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses).
    Nominal 1 Hz logging, but effective cadence over the span is ~0.1 Hz
    (10 s); timestamps are European dd/mm/yyyy.
  - The UCI page carries the company's failure/maintenance table (below).
    KNOWN TYPOS in that table, handled explicitly:
      * event 2 prints maintenance "30Apr 12:00" -- impossible (event runs
        29-30 May); working hypothesis 30 May 12:00, to be cross-checked
        against Veloso et al. 2022 in the GT-building week (W2). The
        pre-registration must state which reading is used.
      * event 1 lists no maintenance timestamp on the page.

Usage (paper2code conda python):
    python code/src/download_metropt3.py            # download + verify + manifest
    python code/src/download_metropt3.py --verify   # local files + manifest only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "metropt3"

# Verified 2026-09-20 by probing: the slug is "metropt+3+dataset" (the
# shorter "metopt+3" / "metropt+3" variants 404). urllib's TLS handshake to
# this host can fail with SSLEOFError (observed 2026-09-20) while curl
# succeeds -- the script falls back to curl automatically.
ZIP_URL = ("https://archive.ics.uci.edu/static/public/791/"
           "metropt+3+dataset.zip")
UCI_PAGE = "https://archive.ics.uci.edu/dataset/791"

# Failure/maintenance table transcribed verbatim from the UCI page
# (accessed 2026-09-20). `maintenance` = as printed; event 2's value is
# impossible as printed and carries an explicit correction hypothesis.
EVENTS = [
    {"id": 1, "start": "2020-04-18 00:00", "end": "2020-04-18 23:59",
     "failure_type": "Air leak", "consequence": "High stress",
     "maintenance_as_printed": None, "maintenance": None,
     "note": "page lists no maintenance timestamp for this event"},
    {"id": 2, "start": "2020-05-29 23:30", "end": "2020-05-30 06:00",
     "failure_type": "Air leak", "consequence": "High stress",
     "maintenance_as_printed": "2020-04-30 12:00",
     "maintenance": "2020-05-30 12:00",
     "note": "as-printed date precedes its own event (typo); corrected to "
             "30 May 12:00 -- CROSS-CHECK vs Veloso et al. 2022 in W2 "
             "before GT freeze; pre-registration must state the choice"},
    {"id": 3, "start": "2020-06-05 10:00", "end": "2020-06-07 14:30",
     "failure_type": "Air leak", "consequence": "High stress",
     "maintenance_as_printed": "2020-06-08 16:00",
     "maintenance": "2020-06-08 16:00", "note": ""},
    {"id": 4, "start": "2020-07-15 14:30", "end": "2020-07-15 19:00",
     "failure_type": "Air leak", "consequence": "High stress",
     "maintenance_as_printed": "2020-07-16 00:00",
     "maintenance": "2020-07-16 00:00", "note": ""},
]

# Veloso et al. 2022 document leak LOCATION for the family (failure 1: leak
# on clients; failure 2: leak on air dryer via pneumatic pilot valve) --
# reconcile against this in W2 when assigning component-level GT.
FAMILY_FAILURE_LOCATIONS = {
    "failure_1": "air leak on clients",
    "failure_2": "air leak on air dryer (pneumatic pilot valve malfunction)",
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_ok(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path):
            return True
    except zipfile.BadZipFile:
        return False


def curl_resume(path: Path) -> None:
    """UCI-specific downloader (observed 2026-09-20): the server does NOT
    support byte ranges (curl rc=33 'cannot resume') AND resets transfers
    mid-flight (rc=56) at ~20 KB/s -- so every failure forces a FULL
    restart. Loop fresh transfers until the zip validates; cap at 10 full
    restarts. curl -o truncates on open, so each invocation starts clean."""
    import subprocess
    import time

    restarts = 0
    while restarts < 10:
        restarts += 1
        rc = 1
        for attempt in range(1, 31):
            rc = subprocess.run(
                ["curl", "-L", "--retry", "3", "--retry-delay", "5",
                 "-o", str(path), ZIP_URL]).returncode
            if rc == 0:
                break
            print(f"  curl rc={rc} at {path.stat().st_size >> 20} MB; "
                  f"restarting transfer ({attempt}/30)", flush=True)
            time.sleep(10)
        if rc == 0 and zip_ok(path):
            return
        print(f"  transfer incomplete -- full restart ({restarts}/10)",
              flush=True)
    sys.exit("UCI download did not complete after 10 full restarts -- "
             "fetch the URL in a browser into code/data/metropt3/, "
             "then re-run this script (it will pick the zip up)")


def verify(csv_path: Path) -> dict:
    """Structural verification: columns, row count, time span, cadence."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df["timestamp"], format="mixed", dayfirst=True)
    span = (str(ts.min()), str(ts.max()))
    n = len(df)
    cadence_s = float((ts.max() - ts.min()).total_seconds() / max(n - 1, 1))
    return {
        "rows": n,
        "columns": list(df.columns),
        "time_span": span,
        "mean_cadence_s": round(cadence_s, 2),
        "n_timestamp_duplicates": int(ts.duplicated().sum()),
        "n_na_timestamps": int(ts.isna().sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="only check local files, no download")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    zip_path = OUT / "metopt+3.zip"
    csv_path = OUT / "MetroPT3(AirCompressor).csv"
    pdf_path = OUT / "Data Description_Metro.pdf"

    if not args.verify and not csv_path.exists():
        if not zip_path.exists():
            print(f"downloading {ZIP_URL} (~208 MB) ...")
            try:
                urllib.request.urlretrieve(ZIP_URL, zip_path)
            except Exception as e:  # noqa: BLE001
                print(f"  urllib failed ({e}); falling back to curl ...")
                curl_resume(zip_path)
        elif not zip_ok(zip_path):
            print(f"partial zip on disk ({zip_path.stat().st_size >> 20} MB)"
                  " -- UCI cannot resume; restarting with curl ...")
            curl_resume(zip_path)
        if not zip_ok(zip_path):
            curl_resume(zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(OUT)
        print("extracted:", [p.name for p in OUT.iterdir()])

    stats = {}
    if csv_path.exists():
        stats = verify(csv_path)
        print(f"csv ok: {stats['rows']} rows, {stats['time_span'][0]} .. "
              f"{stats['time_span'][1]}, cadence {stats['mean_cadence_s']}s")
    else:
        print("csv MISSING")

    manifest = {
        "dataset": "MetroPT-3",
        "source_zip": ZIP_URL,
        "ucy_page": UCI_PAGE,
        "access_date": "2026-09-20",
        "license": "CC BY 4.0",
        "citation": [
            "Davari, N., Veloso, B., Ribeiro, R., Gama, J. (2021). "
            "MetroPT-3 dataset paper.",
            "Veloso, B. et al. (2022). The MetroPT dataset for predictive "
            "maintenance. Scientific Data 9, 764.",
        ],
        "system": "Air Production Unit (APU) compressor, one train, "
                  "Metro do Porto; APU feeds braking and door circuits",
        "signals": {
            "analog": ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                       "Motor_current", "Oil_temperature"],
            "digital": ["COMP", "DV_electric", "Towers", "MPG", "LPS",
                        "Pressure_switch", "Oil_level", "Caudal_impulses"],
            "note": "column names recorded AS FOUND in the CSV (the file is "
                    "known to contain misspellings, e.g. 'eletric'); "
                    "verify() writes the authoritative list",
        },
        "failure_events": EVENTS,
        "family_failure_locations": FAMILY_FAILURE_LOCATIONS,
        "known_quirks": [
            "timestamps European dd/mm/yyyy -- always parse dayfirst",
            "nominal 1 Hz logging, effective cadence ~0.1 Hz (10 s)",
            "UCI failure table contains internal typos (see failure_events)",
        ],
        "files": {},
    }
    for p, key in ((zip_path, "zip"), (csv_path, "csv"), (pdf_path, "pdf")):
        if p.exists():
            manifest["files"][key] = {"name": p.name,
                                      "bytes": p.stat().st_size,
                                      "md5": md5_of(p)}
    manifest["verified"] = stats

    (OUT / "metropt3_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest -> {OUT / 'metropt3_manifest.json'}")

    ok = csv_path.exists() and stats.get("rows", 0) > 1_000_000
    print(f"done: {'OK' if ok else 'INCOMPLETE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
