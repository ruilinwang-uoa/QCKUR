# Week-1 findings — MetroPT industrial case (2026-09-20)

## LATE-BREAKING (21:30): dataset_train.csv = the full Jan–Jun 2022 record

Zenodo rec 7766691's second file `dataset_train.csv` (1.6 GB) probed via
Range (head+tail): spans **2022-01-01 06:00 → 2022-06-02 15:49:53**, 1 Hz,
same 21-col schema. Consequences:

- **ALL THREE 2022 failures are in span** (air leak clients Feb 28 /
  air leak dryer Mar 23 / oil leak May 30–Jun 2) — vs only the oil leak
  in MetroPT2.csv.
- File ends exactly at the oil-leak window → corroborates "train removed
  from service" and pins the oil-leak observation horizon to Jun 2 ~15:50.
- MetroPT2.csv's Apr 28–Jul 28 span then = overlap (Apr 28–Jun 2, verify
  identical) + **post-failure extension Jun 2–Jul 28** (repaired/replaced
  unit? verify signal-level; candidate healthy-era or repair-verification
  vignette).
- Combined 2022 coverage: Jan 1 – Jul 28. Downloads: in progress
  (Zenodo, resumable). NOTE: "train" in the filename is their naming;
  verify after landing that failure windows are NOT excised (row counts
  around each failure must match ~1 Hz expectation).

## Network facts (2026-09-20, this box)

- UCI: urllib TLS dies (SSLEOFError), curl works but server does NOT
  support byte ranges (rc=33 "cannot resume") + resets mid-transfer
  (rc=56) at ~20 KB/s → each reset = full restart. MetroPT-3 crawl is a
  lottery ticket; dataset_train.csv replaces the dependency if verified.
- Zenodo: python-only (curl cannot resolve host — proxy asymmetry),
  ~400 KB/s, drops mid-transfer but honors Range → resumable
  (download_care.py + download_zenodo_file.py both Range-resume).

## Assets acquired

| Source | Span | Rate | Status |
|---|---|---|---|
| MetroPT2.csv (Zenodo 7766691, 1.2 GB) | 2022-04-28 → 2022-07-28 | 1 Hz (median 0.991 s), 7.12M rows, 21 cols | **on disk, verified** |
| MetroPT-3 (UCI 791, 208 MB) | 2020-02 → 2020-08 | ~0.1 Hz, 1.5M rows, 15 cols | downloading (UCI ~30 KB/s) |
| CARE wind (Zenodo 10958775, 5.5 GB) | 3 farms, 95 sub-datasets | 10-min SCADA | downloading (W4-gate track) |

## The design win: two failure modes, two epochs, one system type

- **2020 (MetroPT-3)**: 4 air-leak events WITH maintenance timestamps
  (UCI table; event-2 maintenance date has a known typo → corrected
  hypothesis 2020-05-30 12:00, flagged for W2 cross-check).
- **2022 (MetroPT2.csv)**: oil leak on compressor, 2022-05-30 12:00 →
  ~2022-06-02 (END PROVISIONAL — verify vs github.com/gvbeta/MetroPT
  label files before GT freeze). Two more 2022 air leaks fall OUTSIDE
  this file's span (Feb 28 / Mar 23 < Apr 28 start) — skipped by guard.
- Same APU system type across epochs → ONE question bank instantiated
  across two operational periods = knowledge-reuse evidence for QCKU.
  (Do NOT claim same physical train — unverified.)

## Operator's deterministic rules (Veloso 2022, Sci Data 9:764) — bank seed

- LPS active when pressure < **7 bar**; MPG loads compressor when stored
  air < **8.2 bar**; H1 valve at **10.2 bar**; Motor_current modes
  **0 A (off) / 4 A (offloaded) / 7 A (loaded)**; DV_pressure = **0 under
  load**; Oil_level = 1 when oil BELOW expected; COMP=1 ⇔ no air admission;
  DV_electric=1 ⇔ under load; Towers = active dryer tower.
- GPS columns (2022 file) enable in-service vs parked segmentation
  (paper Table 1 parking polygons); Flowmeter adds a true airflow signal.
- Paper reports mean **6.8 h anomaly lead time** before air-leak failures.

## Coverage probes (2022 file)

- Oil-leak ±7d window: 1,349,112 rows ≈ **92% of expected 1 Hz**,
  4 sub-gaps — event well covered.
- Healthy supply ≥7d from event: **1,602 h** (Apr 60 / May 468 / Jun 489 /
  Jul 585 h) — ample for matched controls.
- Global: 69 gaps >10min over 3 months, largest 19.5 h.

## Draft instances (windows built; anchor provisional for 2022)

- `code/runs/metropt3/window_manifest_metropt2.json`: **5 positives**
  (oil leak × offsets 6/12/24/48/72 h before anchor) + **10 matched
  controls** (month + hour-of-day ±2h, ≥72h from events, ≥80% coverage).
- 2020 source pending CSV arrival → expect +20 positives (4 events × 5)
  + 40 controls → combined n ≈ 75.

## W1 instance inventory (FINAL for the week)

**ADDENDUM (Sep 21, 01:56): MetroPT-3 (2020) LANDED** — the UCI lottery
ticket completed after 1h57m (restart-until-valid loop). CSV verified:
1,516,948 rows, 2020-02-01 → 09-01 03:59, median cadence 10 s, 230 gaps
(~863 h missing overall). Event windows 79–92% covered; healthy supply
3,785 h. DV_electric column ABSENT in 2020 (14 of 15 signals; bank must
treat it as 2022-only). **Signature check is textbook**: during events
TP2 1.2→8.5 bar, H1 7.7→0.1, DV_pressure 0.02→1.9 (nonzero where it must
be 0), Motor_current 2.0→5.6 A, Oil_temperature 62→71–77 °C — the
air-leak → overwork → heat physics is directly visible in the means.
Windows: **15 positives + 30 controls** (5 positive slots dropped on
coverage: ev3 6h/12h at 21%/50% — telemetry gaps immediately before
maintenance, possibly operationally meaningful; ev4 12/24/72h similar).

**COMBINED W1 INVENTORY: 30 positives + 54 controls = 84 instances.**
2020: 4 air-leak events, MAINTENANCE-TIMESTAMP anchors (work-order
anchoring). 2022: 2 air leaks + 1 oil leak, event-end anchors.

**Primary stream: dataset_train_2022.csv → window_manifest_metropt2022.json**
- **15 positives** — all 3 events (2 air leaks + oil leak) × offsets
  {6,12,24,48,72} h before anchor (event end; no maintenance timestamps
  in the 2022 annotation — verify vs gvbeta/MetroPT labels in W2).
- **24 controls** (matched month + hour ±2h, ≥72h from events, ≥80% of
  OBSERVED daily density). 6 slots dropped and WHY: the file ends at the
  oil-leak removal — June 1 is the only full June day and it lies inside
  ev3's exclusion region, so month-matched June controls are structurally
  unavailable. PREREG DECISION: accept 24, or relax matching to May/July
  for ev3's 6/12/24 h slots. n = 39 either way (target 30–60 ✓).
- Coverage guard fixed to calibrate against observed daily rows (this
  source is thinned to 5/6 of 1 Hz — 72,001 rows/day exactly); nominal-
  rate guards silently misjudge thinned sources.

**Companion assets:** MetroPT2.csv (1 Hz full-rate, Apr 28–Jul 28: overlap
+ post-failure Jun 2–Jul 28 repair-era candidate); CARE 5.5 GB verified
md5, shelved for W4 gate; MetroPT-3 (2020, UCI) still crawling — now
strictly optional (its 4 events would add positives, not unblock anything).

## Open items (W2 gate)

1. Verify oil-leak END + any maintenance record vs gvbeta/MetroPT labels.
2. Cross-check 2020 event-2 maintenance typo vs paper/UCI PDF.
3. MetroPT-3 CSV verify + explore + windows (scripts ready).
4. Reconcile "same train?" — default to "same system type" language.
5. GPS-based in-service segmentation (2022) — decide before GT freeze.

## Scripts (code/src/)

download_metropt3.py · download_care.py · metropt3_explore.py ·
metropt3_windows.py (now --csv/--manifest/--tag generalized)
Manifests: code/data/metropt3/{metropt3_manifest.json (on arrival),
metropt2_manifest.json}
