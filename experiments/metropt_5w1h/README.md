# MetroPT 5W1H — In-Service Fleet Instantiation

Third instantiation of the question-centered knowledge-unit framework,
evaluated on real in-service telemetry from the Porto metro Air
Production Unit (MetroPT). This is the principal evaluation of the
RESS manuscript ("Question-Centered Knowledge Units with Conformal
Verification: Measuring and Selectively Bounding Factual Error Rates of
LLM Maintenance Reports on an In-Service Fleet").

## Corpus

Three vehicle-epochs of the same APU type (see `data-manifests/` for
full provenance and sensor semantics; raw CSVs are NOT redistributed
here — fetch with `src/download_metropt3.py` and
`src/download_zenodo_file.py`):

| epoch | vehicle | span | rate |
|---|---|---|---|
| metropt3 | 2020 unit | 2020-02-01 → 09-01 | 10 s |
| metropt2022 | train A | 2022-01-01 → 06-02 | 1 Hz (thinned 5/6 in the released export) |
| metropt2022B | train B | 2022-04-28 → 07-28 | 1 Hz |

Nine documented failures anchor the ground truth; every published label
was forensically verified against the raw telemetry (documented
adjudications in `runs/label_check/`). Instances: 105 windows
(37 positives + 68 controls incl. 3 out-of-coverage candidates).

## Pipeline (one command per stage; all scripts in `src/`)

1. Corpus preparation and frozen window manifests — obtain the raw datasets
   using the download utilities above, verify them against `data-manifests/`,
   and use the archived window manifests in `runs/metropt3/` for the reported
   evaluation (failure-anchored windows, matched controls, and coverage guards).
2. `metropt_features.py` — frozen 11-feature window digest.
3. `metropt_calibrate.py` → `metropt_spec_build.py` — healthy-pool
   baselines + frozen `knowledge/metropt/metropt_spec.json`
   (zero-false-alarm discipline; ghost exclusions logged).
4. `metropt_handlers.py` — two-tier rule/score handler with selective
   acceptance (rule(w) ∨ s(w) ≤ t; abstention otherwise).
5. `metropt_knowledge.py` / `metropt_gt.py` — 24-question 5W1H bank,
   KU inventory, templates, telemetry-grounded GT with lead-time
   honesty.
6. `metropt_campaign.py` — report generation, six systems
   (B1/B2/B3/B3R/B5/B8) × 105 windows, resume-safe.
7. `metropt_l1.py` — repaired-from-day-one Layer-1 checker.
8. `metropt_l2l3_eval.py` — L2/L3 judging (DeepSeek-V4-Flash primary;
   budgets 5000/3000; zero-claim-floor sweeps).
9. `metropt_analysis.py` — main three-layer table, lead-time curve,
   decision usefulness. `metropt_action_eval.py` — blind action
   classifier.
10. Robustness: `metropt_r10_blockers.py` (conformal coverage check),
    `metropt_loeo.py`, `metropt_crossvehicle.py`,
    `metropt_b3r_campaign.py`, `metropt_seed7.py`,
    `metropt_rejudge_seed7.py`, `metropt_risk_coverage.py`,
    `metropt_b8r.py`.
11. Figures: `metropt_figures.py`, `figures_r10.py` (paper Fig. 3–6).

## Human studies

- `human_validation/` — three-rater blinded report-rating study
  (30 windows × 4 systems; instrument + ratings + analysis).
- `adjudication/` — blinded independent-engineer label audit
  (nine events, three exclusions, 37 strata, two contested items;
  instrument + rating + analysis). The auditor: mechatronics lecturer
  and lab engineer with prior industrial maintenance experience on
  rotating equipment; none of the authors.

## Discipline notes

- `docs/PREREGISTRATION.md` was frozen before report generation.
- `docs/KE_COST_LOG.md` — the 19 person-hour contemporaneous log.
- The judging parquet writer is single-writer; resume keys are
  `(system, vehicle, window_id)`.
- Install the MetroPT runtime dependencies with `pip install -r requirements.txt`.
  LLM-dependent stages additionally require provider API keys configured from
  `.env.example`; the frozen outputs under `runs/metropt3/` can be inspected
  without rerunning API calls.

## Key frozen outputs

The main experiment outputs are archived under `runs/metropt3/`, including
`metropt_main_summary.csv` (six-system main table),
`metropt_detection_matrix.csv` (9 signature windows),
`metropt_positive_strata.csv`, `metropt_conformal_coverage.csv`,
`metropt_loeo.csv`, `metropt_crossvehicle_folds.csv`,
`metropt_per_vehicle_final.csv`, `metropt_gt.json`, and the campaign
parquets. Human-rating materials are maintained separately in
`human_validation/`, the independent engineer audit in `adjudication/`,
and label-forensics notes in `runs/label_check/`.
