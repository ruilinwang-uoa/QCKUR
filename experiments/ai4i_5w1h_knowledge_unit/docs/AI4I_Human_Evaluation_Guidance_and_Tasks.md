# AI4I Human Evaluation — Executed Study Documentation

This document describes the targeted human-validation study **as executed**
(2026-09-12/13). An earlier draft of this file described a wider
eight-system design; the executed study targets the four systems that span
the contested comparisons of the manuscript, using the self-contained rating
form `runs/human_validation/rating_form.html`.

## Purpose

The study validates the existing AI4I report-evaluation protocol with human
raters. It is **not** a new report-generation experiment: all reports are
the unmodified canonical outputs behind the main results table, and the
primary 100-instance scoring remains fully automatic.

## Executed design

- **30 AI4I machine instances** (11 faulted, 19 healthy) — the same
  fault-enriched subset used in the synthetic-anchor study.
- **Four anonymized reporting systems per instance**: the template-only
  ceiling (B1), the traditional D2T pipeline (B2), the tool-using agent
  (B5), and the proposed framework (B8) — the systems between which the
  manuscript's contested comparisons lie.
- **120 reports in total**; every report rated by **every** evaluator
  (complete-block design).
- **Five evaluators**: two faculty members, two students, and one
  mechatronics engineer (domain-relevant to the problem setting); none of
  them an author of the manuscript. Mean time on task ≈ 1 hour.
- **Blinding**: reports carry randomized letters (A–D), re-randomized
  per instance; the letter key (`sample_manifest.json`) was never shown to
  evaluators.
- Participation was voluntary, the task was explained at recruitment, and
  responses were collected anonymously; no personal data were retained.

## What each evaluator did (per instance)

For each of the four anonymized reports, given the machine operating
information, the verified analytical results, the verified physical-rule
verdicts, and the six reporting questions/aspects (What/Why/When/Where/
Who/How):

1. Rate **faithfulness, completeness, coherence, and actionability**
   (1–5 each) against the printed facts.
2. Tick which of the six 5W1H aspects the report addresses
   (coverage is marked independently of correctness of the content).
3. **Rank the four reports of the instance by readability**
   (1 = most readable).

No overall-correctness category and no A–E error taxonomy was assigned in
this study (that taxonomy belongs to the cross-domain benchmark); a report
addressing an aspect incorrectly is still marked as addressing it, with the
error reflected in the dimension ratings.

Rating criteria:

- **Faithfulness**: 5 = all checkable claims agree with the facts;
  4 = essentially accurate, minor unsupported elaboration; 3 = one clear
  contradiction or several unsupported specifics; 2 = core status/fault mode
  wrong; 1 = fabricated.
- **Completeness**: 5 = conveys essentially all facts; 4 = most;
  3 = about half; 2 = two aspects; 1 = at most one.
- **Coherence**: 5 = fully consistent; 4 = minor awkwardness; 3 = internal
  tension or self-contradictory arithmetic; 2 = clear self-contradiction;
  1 = incoherent.
- **Actionability**: 5 = specific action + responsible role + timing +
  component; 4 = one missing; 3 = generic but directionally useful;
  2 = vague; 1 = no usable guidance.

## Data and reproduction

All artifacts are in `runs/human_validation/`:

| File | Content |
|---|---|
| `rating_form.html` | The self-contained instrument (autosaves; one JSON export per rater) |
| `sample_manifest.json` | Instance list + per-instance letter→system key (blinding key) |
| `human_validation_ratings-{A..E}.json` | The five anonymized rater exports (raw data) |
| `human_ratings_long.csv` | Long-format per-(rater, instance, system) ratings |
| `human_system_means.csv` | Per-system means (analysis output) |
| `ws3_system_means.csv` | Per-system means + agreement/correlation statistics |

Reproduce with:

```
python src/analyze_human_validation.py runs/human_validation/human_validation_ratings-*.json
python src/ws3_human_validation.py     # full statistics incl. inter-rater and human-vs-judge
```

`ws3_human_validation.py` verifies completeness (600 ratings), decodes the
per-instance blinding key, computes per-system means and the human composite
(the manuscript's Eq. 7 weights), inter-rater agreement (mean pairwise
Spearman per dimension; Kendall's W for the readability ranks), report-level
human-vs-judge correlations against the frozen Layer-3 parquets, and the
B1-vs-B8 readability test.
