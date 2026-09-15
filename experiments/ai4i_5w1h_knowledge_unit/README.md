# AI4I 5W1H Knowledge-Unit Experiment

Reproducibility materials for the AI4I 2020 predictive-maintenance instantiation reported in *Question-Centered Knowledge Units for Bounding LLM Hallucination in Predictive Maintenance Reporting*.

The experiment implements the question-centered knowledge-unit framework with four explicit components: question matching (**Q**), deterministic model handlers (**M**), encoded physical rules (**P**), and answer templates (**T**). The LLM is used for bounded question matching and linguistic refinement rather than for deriving analytical facts.

## Experimental scope

The source dataset contains 10,000 AI4I records. A stratified sample of 315 records (seed 42) is used to construct and verify the knowledge layer and programmatic ground truth. The instantiated library contains 203 executable knowledge units distributed across the six 5W1H categories:

- What: 61
- Why: 29
- When: 26
- Where: 25
- Who: 31
- How: 31

The library is backed by 46 answer templates and 46 deterministic model handlers. Programmatic ground truth is generated for all 203 × 315 = 63,945 knowledge-unit/record pairs.

The primary report-generation and evaluation campaign uses a failure-enriched stratified subset of 100 records (seed 42), containing all 11 faulted records in the 315-record sample. Predictive models are trained under five seeds (42, 1, 7, 2024, 123).

## Environment

Set up the Python environment described in `requirements.txt`, then configure the LLM endpoints in a git-ignored `.env` file using `.env.example`.

```bash
cp .env.example .env
cd src
$PY llm_smoke_test.py
```

The main local preparation pipeline can be run with:

```bash
cd src
$PY run_all.py
```

## Pipeline (`src/`)

| Script | Purpose |
|---|---|
| `data.py` | dataset integrity checks, derived features, 315-record sample, EDA |
| `encode_knowledge.py` | physical-rule and domain-map encoding checks |
| `train_models.py` | LR/RF/GBM training under five seeds; F1-macro and AUPRC |
| `question_bank.py` | 203 executable 5W1H questions |
| `ground_truth.py` | 63,945 programmatic ground-truth records |
| `ku_inventory.py` | knowledge-unit inventory |
| `layer1_eval.py` | Layer-1 programmatic fact checking and perturbation tests |
| `run_experiment.py` | B1–B8 report generation |
| `layer2_eval.py` | Layer-2 claim consistency / QA closed-loop evaluation |
| `layer3_eval.py` | Layer-3 multidimensional report-quality judging |
| `stats_eval.py` / `friedman_omnibus.py` | paired and omnibus statistical analyses |
| `summary_results.py` / `summary_report.py` | result aggregation |

Core modules include `config.py`, `physics_rules.py`, `ku_handlers.py`, `templates.py`, `background.py`, `corpus.py`, `systems.py`, `llm_call.py`, and `llm_config.py`.

The primary RAG configuration uses a TF-IDF retriever with `k=5`. A dense-retrieval variant is retained as a control study rather than as the headline RAG configuration.

## Compared systems (`src/systems.py`)

| ID | System | Knowledge organization | LLM |
|---|---|---|---|
| B1 | Template-only | framework templates, no LLM refinement | no |
| B2 | Traditional D2T | hand-crafted content rules + fixed templates | no |
| B3 | Direct LLM | none | yes |
| B4 | Few-shot LLM | three fixed exemplars | yes |
| B5 | Tool-using agent | function-calling tools; analytical structure remains implicit | yes |
| B6 | RAG | unstructured domain retrieval; no framework templates | yes |
| B7 | RAG + CoVe | retrieval followed by chain-of-verification | yes |
| B8 | **Proposed framework** | **explicit Q/M/P/T knowledge units + bounded LLM refinement** | yes |

The primary campaign uses GLM-4-Flash for LLM-based generation. The cross-LLM campaign repeats the comparison with DeepSeek-V4-Flash. Deterministic systems B1 and B2 do not call an LLM for report generation.

Example commands:

```bash
cd src
$PY run_experiment.py smoke --provider openai
$PY run_experiment.py full --n 100 --systems all --seeds 42
```

Additional seed-1 and seed-7 runs and judge-replication campaigns are orchestrated by the batch scripts in the experiment root.

## Evaluation protocol

The primary 100-instance AI4I campaign uses a three-layer protocol.

**Layer 1 — programmatic fact checking** combines numeric tolerance, physical-rule consistency, and fault classification:

`L1 = 0.30 × S_num + 0.30 × S_rule + 0.40 × S_cls`.

**Layer 2 — report-level consistency** checks extracted claims against verified analytical facts and records 5W1H coverage and contradictions:

`L2 = 0.50 × C_cons + 0.25 × C_cov + 0.25 × (1 − C_contra)`.

**Layer 3 — judged report quality** scores faithfulness, completeness, 5W1H coverage, coherence, and actionability on the predefined rubric:

`L3 = 0.30 × faithfulness + 0.15 × completeness + 0.25 × coverage + 0.15 × coherence + 0.15 × actionability`.

The reported composite score is:

`Final = 0.35 × (100 × L1) + 0.30 × (100 × L2) + 0.35 × (20 × L3)`.

The primary Layer-2/Layer-3 judge is DeepSeek-V4-Flash. Validation runs additionally use GLM-4-Flash, GLM-5.3, and Kimi K2.6 as alternative judges as described in the manuscript.

## Validation and robustness studies

The repository contains supporting artifacts for the validation battery used alongside the primary campaign, including:

- out-of-coverage rejection testing;
- paraphrase matching and end-to-end paraphrase robustness;
- additional generation seeds;
- second-, third-, and fourth-judge checks;
- dense-retrieval control;
- Layer-1 scorer audit and repaired numeric scoring;
- Layer-2 claim-extraction robustness checks;
- synthetic report-degradation anchors;
- fluency/readability diagnostics;
- component ablations for Q, M, P, and T.

The manuscript also reports a targeted human validation on the same 30-instance fault-enriched subset used in the degradation study: 30 instances × 4 systems (the template-only ceiling, the traditional D2T pipeline, the tool-using agent, and the proposed framework — the contested comparisons) = 120 existing canonical reports, each rated by every evaluator (600 ratings). Five evaluators (two faculty members, two students, and one mechatronics engineer; none an author) rated the anonymized reports under blinded, per-instance randomized letters on faithfulness/completeness/coherence/actionability (1–5), 5W1H coverage, and a readability ranking. This validation reuses existing reports and does not alter the primary automatic AI4I scoring protocol. The executed design, evaluator instructions, raw rater exports, and reproduction commands are documented in `docs/AI4I_Human_Evaluation_Guidance_and_Tasks.md` and `runs/human_validation/`.

## Directory layout

```text
data/        processed AI4I tables, the 315-record sample, integrity and physics checks
models/      trained predictive models, metrics and test predictions
knowledge/   question bank, templates, ground truth, maps, KU inventory and RAG corpus
runs/        generated reports, layer outputs and robustness/validation run artifacts
analysis/    tables, statistical summaries, diagnostic reports and figures
docs/        experiment documentation
src/         experiment and evaluation code
```

## Current status

The primary B1–B8 campaign on `n=100`, the three-layer evaluation, statistical comparisons, Q/M/P/T ablations, cross-LLM study, efficiency analysis, and the validation studies listed above are complete. The 315-record sample remains the basis for knowledge-layer verification and the 63,945 programmatic ground-truth pairs; headline report-generation results are reported on the failure-enriched 100-record evaluation subset.

## LLM API configuration (`.env`)

All LLM-dependent scripts read provider URL, API key, and model name from a git-ignored `.env` file at the experiment root. Real environment variables override `.env` values.

Common generation defaults are:

- temperature: `0.3`
- max completion tokens: `512`

Judging and reasoning-model runs use the larger budgets documented in the manuscript and experiment scripts.
