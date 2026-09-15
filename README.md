# Supplementary Materials for QCKUR Predictive-Maintenance Reporting

This repository provides supplementary materials for the manuscript:

**Question-Centered Knowledge Units for Bounding LLM Hallucination in Predictive Maintenance Reporting**

The repository is organized for review and reproducibility. It contains:

1. the public-facing QCKUR prototype components;
2. the **AI4I 5W1H Knowledge-Unit Experiment**;
3. the **CWRU Bearing Knowledge-Unit Experiment**;
4. the **Cross-domain Benchmark Experiment**; and
5. dataset-source and prototype-usage documentation.

Raw datasets are not redistributed in this repository. Dataset acquisition information is provided in the experiment-specific documentation and, where applicable, in [`doc/DATASETS.md`](doc/DATASETS.md).

## Repository Overview

| Component | Location | Purpose |
|---|---|---|
| Public prototype components | repository root | General implementation of the question-centered reporting prototype |
| Prototype usage guide | `doc/PROTOTYPE_USAGE.md` | Installation instructions and UI workflow for the public prototype |
| AI4I 5W1H Knowledge-Unit experiment | `experiments/ai4i_5w1h_knowledge_unit/` | Main predictive-maintenance experiment on the AI4I 2020 dataset |
| CWRU Bearing Knowledge-Unit experiment | `experiments/cwru_bearing_knowledge_unit/` | Second predictive-maintenance instantiation on measured bearing-vibration data |
| Cross-domain benchmark | `experiments/cross_domain_benchmark/` | Human-judged benchmark across regression, binary-classification, and multiclass-classification tasks |
| Dataset documentation | `doc/DATASETS.md` | Dataset-source and filename information for the documented benchmark datasets |

## Quick Navigation for Reviewers

* AI4I experiment: [`experiments/ai4i_5w1h_knowledge_unit/`](experiments/ai4i_5w1h_knowledge_unit/)
* CWRU bearing experiment: [`experiments/cwru_bearing_knowledge_unit/`](experiments/cwru_bearing_knowledge_unit/)
* Cross-domain benchmark: [`experiments/cross_domain_benchmark/`](experiments/cross_domain_benchmark/)
* Dataset documentation: [`doc/DATASETS.md`](doc/DATASETS.md)
* Public prototype usage: [`doc/PROTOTYPE_USAGE.md`](doc/PROTOTYPE_USAGE.md)

Experiment-specific setup, execution commands, and artifact descriptions are provided in the `README.md` file inside each experiment folder.

## Experiment 1: AI4I 5W1H Knowledge-Unit Experiment

The AI4I experiment instantiates the proposed question-centered knowledge-based framework for predictive-maintenance reporting using the AI4I 2020 Predictive Maintenance Dataset. The reporting task is organized through 203 question-centered knowledge units distributed across the six 5W1H categories and linked to deterministic model handlers, answer templates, and encoded physical failure rules.

The repository contains:

* the 203-unit 5W1H knowledge library;
* deterministic handlers, templates, encoded physical rules, and programmatic ground truth;
* implementations of the eight compared systems: template-only, traditional D2T, direct LLM, few-shot LLM, tool-using agent, RAG, RAG with Chain-of-Verification, and the full QCKUR framework;
* the three-layer automatic evaluation pipeline;
* component-ablation, cross-LLM, multi-seed, paraphrase, scorer-audit, degradation, and judge-replication artifacts used in the reported analyses.

The primary report-generation and scoring campaign uses a failure-enriched sample of 100 AI4I instances. The larger stratified 315-record sample is used for knowledge execution and programmatic ground-truth construction.

Detailed instructions are available in:

```text
experiments/ai4i_5w1h_knowledge_unit/README.md
```

## Experiment 2: CWRU Bearing Knowledge-Unit Experiment

The CWRU experiment provides a second predictive-maintenance instantiation using measured bearing-vibration data from the Case Western Reserve University Bearing Data Center.

The main benchmark contains 40 records: four normal baselines and 36 single-point bearing faults spanning inner-race, ball, and load-zone-centered outer-race conditions, three defect diameters, and motor loads 0–3 hp. A separate set of 16 outer-race files at unseen clock positions is used as a robustness probe.

The experiment uses a two-tier diagnostic handler:

* a rule tier based on characteristic-frequency evidence from Hilbert-envelope spectra; and
* a random-forest model tier over deterministic file-level spectral features when no rule fires.

The reporting comparison uses the applicable systems B1, B2, B3, B4, B5, and B8 under the three-layer evaluation protocol. The repository includes the CWRU manifest, bearing-specific knowledge files, generation and evaluation scripts, saved report outputs, layer-level evaluation outputs, and the aggregate result table.

Detailed instructions are available in:

```text
experiments/cwru_bearing_knowledge_unit/README.md
experiments/cwru_bearing_knowledge_unit/docs/CWRU_EXPERIMENT_PROTOCOL.md
```

## Experiment 3: Cross-domain Benchmark

The cross-domain benchmark evaluates whether the proposed framework generalizes beyond the predictive-maintenance instantiations. It covers 34 public datasets across regression, binary classification, and multiclass classification tasks, comprising 306 unique question instances. Each question was generated five times per method–LLM condition, yielding 1,530 answers per condition.

The benchmark compares:

* the QCKUR prototype;
* Direct LLM generation; and
* a LangChain single-agent baseline.

The benchmark evaluates answer error rate, error types, token usage, and runtime under matched LLM settings. Its outputs are assessed under a blinded human-evaluation protocol.

Detailed instructions, configurations, and notebooks are available in:

```text
experiments/cross_domain_benchmark/README.md
```

## Public Prototype

The repository root contains the public-facing prototype components for question-centered data-to-text reporting. The prototype supports model recommendation, model fitting, template-guided answer construction, and LLM-based linguistic refinement using API-based LLMs or local Ollama models.

The prototype guide is available at:

```text
doc/PROTOTYPE_USAGE.md
```

## Data Availability

Raw datasets are not redistributed in this repository.

Users should obtain datasets from their original providers and place them in the locations specified by the corresponding experiment documentation. Users are responsible for complying with the licenses and terms of use of the original dataset providers.

## Reproducibility Notes

Some experiments require LLM API calls. Exact generated text may vary with provider-side model versions, API endpoints, decoding behavior, and service conditions. Saved configurations, scripts, generated outputs, evaluation outputs, and summary analyses are included where they are present in the repository.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── LLMcomponents.py
├── LocalLLMcomponents.py
├── NLGcomponents.py
├── datasciencecomponents.py
├── defaultUI.py
├── defaultUIforLocal.py
├── stUIforAPI.py
├── stUIforLocal.py
├── apptemplates/
├── doc/
│   ├── DATASETS.md
│   └── PROTOTYPE_USAGE.md
├── experiments/
│   ├── ai4i_5w1h_knowledge_unit/
│   ├── cwru_bearing_knowledge_unit/
│   └── cross_domain_benchmark/
└── readme/
```
