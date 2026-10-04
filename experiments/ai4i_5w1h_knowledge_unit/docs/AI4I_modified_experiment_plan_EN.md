# AI4I Dataset Experiment Plan and Implemented Protocol

> **Paper:** *Question-Centered Knowledge Units for Bounding LLM Hallucination in Predictive Maintenance Reporting*
>
> **Provenance.** The primary automatic AI4I campaign was designed from internal experiment plan v6.0 dated 2026-07-23, one day before the first LLM-dependent run on 2026-07-24. The Layer-1 sub-weights, Layer-3 rubric, and composite weights were fixed before that run. This public English document aligns the terminology with the implemented study and records subsequent validation extensions separately; those later validation additions did not change the already generated primary reports or the predefined scoring weights.

---

## 1. Objective and scope

The AI4I study evaluates a question-centered knowledge-based reporting framework in which each knowledge unit has four explicit components:

`KU = {Q, M, P, T}`

- **Q — question:** a predefined analytical question organized under What / Why / When / Where / Who / How;
- **M — model handler:** the deterministic analytical procedure that computes the required evidence;
- **P — physical rules:** domain rules bound to the unit when relevant;
- **T — answer template:** the deterministic structure that converts verified evidence into an intermediate answer.

The LLM is restricted to semantic question matching and linguistic refinement. Analytical facts are established before free-form realization.

The AI4I experiment has three purposes:

1. evaluate report reliability and quality against seven comparison systems;
2. identify the contribution of Q, M, P, and T through component ablation;
3. test robustness to model, judge, seed, retrieval, paraphrase, and evaluation-protocol choices.

The primary AI4I score remains automatic. A targeted human study was added subsequently as an independent validation of already generated canonical reports.

---

## 2. Dataset and sampling

The AI4I 2020 Predictive Maintenance Dataset contains 10,000 records and a machine-failure prevalence of approximately 3.4%.

A stratified 315-record sample is drawn with seed 42 for knowledge-layer construction and programmatic verification. The instantiated knowledge library contains 203 executable units:

| 5W1H category | Units |
|---|---:|
| What | 61 |
| Why | 29 |
| When | 26 |
| Where | 25 |
| Who | 31 |
| How | 31 |
| **Total** | **203** |

The implementation contains 46 answer templates and 46 deterministic model handlers. Programmatic ground truth is generated for all `203 × 315 = 63,945` knowledge-unit/record pairs.

Report generation and headline scoring use a failure-enriched stratified subset of `n = 100` records (seed 42) containing all 11 faulted records in the 315-record sample. This increases the number of fault-relevant cases available for evaluation while preserving a large healthy stratum.

Predictive models are trained under five seeds: `42, 1, 7, 2024, 123`.

---

## 3. Domain knowledge and physical rules

The implementation encodes the AI4I failure mechanisms used by the reporting system. In particular:

- **Power failure (PWF):** mechanical power is computed from torque and rotational speed; the safe envelope is 3,500–9,000 W.
- **Heat dissipation failure (HDF):** triggered when process temperature minus air temperature is below 8.6 K and rotational speed is below 1,380 rpm.
- **Overstrain failure (OSF):** tool-wear × torque is compared with the product-type threshold of 11,000 / 12,000 / 13,000 min·Nm for L / M / H products.
- **Tool-wear failure (TWF):** treated as probabilistic within the dataset's wear band rather than as a deterministic physical rule.
- **Random failure (RNF):** has no deterministic physical rule.

These rules are represented explicitly in the knowledge layer rather than inferred by the generation model.

---

## 4. Compared systems

Eight systems are evaluated under matched input conditions.

| ID | System | Main distinction |
|---|---|---|
| B1 | Template-only | framework intermediate answer returned without LLM refinement |
| B2 | Traditional D2T | hand-crafted content rules and fixed templates |
| B3 | Direct LLM | end-to-end generation without explicit knowledge units |
| B4 | Few-shot LLM | direct generation with three fixed exemplars |
| B5 | Tool-using agent | function-calling analytical tools; structure remains implicit |
| B6 | RAG | TF-IDF retrieval over a domain corpus, `k=5`, excluding framework templates |
| B7 | RAG + CoVe | RAG followed by chain-of-verification |
| B8 | Proposed framework | explicit Q/M/P/T knowledge units + bounded LLM refinement |

For the primary AI4I campaign, LLM-based systems use GLM-4-Flash. A cross-LLM campaign repeats the comparison with DeepSeek-V4-Flash. The same dataset split, fitted model context, background information, and question are supplied to the compared systems.

Primary generation settings are temperature `0.3` and max completion tokens `512`. DeepSeek-V4-Flash generation uses a larger output budget because its reasoning tokens share the completion budget.

---

## 5. Three-layer evaluation protocol

### 5.1 Layer 1 — programmatic fact checking

Layer 1 evaluates machine-checkable factual content against programmatic ground truth:

`L1 = 0.30 × S_num + 0.30 × S_rule + 0.40 × S_cls`

where:

- `S_num` is numeric-tolerance agreement;
- `S_rule` is physical-rule consistency;
- `S_cls` is fault-mode classification agreement.

A perturbation test is used to verify that the scorer responds to fabricated or corrupted factual content. A later scorer audit identifies formatting artifacts in the original numeric matcher; repaired numeric scoring is therefore reported as a secondary post-hoc analysis rather than replacing the original primary score.

### 5.2 Layer 2 — report-level consistency

The judge extracts factual claims, compares them with verified analytical facts, records 5W1H coverage, and identifies contradictions:

`L2 = 0.50 × C_cons + 0.25 × C_cov + 0.25 × (1 − C_contra)`

where `C_cons`, `C_cov`, and `C_contra` are the consistent-claim fraction, six-aspect coverage fraction, and contradiction fraction, respectively.

### 5.3 Layer 3 — multidimensional report quality

Reports are rated on five predefined dimensions:

| Dimension | Weight |
|---|---:|
| Faithfulness | 0.30 |
| Completeness | 0.15 |
| 5W1H coverage | 0.25 |
| Coherence | 0.15 |
| Actionability | 0.15 |

The Layer-3 score is:

`L3 = 0.30 × faithfulness + 0.15 × completeness + 0.25 × coverage + 0.15 × coherence + 0.15 × actionability`

### 5.4 Composite score

The predefined aggregate is:

`Final = 0.35 × (100 × L1) + 0.30 × (100 × L2) + 0.35 × (20 × L3)`

The primary Layer-2/Layer-3 judge is DeepSeek-V4-Flash.

---

## 6. Statistical analysis

Because the same instances are scored under each system, the design-matched omnibus analysis uses a Friedman test. The originally planned Kruskal–Wallis statistic is retained only for transparency because its independence assumption does not match the paired design.

Paired Wilcoxon signed-rank tests are the authoritative system-to-system comparisons. Cliff's delta is used as an effect-size measure where reported. Rank agreement between alternative judges over the eight systems uses Spearman correlation with exact two-sided permutation p-values.

No correction for multiple pairwise comparisons is applied; near-threshold results are interpreted accordingly.

---

## 7. Component ablation

The implemented ablation study removes one framework component at a time from B8 while keeping the remaining configuration fixed:

| Configuration | Removed component | Purpose |
|---|---|---|
| `−T` | answer templates | contribution of deterministic answer structuring |
| `−Q` | question matching | contribution of routing to the correct knowledge unit |
| `−P` | encoded physical rules | contribution of domain-rule grounding |
| `−M` | model handlers | contribution of computed analytical evidence |

All ablations are evaluated on the same 100 AI4I instances as the primary campaign.

The reported final study does **not** use a 5W1H-vs-non-5W1H taxonomy experiment or six separate category-removal ablations as headline experiments; the component ablation above is the implemented RQ5 analysis.

---

## 8. Robustness and validation battery

The primary automatic evaluation is supplemented by the following checks:

- out-of-coverage rejection test;
- paraphrase matching test and end-to-end paraphrase run;
- additional generation seeds for B8 and the five LLM-based baselines;
- cross-LLM generation with DeepSeek-V4-Flash;
- alternative-judge checks with GLM-4-Flash, GLM-5.3, and Kimi K2.6;
- dense-retrieval control for the RAG route;
- Layer-1 perturbation test and numeric-scorer audit;
- Layer-2 claim-extraction robustness re-judging;
- synthetic report-degradation anchors;
- coverage sensitivity and coverage-free Layer-3 recomputation;
- fluency/readability diagnostics.

These studies bound the sensitivity of the conclusions to model family, judging behavior, decoding seed, retrieval implementation, and evaluation artifacts.

---

## 9. Targeted human validation

The primary 100-instance AI4I scoring remains automatic. An additional human study evaluates a fixed 30-instance fault-enriched subset already used in the degradation analysis (11 faulted, 19 healthy).

No new reports are generated for this study. The eight systems contribute `30 × 8 = 240` existing canonical reports. Each report receives three independent ratings, producing 720 report ratings in total. Fifteen evaluators with quantitative backgrounds participate under blinded and randomized conditions.

Evaluators assess:

- overall correctness;
- 5W1H coverage;
- faithfulness;
- completeness;
- coherence;
- actionability;
- fluency/readability as an auxiliary human-only 1–5 measure.

Incorrect reports receive one primary A–E error category under the same error taxonomy used in the cross-domain benchmark. Overall correctness is summarized by majority vote; 1–5 dimensions are averaged across raters; Fleiss's kappa measures agreement for correctness and primary error type; and system-level agreement with automatic Layer-3 rankings is summarized using Spearman's rho.

This human validation is an independent check on the already generated AI4I outputs and does not modify the primary automatic protocol or its predefined weights.

---

## 10. LLM and judging configuration

The implemented study uses the following roles:

- **GLM-4-Flash:** primary AI4I generation backend;
- **DeepSeek-V4-Flash:** primary Layer-2/Layer-3 judge and cross-LLM generation backend;
- **GLM-4-Flash:** second-judge Layer-3 re-evaluation;
- **GLM-5.3:** third judge for the DeepSeek-generated cross-LLM reports;
- **Kimi K2.6:** family-disjoint fourth judge for the primary GLM-4-Flash reports.

Primary AI4I judging uses a maximum of 1,500 completion tokens. GLM-5.3 uses 3,000 for its third-judge pass; Kimi judging uses 1,500 in non-thinking mode. Model endpoints and exact runtime settings are supplied through the repository configuration files.

---

## 11. Research questions mapped to the implemented experiments

| RQ | Implemented evidence |
|---|---|
| **RQ1 — Matching** | verbatim matching, paraphrase matching, and out-of-coverage rejection |
| **RQ2 — Accuracy / quality** | B1–B8 primary comparison, three-layer protocol, targeted human validation, second CWRU instantiation |
| **RQ3 — Generality** | cross-domain benchmark, cross-LLM generation, measured CWRU instantiation |
| **RQ4 — Efficiency and engineering cost** | token use, runtime, LLM-call count, knowledge-engineering effort |
| **RQ5 — Components** | one-at-a-time Q/M/P/T ablation |

---

## 12. Relationship between the original planning scope and the implemented study

The initial planning document was broader than the final reported AI4I experiment. The table below records the main implementation changes so that the public protocol is not mistaken for an unmodified pre-run checklist.

| Initial planning item | Implemented study |
|---|---|
| KU represented as Q/T/M | Final formalization is Q/M/P/T, with physical rules represented explicitly as P |
| 315 records proposed for the main report campaign | 315 records retained for knowledge-layer verification and 63,945 GT pairs; headline report generation/scoring uses a failure-enriched `n=100` subset |
| GPT-4-oriented baseline wording | Primary AI4I backend is GLM-4-Flash; cross-LLM backend is DeepSeek-V4-Flash |
| FAISS / embedding retrieval proposed | Primary RAG uses TF-IDF with `k=5`; dense retrieval is a control study |
| Fully automatic evaluation only | Primary AI4I scoring remains fully automatic; a later targeted human study validates 240 existing reports without changing the primary outputs or weights |
| 5W1H-vs-non-5W1H and six category-removal studies proposed | Not used as headline experiments in the final manuscript; RQ5 is evaluated with Q/M/P/T component ablation |
| Kruskal–Wallis + post-hoc comparisons proposed | Paired design is analyzed with Friedman omnibus and paired Wilcoxon tests; the originally planned Kruskal–Wallis statistic is retained for transparency |

---

## 13. Repository outputs

The experiment directory contains:

- processed AI4I data and the 315-record sample;
- trained model artifacts and predictive metrics;
- the 203-unit question bank, templates, physical-rule maps, handlers, and programmatic ground truth;
- B1–B8 report outputs for the primary and robustness campaigns;
- Layer-1, Layer-2, and Layer-3 evaluation outputs;
- statistical analyses, ablation outputs, judge-replication outputs, paraphrase and multi-seed results;
- scorer-audit, degradation-anchor, coverage, and fluency analyses.

The repository README gives the execution-oriented view of these files, while this document records the experimental design and its implemented scope.
