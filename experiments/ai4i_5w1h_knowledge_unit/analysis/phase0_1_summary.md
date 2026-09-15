# AI4I 5W1H KU Experiment — Phase 0+1 Summary
> Phase 0+1 foundation complete; subsequent LLM-dependent experiments and validation studies have also been completed.
> Plan: `docs/AI4I_modified_experiment_plan_EN.md` v6.0 | Scale: 315 stratified samples.
## 1. Data integrity (Phase 0.2)
| Item | Value |
|---|---|
| Rows x Cols | 10,000 x 14 |
| Missing / Duplicates | 0 / 0 |
| Failures | 339 (3.39%), imbalance ~1:28 |
| Multi-mode rows | 24 (sum of modes 373) |

| Mode | Count | Rule |
|---|---|---|
| TWF | 46 | tool wear in [200,240) probabilistic / >=240 certain |
| HDF | 115 | (process-air) temp diff <8.6K AND rpm<1380 |
| PWF | 95 | power outside [3500,9000]W |
| OSF | 98 | tool_wear x torque > type limit |
| RNF | 19 | random (not rule-derivable) |

## 2. Physics-rule re-derivation (Phase 0.3)
Rules re-derived from variables only, compared to labels:

| Mode | n_true | Precision | Recall |
|---|---|---|---|
| TWF | 46 | 0.1429 | 0.0435 |
| HDF | 115 | 1.0 | 1.0 |
| PWF | 95 | 1.0 | 1.0 |
| OSF | 98 | 1.0 | 1.0 |

HDF/PWF/OSF reproduce labels exactly (P=R=1.0); TWF is probabilistic in [200,240) (790 rows, 43 labelled). Encoding test: **29 rows, 24 faulted, ALL PASSED**.

## 3. Model quality (Phase 0.4, mean +/- std over 5 seeds)
Primary metrics: **F1-macro** and **AUPRC** (accuracy omitted due to 3.4% imbalance).

| model               |   accuracy_mean |   accuracy_std |   f1_macro_mean |   f1_macro_std |   precision_macro_mean |   precision_macro_std |   recall_macro_mean |   recall_macro_std |   auprc_mean |   auprc_std |   roc_auc_mean |   roc_auc_std |   train_time_s_mean |   train_time_s_std |
|:--------------------|----------------:|---------------:|----------------:|---------------:|-----------------------:|----------------------:|--------------------:|-------------------:|-------------:|------------:|---------------:|--------------:|--------------------:|-------------------:|
| gradient_boosting   |          0.9852 |         0.0036 |          0.895  |         0.0225 |                 0.8738 |                0.0367 |              0.9205 |             0.0067 |       0.8853 |      0.0226 |         0.9787 |        0.0085 |              0.6322 |             0.0359 |
| logistic_regression |          0.8444 |         0.0097 |          0.5987 |         0.0063 |                 0.5826 |                0.0029 |              0.8759 |             0.0181 |       0.4739 |      0.0591 |         0.9409 |        0.0118 |              0.0236 |             0.0077 |
| random_forest       |          0.992  |         0.0021 |          0.9359 |         0.0152 |                 0.9648 |                0.0351 |              0.9126 |             0.0227 |       0.9081 |      0.0188 |         0.9823 |        0.0059 |              0.8538 |             0.011  |

Best mean F1-macro: **random_forest** (0.936).

## 4. Stratified 315-sample (seed 42)
Failure rate 3.49% | Modes: TWF=2, HDF=4, PWF=1, OSF=4, RNF=3

## 5. 5W1H Knowledge Units (Phase 1)
- Question bank: **203** questions (invalid dropped: 0)
| Category | Questions | M component |
|---|---|---|
| What | 61 | classifier/statistics |
| Why | 29 | physics causal engine |
| When | 26 | threshold comparator |
| Where | 25 | feature->component map |
| Who | 31 | fault->responder map |
| How | 31 | fault->action map |

- KU inventory: **203** executable units, 46 templates, 46 M handlers.
- Programmatic ground truth: **63,945** records (203 Q x 315 instances), empty/error: 0.

## 6. Layer 1 programmatic fact-checking
L1 = 0.30*tolerance + 0.30*rule + 0.40*classification.

| Setting | L1 | Tolerance | Rule | Classification |
|---|---|---|---|---|
| GT self-validation | 1.0 | 1.0 | 1.0 | 1.0 |
| Perturbed (injected errors) | 0.1189 | - | - | - |

Self-consistent GT scores L1=1.0; 18581/25200 injected errors are caught (L1 drops to 0.1189), confirming the scorer discriminates. The same scorer applies to LLM reports in Phase 3.

## 7. Status

Phase 0+1, B1–B8 generation, three-layer evaluation, component ablations,
cross-LLM evaluation, and the reported robustness and validation studies have
been completed.

The 315-sample, 203-KU bank, GT, and Layer 1 scorer are the required inputs for
all of the above.
