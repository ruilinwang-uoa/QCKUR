# Cross-Domain Human Evaluation Guidance and Tasks

## Overview

You will receive several sets of generated analytical question–answer pairs. For each assigned item, determine whether the generated answer correctly answers the question. Use the workflow and error categories below, then complete the corresponding checklist according to the assigned group, model, and question number.

## Workflow

1. Review each assigned generated analytical answer together with its question and the corresponding analysis result.
2. Decide whether the answer is correct.
3. A correct answer should:
   - provide information supported by the analysis results;
   - fully answer the question, including any necessary caveat;
   - provide a textual explanation rather than only listing numbers.
4. If the answer is incorrect, select the most appropriate error category (A–E) below.
5. Record `Y` or `N` in the **Correct?** column. If you record `N`, also record the corresponding error category.

## Error Categories

| Category | Definition |
|---|---|
| **A** | **Hallucination or unsupported content.** The answer contradicts the analysis results or introduces factual content that is not supported by them. |
| **B** | **Incomplete answer or missing necessary caveat.** The answer omits information needed to answer the question fully or leaves out an important qualification. |
| **C** | **Numeric-only response or insufficient explanation.** The answer only lists values, or does not explain what the reported values mean in relation to the question. |
| **D** | **Incorrect interpretation.** The answer uses the analytical result incorrectly or draws an invalid conclusion from it. |
| **E** | **Other error.** The answer is incorrect for a reason not covered by Categories A–D. |

### Examples

- **A:** If the task is to identify the number with the largest absolute value from `-1, -3, 0, 2`, answering `2` is incorrect because `-3` has the largest absolute value.
- **B:** If the task is to identify the non-negative values from `-1, -3, 0, 2`, answering only `2` is incomplete because `0` is also non-negative.
- **C:** If the question asks how the dependent variable relates to the independent variables, simply reporting `R² = 0.7` is insufficient without explaining what that value indicates.
- **D:** If the question asks which independent variable has the greatest impact on the dependent variable, selecting a variable solely because it has the smallest p-value is not sufficient; the relevant effect measure must also be interpreted correctly.

## Checklist

### Group B — Binary Classification

#### RC

| Question number | Correct? (Y/N) | Incorrect reason (A–E) |
|---|---|---|
| 1-1 |  |  |
| 1-2 |  |  |
| 1-3 |  |  |
| 2-1 |  |  |
| 2-2 |  |  |
| 2-3 |  |  |
| 3-1 |  |  |
| 3-2 |  |  |
| 3-3 |  |  |
| 4-1 |  |  |
| 4-2 |  |  |
| 4-3 |  |  |
| 5-1 |  |  |
| 5-2 |  |  |
| 5-3 |  |  |
| 6-1 |  |  |
| 6-2 |  |  |
| 6-3 |  |  |
| 7-1 |  |  |
| 7-2 |  |  |
| 7-3 |  |  |
| 8-1 |  |  |
| 8-2 |  |  |
| 8-3 |  |  |
| 9-1 |  |  |
| 9-2 |  |  |
| 9-3 |  |  |

#### LDA

| Question number | Correct? (Y/N) | Incorrect reason (A–E) |
|---|---|---|
| 1-1 |  |  |
| 1-2 |  |  |
| 1-3 |  |  |
| 2-1 |  |  |
| 2-2 |  |  |
| 2-3 |  |  |
| 3-1 |  |  |
| 3-2 |  |  |
| 3-3 |  |  |
| 4-1 |  |  |
| 4-2 |  |  |
| 4-3 |  |  |
| 5-1 |  |  |
| 5-2 |  |  |
| 5-3 |  |  |
| 6-1 |  |  |
| 6-2 |  |  |
| 6-3 |  |  |
| 7-1 |  |  |
| 7-2 |  |  |
| 7-3 |  |  |
| 8-1 |  |  |
| 8-2 |  |  |
| 8-3 |  |  |
| 9-1 |  |  |
| 9-2 |  |  |
| 9-3 |  |  |

#### LOG

| Question number | Correct? (Y/N) | Incorrect reason (A–E) |
|---|---|---|
| 1-1 |  |  |
| 1-2 |  |  |
| 1-3 |  |  |
| 2-1 |  |  |
| 2-2 |  |  |
| 2-3 |  |  |
| 3-1 |  |  |
| 3-2 |  |  |
| 3-3 |  |  |
| 4-1 |  |  |
| 4-2 |  |  |
| 4-3 |  |  |
| 5-1 |  |  |
| 5-2 |  |  |
| 5-3 |  |  |
| 6-1 |  |  |
| 6-2 |  |  |
| 6-3 |  |  |
| 7-1 |  |  |
| 7-2 |  |  |
| 7-3 |  |  |
| 8-1 |  |  |
| 8-2 |  |  |
| 8-3 |  |  |
| 9-1 |  |  |
| 9-2 |  |  |
| 9-3 |  |  |

### Group M — Multi-Class Classification

#### RF

| Question number | Correct? (Y/N) | Incorrect reason (A–E) |
|---|---|---|
| 1-1 |  |  |
| 1-2 |  |  |
| 1-3 |  |  |
| 2-1 |  |  |
| 2-2 |  |  |
| 2-3 |  |  |
| 3-1 |  |  |
| 3-2 |  |  |
| 3-3 |  |  |
| 4-1 |  |  |
| 4-2 |  |  |
| 4-3 |  |  |
| 5-1 |  |  |
| 5-2 |  |  |
| 5-3 |  |  |
| 6-1 |  |  |
| 6-2 |  |  |
| 6-3 |  |  |

#### DT

| Question number | Correct? (Y/N) | Incorrect reason (A–E) |
|---|---|---|
| 1-1 |  |  |
| 1-2 |  |  |
| 1-3 |  |  |
| 2-1 |  |  |
| 2-2 |  |  |
| 2-3 |  |  |
| 3-1 |  |  |
| 3-2 |  |  |
| 3-3 |  |  |
| 4-1 |  |  |
| 4-2 |  |  |
| 4-3 |  |  |
| 5-1 |  |  |
| 5-2 |  |  |
| 5-3 |  |  |
| 6-1 |  |  |
| 6-2 |  |  |
| 6-3 |  |  |

#### LDA

| Question number | Correct? (Y/N) | Incorrect reason (A–E) |
|---|---|---|
| 1-1 |  |  |
| 1-2 |  |  |
| 1-3 |  |  |
| 2-1 |  |  |
| 2-2 |  |  |
| 2-3 |  |  |
| 3-1 |  |  |
| 3-2 |  |  |
| 3-3 |  |  |
| 4-1 |  |  |
| 4-2 |  |  |
| 4-3 |  |  |
| 5-1 |  |  |
| 5-2 |  |  |
| 5-3 |  |  |
| 6-1 |  |  |
| 6-2 |  |  |
| 6-3 |  |  |

### Group R — Regression

#### GB

| Question number | Correct? (Y/N) | Incorrect reason (A–E) |
|---|---|---|
| 1-1 |  |  |
| 1-2 |  |  |
| 1-3 |  |  |
| 2-1 |  |  |
| 2-2 |  |  |
| 2-3 |  |  |
| 3-1 |  |  |
| 3-2 |  |  |
| 3-3 |  |  |
| 4-1 |  |  |
| 4-2 |  |  |
| 4-3 |  |  |
| 5-1 |  |  |
| 5-2 |  |  |
| 5-3 |  |  |
| 6-1 |  |  |
| 6-2 |  |  |
| 6-3 |  |  |
| 7-1 |  |  |
| 7-2 |  |  |
| 7-3 |  |  |
| 8-1 |  |  |
| 8-2 |  |  |
| 8-3 |  |  |
| 9-1 |  |  |
| 9-2 |  |  |
| 9-3 |  |  |
| 10-1 |  |  |
| 10-2 |  |  |
| 10-3 |  |  |
| 11-1 |  |  |
| 11-2 |  |  |
| 11-3 |  |  |
| 12-1 |  |  |
| 12-2 |  |  |
| 12-3 |  |  |
| 13-1 |  |  |
| 13-2 |  |  |
| 13-3 |  |  |
| 14-1 |  |  |
| 14-2 |  |  |
| 14-3 |  |  |
| 15-1 |  |  |
| 15-2 |  |  |
| 15-3 |  |  |
| 16-1 |  |  |
| 16-2 |  |  |
| 16-3 |  |  |
| 17-1 |  |  |
| 17-2 |  |  |
| 17-3 |  |  |
| 18-1 |  |  |
| 18-2 |  |  |
| 18-3 |  |  |
| 19-1 |  |  |
| 19-2 |  |  |
| 19-3 |  |  |

#### RF

| Question number | Correct? (Y/N) | Incorrect reason (A–E) |
|---|---|---|
| 1-1 |  |  |
| 1-2 |  |  |
| 1-3 |  |  |
| 2-1 |  |  |
| 2-2 |  |  |
| 2-3 |  |  |
| 3-1 |  |  |
| 3-2 |  |  |
| 3-3 |  |  |
| 4-1 |  |  |
| 4-2 |  |  |
| 4-3 |  |  |
| 5-1 |  |  |
| 5-2 |  |  |
| 5-3 |  |  |
| 6-1 |  |  |
| 6-2 |  |  |
| 6-3 |  |  |
| 7-1 |  |  |
| 7-2 |  |  |
| 7-3 |  |  |
| 8-1 |  |  |
| 8-2 |  |  |
| 8-3 |  |  |
| 9-1 |  |  |
| 9-2 |  |  |
| 9-3 |  |  |
| 10-1 |  |  |
| 10-2 |  |  |
| 10-3 |  |  |
| 11-1 |  |  |
| 11-2 |  |  |
| 11-3 |  |  |
| 12-1 |  |  |
| 12-2 |  |  |
| 12-3 |  |  |
| 13-1 |  |  |
| 13-2 |  |  |
| 13-3 |  |  |
| 14-1 |  |  |
| 14-2 |  |  |
| 14-3 |  |  |
| 15-1 |  |  |
| 15-2 |  |  |
| 15-3 |  |  |
| 16-1 |  |  |
| 16-2 |  |  |
| 16-3 |  |  |
| 17-1 |  |  |
| 17-2 |  |  |
| 17-3 |  |  |
| 18-1 |  |  |
| 18-2 |  |  |
| 18-3 |  |  |
| 19-1 |  |  |
| 19-2 |  |  |
| 19-3 |  |  |

#### LIN

| Question number | Correct? (Y/N) | Incorrect reason (A–E) |
|---|---|---|
| 1-1 |  |  |
| 1-2 |  |  |
| 1-3 |  |  |
| 2-1 |  |  |
| 2-2 |  |  |
| 2-3 |  |  |
| 3-1 |  |  |
| 3-2 |  |  |
| 3-3 |  |  |
| 4-1 |  |  |
| 4-2 |  |  |
| 4-3 |  |  |
| 5-1 |  |  |
| 5-2 |  |  |
| 5-3 |  |  |
| 6-1 |  |  |
| 6-2 |  |  |
| 6-3 |  |  |
| 7-1 |  |  |
| 7-2 |  |  |
| 7-3 |  |  |
| 8-1 |  |  |
| 8-2 |  |  |
| 8-3 |  |  |
| 9-1 |  |  |
| 9-2 |  |  |
| 9-3 |  |  |
| 10-1 |  |  |
| 10-2 |  |  |
| 10-3 |  |  |
| 11-1 |  |  |
| 11-2 |  |  |
| 11-3 |  |  |
| 12-1 |  |  |
| 12-2 |  |  |
| 12-3 |  |  |
| 13-1 |  |  |
| 13-2 |  |  |
| 13-3 |  |  |
| 14-1 |  |  |
| 14-2 |  |  |
| 14-3 |  |  |
| 15-1 |  |  |
| 15-2 |  |  |
| 15-3 |  |  |
| 16-1 |  |  |
| 16-2 |  |  |
| 16-3 |  |  |
| 17-1 |  |  |
| 17-2 |  |  |
| 17-3 |  |  |
| 18-1 |  |  |
| 18-2 |  |  |
| 18-3 |  |  |
| 19-1 |  |  |
| 19-2 |  |  |
| 19-3 |  |  |

## Scope Check

- Binary classification: 9 datasets × 3 questions × 3 models = **81** question instances.
- Multi-class classification: 6 datasets × 3 questions × 3 models = **54** question instances.
- Regression: 19 datasets × 3 questions × 3 models = **171** question instances.
- Total: **306 unique question instances**.
