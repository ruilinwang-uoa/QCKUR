# AI4I first-batch results (n=30 instances/system, seed 42)
Layer-3 LLM-judge scores. Within each model column all systems are judged by the same judge (fair within-model comparison). L3 in 0-5; coverage in 0-6.
Each model judges the OTHER model's reports (cross-judge, avoids self-bias).

## glm-4-flash  (judge: deepseek-v4-flash)
| system          | id   |   n |   L3 |   faith |   compl |   coh |   act |   cov6 |
|:----------------|:-----|----:|-----:|--------:|--------:|------:|------:|-------:|
| Template-only   | B1   |  30 | 5    |    5    |    5    |  5    |  5    |   6    |
| Traditional D2T | B2   |  30 | 4.16 |    4.83 |    3.17 |  4.93 |  4.9  |   3.67 |
| Direct LLM      | B3   |  30 | 1.81 |    1.17 |    1.5  |  3.63 |  1.3  |   2.37 |
| Few-shot LLM    | B4   |  30 | 1.71 |    1.13 |    1.33 |  3.3  |  1.3  |   2.3  |
| Tool agent      | B5   |  30 | 2.8  |    2.3  |    2.67 |  3.83 |  2.6  |   3.6  |
| RAG             | B6   |  30 | 1.92 |    1.47 |    1.87 |  3.33 |  1.37 |   2.37 |
| RAG+CoVe        | B7   |  30 | 2.7  |    2.1  |    2.47 |  3.27 |  2.67 |   3.9  |
| Full framework  | B8   |  30 | 4.92 |    4.8  |    4.93 |  4.97 |  5    |   6    |

**B8 framework L3 = 4.92**. Advantage vs baselines:
- vs B3 (Direct LLM): **+3.11**
- vs B6 (RAG): **+3.00**
- vs B7 (RAG+CoVe): **+2.22**
- vs B4 (Few-shot LLM): **+3.21**
- vs B5 (Tool agent): **+2.12**
- vs B2 (Traditional D2T): **+0.76**

## deepseek-v4-flash  (judge: glm-4-flash)
| system          | id   |   n |   L3 |   faith |   compl |   coh |   act |   cov6 |
|:----------------|:-----|----:|-----:|--------:|--------:|------:|------:|-------:|
| Template-only   | B1   |  30 | 5    |    5    |    5    |  5    |  5    |   6    |
| Traditional D2T | B2   |  30 | 3.79 |    3    |    3.8  |  4.23 |  3.8  |   5.37 |
| Direct LLM      | B3   |  30 | 3.6  |    2.57 |    3.53 |  4.17 |  3.03 |   5.87 |
| Few-shot LLM    | B4   |  30 | 3.82 |    2.97 |    3.8  |  4.43 |  3.43 |   5.67 |
| Tool agent      | B5   |  30 | 3.87 |    2.87 |    4.03 |  4.33 |  3.47 |   5.93 |
| RAG             | B6   |  30 | 3.78 |    2.7  |    3.83 |  4.5  |  3.37 |   5.83 |
| RAG+CoVe        | B7   |  30 | 3.6  |    2.63 |    3.47 |  4.4  |  3.1  |   5.6  |
| Full framework  | B8   |  30 | 4.22 |    3.23 |    4.23 |  4.9  |  4.17 |   6    |

**B8 framework L3 = 4.22**. Advantage vs baselines:
- vs B3 (Direct LLM): **+0.62**
- vs B6 (RAG): **+0.44**
- vs B7 (RAG+CoVe): **+0.62**
- vs B4 (Few-shot LLM): **+0.40**
- vs B5 (Tool agent): **+0.35**
- vs B2 (Traditional D2T): **+0.43**

## Cross-LLM summary
| Model | B8 L3 | best non-framework | B8 advantage |
|---|---|---|---|
| glm-4-flash | 4.92 | 5.00 | +-0.08 |
| deepseek-v4-flash | 4.22 | 5.00 | +-0.78 |

Note: primary = glm-4-flash (non-reasoning); cross-LLM = deepseek-v4-flash (reasoning). Cross-model absolute L3 values use different judges and are not directly comparable; compare the within-model B8-vs-baseline *pattern*.
