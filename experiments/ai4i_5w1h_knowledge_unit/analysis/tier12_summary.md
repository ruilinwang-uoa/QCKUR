# Tier 1 + Tier 2 results

## a. Two-judge agreement (DeepSeek primary vs GLM second judge, n=800)

```
          dim  pearson_r        p  spearman_rho  mean_ds  mean_glm
           L3      0.770 3.7e-158         0.770    2.951     3.463
 faithfulness      0.720 1.6e-128         0.794    2.758     2.618
 completeness      0.796 2.0e-176         0.810    2.674     3.424
    coherence      0.560  2.9e-67         0.642    3.938     4.032
actionability      0.834 1.7e-208         0.843    2.806     3.030
```

System-mean rank correlation (Spearman): 0.976

```
        L3_deepseek  L3_glm
method                     
B1            4.899   4.888
B2            3.958   3.623
B3            1.742   2.890
B4            1.394   2.693
B5            3.361   3.512
B6            1.790   3.052
B7            1.654   3.042
B8            4.807   4.006
```

B8 vs baselines under the GLM judge (paired, n=100 each):

```
baseline  d_L3_glm wilcoxon_p_glm
      B2      0.38        8.6e-07
      B5      0.49        8.6e-11
      B3      1.12        1.6e-14
      B4      1.31        7.3e-16
      B6      0.95        5.6e-15
      B7      0.96        1.9e-14
```

## b. Out-of-coverage rejection test (matching mechanism)

in-coverage n=36, OOC n=40, provider=openai, seed=42

- positive matching accuracy (exact text): **72.2%**
- wrong bank question: 13.9%; wrongly rejected: 0.0%; invalid output: 13.9%
- OOC rejection rate: **97.5%**
- OOC wrongly accepted: 2.5%; invalid output: 0.0%

## c. Dense-retrieval RAG (B6-dense, all-MiniLM-L6-v2) vs TF-IDF B6

n=100

- B6 TF-IDF  : L1=0.457 L2=0.362 L3=1.790 Final=39.4
- B6 dense   : L1=0.407 L2=0.348 L3=1.699 Final=36.6
- B8 (v3)    : Final=83.9

paired B8 vs B6-dense: dFinal=+47.3, Wilcoxon p=6.0e-18

## d. B8 multi-seed stability (n=100 instances per seed)

```
 seed  B8 Final mean   sd
   42           83.9 12.8
    1           87.5  9.7
    7           87.6  9.9
```

across-seed mean = 86.3, sd = 2.12

seed 42: B8 83.9 vs B2 83.3 (d=+0.6, Wilcoxon p=5.9e-03)
seed 1: B8 87.5 vs B2 83.3 (d=+4.3, Wilcoxon p=3.9e-10)
seed 7: B8 87.6 vs B2 83.3 (d=+4.3, Wilcoxon p=1.1e-09)
