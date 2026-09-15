# De-circularization analyses (Tier 0)

## A. B8 numeric-mismatch audit

```json
{
  "orig_hit": 44,
  "derived": 1,
  "dataset_stat": 0,
  "unit": 2,
  "identifier_or_marker": 569,
  "genuine": 0
}
```

Recovery examples: derived-feature: 7.8 (Overstrain=7577, dT=7.8)

Remaining genuine-miss values: 

## A2. Numeric scorer original vs repaired

```
sys            name  num_orig  num_rep
 B1   Template-only     0.996    1.000
 B2 Traditional D2T     1.000    1.000
 B3      Direct LLM     0.248    0.997
 B4        Few-shot     0.662    1.000
 B5      Tool agent     0.145    0.998
 B6             RAG     0.386    1.000
 B7        RAG+CoVe     0.340    0.977
 B8       Framework     0.089    1.000
```

## B. Final variants

```
           L1  L1_rep     L2     L3  L3_nocov  Final_orig  Final_L1rep  Final_L3nocov  Final_both
method                                                                                           
B1      0.987   0.988  0.981  4.899     4.890        98.3         98.3           98.2        98.2
B2      0.988   0.988  0.700  3.958     4.422        83.3         83.3           86.5        86.5
B3      0.392   0.617  0.287  1.742     1.612        34.5         42.4           33.6        41.5
B4      0.515   0.616  0.287  1.394     1.462        36.4         39.9           36.9        40.4
B5      0.580   0.836  0.580  3.361     3.304        61.3         70.2           60.9        69.8
B6      0.457   0.641  0.362  1.790     1.754        39.4         45.8           39.1        45.6
B7      0.433   0.624  0.326  1.654     1.678        36.5         43.2           36.7        43.4
B8      0.715   0.988  0.841  4.807     4.770        83.9         93.5           83.6        93.2
```

## C. Weight grid

```
 w_L1  w_L2  w_L3   B8   B2  B8-B2
 0.35  0.30  0.35 93.2 86.5    6.7
 0.33  0.33  0.33 92.8 85.8    7.0
 0.50  0.25  0.25 94.3 89.0    5.3
 0.25  0.50  0.25 90.6 81.8    8.8
 0.25  0.25  0.50 93.4 86.4    7.0
 0.60  0.20  0.20 95.2 91.0    4.2
```
