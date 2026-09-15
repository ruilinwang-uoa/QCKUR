# Synthetic-anchor degradation results (protocol validity)

## B1: layer means by degradation level (n=30)

```
          L1  classification  numerical     L2     L3  faithfulness   cov6   Final
level                                                                             
D0     0.969           0.933      0.987  0.958  4.762         4.733  5.667  96.014
D1     0.950           0.933      0.989  0.867  4.042         4.733  3.733  87.557
D2     0.901           0.933      0.759  0.871  4.179         3.667  5.633  86.936
D3     0.640           0.110      0.987  0.721  2.672         1.200  4.833  62.741
```

Spearman(level, score) per layer, B1:
```
         layer    rho       p
            L1 -0.670 5.6e-17
classification -0.656 4.0e-16
     numerical -0.119 2.0e-01
            L2 -0.603 3.1e-13
            L3 -0.621 3.7e-14
  faithfulness -0.699 7.3e-19
          cov6 -0.068 4.6e-01
         Final -0.620 4.2e-14
```

Paired Wilcoxon Dk vs D0 (Final):
  D1: drop +8.5 (p=1.9e-09)
  D2: drop +9.1 (p=3.3e-03)
  D3: drop +33.3 (p=9.3e-09)

## B8: layer means by degradation level (n=30)

```
          L1  classification  numerical     L2     L3  faithfulness   cov6   Final
level                                                                             
D0     0.723           0.933      0.167  0.937  4.609         4.367  5.900  85.694
D1     0.717           0.933      0.178  0.850  3.664         4.267  3.767  76.253
D2     0.697           0.933      0.080  0.784  3.638         2.533  5.400  73.402
D3     0.394           0.110      0.167  0.720  2.287         1.133  3.933  51.400
```

Spearman(level, score) per layer, B8:
```
         layer    rho       p
            L1 -0.598 5.4e-13
classification -0.656 4.0e-16
     numerical -0.030 7.5e-01
            L2 -0.673 3.9e-17
            L3 -0.687 4.6e-18
  faithfulness -0.753 3.5e-23
          cov6 -0.334 1.9e-04
         Final -0.740 4.8e-22
```

Paired Wilcoxon Dk vs D0 (Final):
  D1: drop +9.4 (p=1.9e-09)
  D2: drop +12.3 (p=1.9e-09)
  D3: drop +34.3 (p=9.3e-09)

## Judge-coverage anchor (D1 deleted two 5W1H sections)

B1: judge coverage 5.67/6 (D0) -> 3.73/6 (D1) (expected drop ~2)
B8: judge coverage 5.90/6 (D0) -> 3.77/6 (D1) (expected drop ~2)
