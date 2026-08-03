# capacity_sweep (5 seeds)

| model | acc mean | std | min | max | n | n_mem |
|---|---|---|---|---|---|---|
| iqc_fixed_K23 | 0.8390 | 0.0075 | 0.8306 | 0.8484 | 5 | 46.0 |
| iqc_fixed_K19 | 0.8388 | 0.0032 | 0.8358 | 0.8428 | 5 | 38.0 |
| iqc_fixed_K17 | 0.8379 | 0.0066 | 0.8315 | 0.8461 | 5 | 34.0 |
| iqc_fixed_K11 | 0.8346 | 0.0053 | 0.8299 | 0.8414 | 5 | 22.0 |
| iqc_fixed_K13 | 0.8332 | 0.0046 | 0.8280 | 0.8389 | 5 | 26.0 |
| knn | 0.8243 | 0.0038 | 0.8181 | 0.8279 | 5 |  |
| iqc_fixed_K7 | 0.8180 | 0.0086 | 0.8089 | 0.8286 | 5 | 14.0 |
| iqc_fixed_K5 | 0.8147 | 0.0066 | 0.8045 | 0.8221 | 5 | 10.0 |
| iqc_fixed_K3 | 0.8062 | 0.0096 | 0.7924 | 0.8161 | 5 | 6.0 |
| iqc_fixed_K1 | 0.8020 | 0.0281 | 0.7525 | 0.8206 | 5 | 2.0 |
| iqc_fixed_K2 | 0.8018 | 0.0108 | 0.7862 | 0.8121 | 5 | 4.0 |

## McNemar median p-values (paired, per-seed)

- iqc_fixed_K19 vs iqc_fixed_K2: p=6.26e-133
- iqc_fixed_K17 vs iqc_fixed_K2: p=7.03e-133
- iqc_fixed_K11 vs iqc_fixed_K2: p=1.967e-129
- iqc_fixed_K2 vs iqc_fixed_K23: p=4.376e-126
- iqc_fixed_K13 vs iqc_fixed_K2: p=1.345e-122
- iqc_fixed_K17 vs iqc_fixed_K3: p=5.664e-121
- iqc_fixed_K19 vs iqc_fixed_K3: p=2.602e-116
- iqc_fixed_K11 vs iqc_fixed_K3: p=1.563e-114
- iqc_fixed_K13 vs iqc_fixed_K3: p=2.63e-111
- iqc_fixed_K23 vs iqc_fixed_K3: p=6.965e-98
- iqc_fixed_K1 vs iqc_fixed_K17: p=2.399e-79
- iqc_fixed_K1 vs iqc_fixed_K23: p=2.606e-77
- iqc_fixed_K11 vs iqc_fixed_K5: p=9.341e-74
- iqc_fixed_K11 vs iqc_fixed_K7: p=1.277e-68
- iqc_fixed_K19 vs iqc_fixed_K5: p=6.197e-66
- iqc_fixed_K19 vs iqc_fixed_K7: p=7.962e-63
- iqc_fixed_K23 vs iqc_fixed_K5: p=1.709e-62
- iqc_fixed_K13 vs iqc_fixed_K7: p=2.7e-62
- iqc_fixed_K17 vs iqc_fixed_K5: p=1.193e-59
- iqc_fixed_K1 vs iqc_fixed_K19: p=9.882e-59
- iqc_fixed_K1 vs iqc_fixed_K2: p=3.004e-58
- iqc_fixed_K13 vs iqc_fixed_K5: p=7.068e-56
- iqc_fixed_K17 vs iqc_fixed_K7: p=1.205e-55
- iqc_fixed_K2 vs knn: p=2.967e-52
- iqc_fixed_K1 vs iqc_fixed_K11: p=3.492e-50
- iqc_fixed_K23 vs iqc_fixed_K7: p=4.112e-50
- iqc_fixed_K3 vs knn: p=5.668e-49
- iqc_fixed_K2 vs iqc_fixed_K5: p=3.81e-46
- iqc_fixed_K1 vs iqc_fixed_K13: p=3.635e-45
- iqc_fixed_K19 vs knn: p=3.571e-40
- iqc_fixed_K3 vs iqc_fixed_K7: p=2.808e-35
- iqc_fixed_K2 vs iqc_fixed_K7: p=5.984e-32
- iqc_fixed_K17 vs knn: p=1.034e-30
- iqc_fixed_K1 vs knn: p=6.492e-27
- iqc_fixed_K11 vs knn: p=1.645e-25
- iqc_fixed_K5 vs iqc_fixed_K7: p=1.899e-25
- iqc_fixed_K13 vs knn: p=7.577e-25
- iqc_fixed_K23 vs knn: p=2.465e-21
- iqc_fixed_K1 vs iqc_fixed_K3: p=5.341e-20
- iqc_fixed_K3 vs iqc_fixed_K5: p=2.791e-16
- iqc_fixed_K17 vs iqc_fixed_K19: p=3.899e-13
- iqc_fixed_K1 vs iqc_fixed_K7: p=1.499e-11
- iqc_fixed_K1 vs iqc_fixed_K5: p=1.458e-10
- iqc_fixed_K13 vs iqc_fixed_K23: p=3.623e-10
- iqc_fixed_K13 vs iqc_fixed_K17: p=2.007e-09
- iqc_fixed_K5 vs knn: p=3.386e-08
- iqc_fixed_K13 vs iqc_fixed_K19: p=8.946e-08
- iqc_fixed_K11 vs iqc_fixed_K19: p=5.499e-07
- iqc_fixed_K2 vs iqc_fixed_K3: p=4.824e-06
- iqc_fixed_K7 vs knn: p=2.042e-05
- iqc_fixed_K11 vs iqc_fixed_K23: p=0.0001045
- iqc_fixed_K19 vs iqc_fixed_K23: p=0.0001252
- iqc_fixed_K17 vs iqc_fixed_K23: p=0.002314
- iqc_fixed_K11 vs iqc_fixed_K17: p=0.01859
- iqc_fixed_K11 vs iqc_fixed_K13: p=0.0215
