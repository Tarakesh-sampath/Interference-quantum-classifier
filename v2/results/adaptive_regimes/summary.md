# adaptive_regimes (5 seeds)

| model | acc mean | std | min | max | n | n_mem |
|---|---|---|---|---|---|---|
| iqc_regime2 | 0.8172 | 0.0291 | 0.7769 | 0.8435 | 5 | 1.0 |
| iqc_adaptive | 0.8143 | 0.0093 | 0.8000 | 0.8254 | 5 | 2.8 |
| iqc_fixed | 0.8062 | 0.0096 | 0.7924 | 0.8161 | 5 | 6.0 |

## McNemar median p-values (paired, per-seed)

- iqc_fixed vs iqc_regime2: p=6.854e-101
- iqc_adaptive vs iqc_regime2: p=3.045e-69
- iqc_adaptive vs iqc_fixed: p=3.839e-09
