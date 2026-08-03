# main_comparison (5 seeds)

| model | acc mean | std | min | max | n | n_mem |
|---|---|---|---|---|---|---|
| linsvm | 0.8405 | 0.0112 | 0.8264 | 0.8543 | 5 |  |
| logreg | 0.8404 | 0.0115 | 0.8268 | 0.8551 | 5 |  |
| knn | 0.8336 | 0.0114 | 0.8237 | 0.8479 | 5 |  |
| qsvm | 0.8321 | 0.0097 | 0.8185 | 0.8400 | 5 |  |
| iqc_static | 0.8267 | 0.0135 | 0.8088 | 0.8419 | 5 | 2.0 |
| iqc_fixed | 0.8136 | 0.0196 | 0.7844 | 0.8322 | 5 | 6.0 |
| iqc_adaptive | 0.8089 | 0.0280 | 0.7667 | 0.8369 | 5 | 6.4 |
| vqc | 0.7162 | 0.0266 | 0.6920 | 0.7580 | 5 |  |

## McNemar median p-values (paired, per-seed)

- iqc_fixed vs linsvm: p=3.429e-88
- iqc_fixed vs logreg: p=1.148e-84
- iqc_fixed vs knn: p=1.782e-57
- iqc_adaptive vs linsvm: p=2.871e-42
- iqc_adaptive vs logreg: p=6.967e-36
- iqc_adaptive vs iqc_static: p=3.565e-20
- iqc_fixed vs iqc_static: p=4.693e-19
- knn vs linsvm: p=4.716e-15
- knn vs logreg: p=9.633e-14
- iqc_static vs knn: p=1.1e-12
- iqc_adaptive vs iqc_fixed: p=6.347e-08
- iqc_static vs logreg: p=2.652e-06
- iqc_static vs linsvm: p=2.102e-05
- iqc_adaptive vs knn: p=4.565e-05
- linsvm vs logreg: p=0.5334
