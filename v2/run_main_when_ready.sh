#!/usr/bin/env bash
set -uo pipefail
cd /data/tarakesh/Interference-quantum-classifier
export TQDM_DISABLE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
# wait for all 5 CNN embedding sets
for s in 0 1 2 3 4; do
  until [ -f "v2/results/embeddings/cnn_seed${s}/test_labels.npy" ]; do sleep 30; done
done
echo "=== all 5 embeddings ready, running main_comparison ($(date +%H:%M:%S)) ==="
uv run python -m v2.experiments.run_experiment --config v2/configs/main_comparison.yaml
echo "=== aggregating main_comparison ($(date +%H:%M:%S)) ==="
uv run python -m v2.analysis.aggregate --exp_dir v2/results/main_comparison
echo "=== MAIN PIPELINE COMPLETE ($(date +%H:%M:%S)) ==="
