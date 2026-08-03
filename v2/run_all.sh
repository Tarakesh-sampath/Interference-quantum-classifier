#!/usr/bin/env bash
# Full v2 pipeline orchestration. Run from repo root: bash v2/run_all.sh [stage]
# Stages: embed0 | sweeps | cnn_rest | main | aggregate | all
set -euo pipefail
cd "$(dirname "$0")/.."
export TQDM_DISABLE=1
# Cap BLAS/OMP threads: KMeans n_init and numpy otherwise exhaust OpenBLAS's
# 128 thread-region limit (esp. alongside concurrent training).
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
LOG=v2/results/pipeline.log
mkdir -p v2/results
run() { echo "=== $* ($(date +%H:%M:%S)) ===" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

stage="${1:-all}"

embed0() {
  run uv run python -m v2.experiments.extract_embeddings_v2 --seed 0
}

sweeps() {  # pinned cnn_seed0; fast
  run uv run python -m v2.experiments.run_experiment  --config v2/configs/capacity_sweep.yaml
  run uv run python -m v2.experiments.run_experiment  --config v2/configs/adaptive_regimes.yaml
  run uv run python -m v2.experiments.run_shot_sweep  --config v2/configs/shot_sweep.yaml
  run uv run python -m v2.experiments.run_noise_sweep --config v2/configs/noise_depolarizing.yaml
  run uv run python -m v2.experiments.run_noise_sweep --config v2/configs/noise_thermal.yaml
}

cnn_rest() {  # per-seed CNNs for main comparison
  for s in 1 2 3 4; do
    run uv run python -m v2.experiments.train_cnn_v2 --seed "$s"
    run uv run python -m v2.experiments.extract_embeddings_v2 --seed "$s"
  done
}

main_cmp() {
  run uv run python -m v2.experiments.run_experiment --config v2/configs/main_comparison.yaml
}

aggregate() {
  for e in main_comparison capacity_sweep adaptive_regimes; do
    run uv run python -m v2.analysis.aggregate --exp_dir "v2/results/$e" || true
  done
  run uv run python -m v2.analysis.plots shot     v2/results/shot_sweep/shot_sweep.json || true
  run uv run python -m v2.analysis.plots noise    v2/results/noise_depolarizing/noise_depolarizing.json || true
  run uv run python -m v2.analysis.plots capacity v2/results/capacity_sweep || true
}

case "$stage" in
  embed0) embed0 ;;
  sweeps) sweeps ;;
  cnn_rest) cnn_rest ;;
  main) main_cmp ;;
  aggregate) aggregate ;;
  all) embed0; sweeps; cnn_rest; main_cmp; aggregate ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
echo "=== stage '$stage' done ($(date +%H:%M:%S)) ===" | tee -a "$LOG"
