#!/usr/bin/env bash
set -euo pipefail

# Run all optimizer variants in one cross-stage sweep

# Run sweep:
# NPROC=4 ./run_stages_sweep.sh 2>&1 | tee logs/stages_sweep.log

# Process results:
# uv run python3 process_stages_sweep.py logs/stages_sweep.log

OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 0_builtin "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 1_basic "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 2_dist "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 3_fused "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 4_async "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat "$@"
