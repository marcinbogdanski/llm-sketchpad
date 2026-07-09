#!/usr/bin/env bash
set -euo pipefail

# Run
# CUDA_VISIBLE_DEVICES=0,1 NPROC=2 ./train_all.sh

OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 0_builtin "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 1_basic "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 2_dist "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 3_fused "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 4_async "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat "$@"
