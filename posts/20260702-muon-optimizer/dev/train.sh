#!/usr/bin/env bash
set -euo pipefail

# Run
# CUDA_VISIBLE_DEVICES=0,1 NPROC=2 ./train_all.sh

OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py "$@"
