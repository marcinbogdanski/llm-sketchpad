#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep, each run varies ONE knob from train.py defaults

# Run sweep:
# NPROC=4 ./run_hyperparam_sweep.sh 2>&1 | tee logs/hyperparam_sweep.log

# Process results:
# uv run python3 process_hyperparam_sweep.py logs/hyperparam_sweep.log

OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat --matrix-lr 0.005 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat --matrix-lr 0.01 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat --matrix-lr 0.04 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat --embedding-lr 0.03 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat --embedding-lr 0.6 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat --unembedding-lr 0.001 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat --unembedding-lr 0.03 "$@"
