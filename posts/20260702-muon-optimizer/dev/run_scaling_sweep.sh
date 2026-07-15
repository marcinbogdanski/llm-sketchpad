#!/usr/bin/env bash
set -euo pipefail

# Weak scaling sweep: fixed per-GPU batch, vary number of GPUs (1/2/4/8)
# Perf-only runs (50 iters); loss is NOT comparable across world sizes (effective batch grows)
# NOTE: before running, remove lines exceeding your GPU count

# Run sweep:
# ./run_scaling_sweep.sh 2>&1 | tee logs/scaling_sweep.log

# Process results:
# uv run python3 process_stages_sweep.py logs/scaling_sweep.log

OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=1 train.py --stage 0_builtin --num-iterations 50 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=2 train.py --stage 0_builtin --num-iterations 50 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=4 train.py --stage 0_builtin --num-iterations 50 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=8 train.py --stage 0_builtin --num-iterations 50 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=1 train.py --stage 5_nanochat --num-iterations 50 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=2 train.py --stage 5_nanochat --num-iterations 50 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=4 train.py --stage 5_nanochat --num-iterations 50 "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=8 train.py --stage 5_nanochat --num-iterations 50 "$@"
