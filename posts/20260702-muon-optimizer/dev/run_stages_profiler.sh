#!/usr/bin/env bash
set -euo pipefail

# Capture one torch.profiler trace per stage per rank (profiler traces step 10)
# Writes trace_stage_<stage>_rank<N>.json.gz into current directory, open in ui.perfetto.dev

# Run:
# NPROC=4 ./run_stages_profiler.sh 2>&1 | tee logs/stages_profiler.log

OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 0_builtin --num-iterations 12 --profile "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 1_basic --num-iterations 12 --profile "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 2_dist --num-iterations 12 --profile "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 3_fused --num-iterations 12 --profile "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 4_async --num-iterations 12 --profile "$@"
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=${NPROC} train.py --stage 5_nanochat --num-iterations 12 --profile "$@"
