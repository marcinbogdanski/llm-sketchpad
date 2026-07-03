#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,2 OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=2 train.py
