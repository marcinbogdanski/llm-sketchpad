#!/usr/bin/env bash
# Coordinate-wise sanity sweep around defaults.
# Stage 5_nanochat, d12, 500 steps, 2 GPUs (via train.sh).
# Each run varies ONE knob from defaults: mlr=0.02 elr=0.1 ulr=0.01 wd=0.0
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

run() {
  local name="$1"; shift
  local log="logs/sweep_${name}.log"
  if [[ -f "$log" ]] && grep -q "^Bye" "$log"; then
    echo "=== ${name}: already done, skipping"
    return
  fi
  echo "=== ${name}: $*"
  ./train.sh --stage 5_nanochat --num-iterations 500 "$@" > "$log" 2>&1
  grep -E "^\s+499:" "$log" || tail -2 "$log"
}

run default
run mlr0.005 --matrix-lr 0.005
run mlr0.01  --matrix-lr 0.01
run mlr0.04  --matrix-lr 0.04
run elr0.03  --embedding-lr 0.03
run elr0.3   --embedding-lr 0.3
run ulr0.003 --unembedding-lr 0.003
run ulr0.03  --unembedding-lr 0.03
run wd0.1    --weight-decay 0.1

echo "SWEEP DONE"
