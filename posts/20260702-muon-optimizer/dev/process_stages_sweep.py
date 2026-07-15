#!/usr/bin/env python3
"""Process results of run_stages_sweep.sh and print markdown table.

Run sweep:
NPROC=4 ./run_stages_sweep.sh 2>&1 | tee logs/stages_sweep.log

Process results:
uv run python3 process_stages_sweep.py logs/stages_sweep.log
"""
import sys
import json

# Example:
# {
#   "args": {
#     "depth": 12,
#     "stage": "5_nanochat",
#     "num_iterations": 500,
#     "device_batch_size": 16,
#     "embedding_lr": 0.3,
#     "unembedding_lr": 0.003,
#     "matrix_lr": 0.02,
#     "weight_decay": 0.0,
#     "profile": false
#   },
#   "ddp_spec": {
#     "rank": 0,
#     "local_rank": 0,
#     "world_size": 4,
#     "master": true,
#     "device": "cuda:0"
#   },
#   "results": {
#     "dt_median_ms": 649.0336656570435,
#     "tps_median": 100974.73130894419,
#     "max_mem_GB": 9.085328578948975,
#     "loss_mean_last10": 4.396883320808411
#   }
# }

# Extract multiple JSON results from log file, one per each train.py run
runs = []
for line in open(sys.argv[1]):
    if line.startswith("RESULTS: "):
        runs.append(json.loads(line.split("RESULTS: ")[1]))  # See example above

# Print markdown table of results
comments = {
    '0_builtin': 'PyTorch DDP with built-in AdamW/Muon',
    '1_basic': 'PyTorch DDP with hand-written AdamW/Muon',
    '2_dist': 'Naive ZeRO-2 style optimizers, comms inside optimizers',
    '3_fused': 'ZeRO-2 style optimizers with fused compute sections (torch.compile)',
    '4_async': 'ZeRO-2 style, fused compute, hide comms behind compute',
    '5_nanochat': 'Muon extended with Polar Express, NorMuon, Cautious Weight Decay',
}
print("| Stage | Step Time | Throughput | Max Memory | Final Loss (avg 10 steps) | NProc | Comment |")
print("|---|---:|---:|---:|---:|---:|---|")

for i, run in enumerate(runs):
    args = run["args"]
    ddp = run["ddp_spec"]
    results = run["results"]
    stage = args['stage']
    comment = comments.get(stage, "")
    print(f"| {stage:>10} | {results['dt_median_ms']:.1f} ms | {results['tps_median']/1000:.0f}K tok/s | "
          f"{results['max_mem_GB']:.2f} GB | {results['loss_mean_last10']:.4f} | "
          f"{ddp['world_size']} | {comment} |")
