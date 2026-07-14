"""Process results of run_hyperparam_sweep.sh and print markdown table.

Run sweep:
NPROC=4 ./run_hyperparam_sweep.sh 2>&1 | tee logs/hyperparam_sweep.log

Process results:
uv run python3 process_sweep_results.py logs/hyperparam_sweep.log
"""
#!/usr/bin/env python3
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

# Add human readable labels indicating what hyperparameter was changed for each run
labels = ["default"]  # first run has default args
for run in runs[1:]:
    diff = {k: v for k, v in run["args"].items() if v != runs[0]["args"][k]}  # collect args that differ from run 0
    label = ", ".join(f"{k}={v}" for k, v in diff.items())  # to string
    labels.append(label)

# Print markdown table of results
print("| label | matrix_lr | embedding_lr | unembedding_lr | weight_decay | final loss (mean last 10) |")
print("|---|---:|---:|---:|---:|---:|")
for i, run in enumerate(runs):
    label = labels[i]
    args = run["args"]
    results = run["results"]
    print(f"| {label:>20} | {args['matrix_lr']:>6} | {args['embedding_lr']:>6} | " 
          f"{args['unembedding_lr']:>6} | {args['weight_decay']:>6} | {results['loss_mean_last10']:.4f} |")

