# Muon Optimizer

## Introduction

Andrej Karpathy Nanochat implements ZeRO-2 inspired AdamW/Muon with extensions: Polar Express, NorMuon and Cautious Weight Decay. In this post I will rebuild the optimizers from scratch. I will start from PyTorch built-in optimizers wrapped in DDP, go through naive implementations, distributed comms, fused kernel, comms/compute overlap and Muon extensions. On the way I will lay out intermediate steps and profile the gains on a 8xH200 node. Finally will show why the same optimizations regress on my 4x3090.

Results Nanochat-style ZeRO-2 vs PyTorch DDP on 8xH200:

- throughput improved ~21.2%: from 495K tok/s on PyTorch DDP to 600K tok/s ZeRO-2 style
- CUDA max memory reduced ~7.2%: from 67.8 GB DDP to 62.9 GB ZeRO-2 style
- better per-step loss at same step time: thanks to Muon extensions loss reduced 6.1636 -> 6.1567 (avg over last 10 steps, 50 step run)
- recorded on 8xH200, model depth 26 (~1.03B), batch 32x1024 per GPU (262K tok/step)

<span style="color: red;">TODO: SWAP TABLE, SWAP NUMBERS IN PROSE</span>

| Stage | Step Time | Throughput | Max Memory | Final Loss (avg 10 steps) | Comment |
|---|---:|---:|---:|---:|---|
| 0_builtin | 529.2 ms | 495K tok/s | 67.8 GB | 6.1640 | PyTorch DDP with built-in AdamW/Muon |
| 1_basic | 537.9 ms | 487K tok/s | 67.8 GB | 6.1637 | PyTorch DDP with hand-written AdamW/Muon |
| 2_dist | 445.6 ms | 588K tok/s | 57.6 GB | 6.1636 | Naive ZeRO-2 style optimizers, comms inside optimizers |
| 3_fused | 442.2 ms | 593K tok/s | 57.6 GB | 6.1635 | ZeRO-2 style optimizers with fused compute sections (torch.compile) |
| 4_async | 436.5 ms | 601K tok/s | 62.9 GB | 6.1636 | ZeRO-2 style, fused compute, hide comms behind compute |
| 5_nanochat | 436.9 ms | 600K tok/s | 62.9 GB | 6.1567 | Muon extended with Polar Express, NorMuon, Cautious Weight Decay |

Key insight is that **AdamW and Muon shard differently**. AdamW is element wise and as such can shard within tensors. But Muon _needs complete matrix_ to orthogonalize, as such it needs to shard across tensors (whole tensors go to different GPUs). We will explore how that affects optimizer code in Stage 2.

Gains are not universal. While on 8xH200 sharding throughput increase is ~21.2%, in local testing **on my 4x3090 PCIe sharded optimizers reduce performance** by ~10.6% (113K → 101K tok/s). In reference implementation (`0_builtin`) PyTorch DDP runs the full optimizer update redundantly on every rank (bad), but it does overlap comms with backward pass through PyTorch hooks (good). This implementation (`4_async`), through sharding, reduces per-rank optimizer work and state by 8 (good), but we move comms to optimizers and no longer hide it under backward pass (bad). On 8xH200 comms are so fast even non-overlapped non-fused version wins (`2_dist`). On 4x3090 PCIe slow comms dominate the optimizer such that even `4_async` looses. Since `nanochat` doesn't include backward-pass-overlap we skip it as well. In short: different hardware, different bottleneck, different optimizations.

This post is organized as:

- Introduction
- Setup - how to run code, establish stages 0-4 should have same loss-signature (same math, different implementations)
- Stage 0 "builtin" - reference implementation using PyTorch `torch.optim` and `DistributedDataParallel`
- Stage 1 "basic" - from-scratch hand-written implementations of AdamW and Muon still under DDP
- Stage 2 "dist" - move comms to optimizers - explore how sharding differs between AdamW and Muon (within tensors vs complete matrices)
- Stage 3 "fused" - fuse hot paths within optimizers with `torch.compile`
- Stage 4 "async" - overlap communication with computation
- Stage 5 "nanochat" - implement extensions to Muon improving loss withing same step-time
- 4x3090 Comparison - explore why same code causes regression on local node (spoiler: PCIe is slow)
- Closing Section - lessons learned

## Setup

Steps to reproduce:

```bash
# Setup
git clone https://github.com/marcinbogdanski/llm-sketchpad.git
cd llm-sketchpad/posts/20260702-muon-optimizer
uv sync

# Run sweep across stages 0-5:
NPROC=8 ./run_stages_sweep.sh 2>&1 | tee logs/stages_sweep.log
# Process results into table:
uv run python3 process_stages_sweep.py logs/stages_sweep.log
```

Main entry point is [train.py](train.py) script: a frozen harness reused between stages. It contains model, dataloader and train loop. Optimizers are located in six `optim_*.py` files, exactly one per stage. Each contains as `setup_optimizers()` that wires optimizers with the model. Stages 1-5 contain actual AdamW/Muon implementations as well. The `*.sh` are thin convenience scripts.

The main idea is **stages 0-4 implement same AdamW/Muon algorithms, with incrementally added optimizations**, as such their loss trajectories should be overlapping very closely. The **stage 5 expands Muon and should reduce loss** at same step count. The first contract is crucial, and helped detect some subtle bugs during development.

Experiments were performed with model depth 12 or 26 (`n_layer`, depending on experiment). Num heads is set to equal number of layers (`n_head = args.depth`) and hidden size is auto-scaled with depth (`n_embd = 64 * n_layer`). In addition:

- the LRs for AdamW (`embedding_lr`, `unembedding_lr`) and Muon (`matrix_lr`) are result of a quick sanity sweep (see [run_hyperparam_sweep.sh](run_hyperparam_sweep.sh), stage 5, d12, W=4, 500 steps, single seed).
- in addition AdamW LRs are scaled by muP-flavored heuristic `(n_embd/768)**-0.5` which has effect for depths `n_layer != 12`
- weight decay - enabled for Muon stage `5_nanochat` for Cautious Weight Decay at `WD=0.1`. Disabled otherwise throughout for Muon and AdamW
- other params for Muon (`momentum=0.95`, `ns_steps=5`), AdamW (`betas=(0.9, 0.95)`), LR scheduler (50 step warmup, hold, 40% linear warmdown to 0.1) come from `nanochat`/PyTorch conventions.

Let's move on to high level structure of `train.py`:

```python
# train.py

class GPTModel(nn.Module):
    def __init__(self, config):
        # ...
    def init_weights(self):
        # ...
    def forward(self, idx, targets):
        # ...
        return loss
```

The `GPTModel` is fairly standard decoder-only transformer, including multi-head SDPA attention, MLP block with 4x up-projection, GELU(tanh) activation and pre-norm blocks. Trained with no dropout and GPT-2 vocab 50257 padded up to more "round" 50304. Few points of note:

- normalization layers are param-free `F.rms_norm`, which has no learnable weights (zero 1-D params). This simplifies things and avoids extra bucked of 1-D params to AdamW optimizer.
- all `nn.Linear` have no biases for the same reason to avoid 1-D "small" params
- embeddings `wte` and output head `lm_head` are untied to allow separate learning rates
- learned position embeddings `wpe` instead of RoPE - deliberate simplification to avoid RoPE machinery
- residual projections zeroed (`attn.c_proj`, `mlp.c_proj`) and `lm_head` initialized to small (std 0.001) values, resulting in expected initial loss of `ln(50304) ~= 10.83`

Overall the architecture is inspired by `nanochat` but simplified heavily towards vanilla for the purpose of this blog. Muon handles matrices, AdamW embeddings/head, no 1-D params.

Next, the data loader:

```python
# train.py

class DataLoader:
    """Simple data loader for first shard of ClimbMix dataset"""
    # ...
    def __init__(self, batch_size, block_size, proc_rank, world_size):
        # Download and tokenize on first run
        # ...

    def get_batch(self):
        # Load and iterate
        # ...
        return x, y
```

Dataset is composed of first shard of ClimbMix (~86K docs, ~55M tokens). The longest run we do is 500 steps x 8 GPU x 32 batch x 1024 block ~= 131M tokens, or roughly 2.4 passes. This is deliberate concession: hopping shards would add complexity and 2.4 reuse is benign for loss comparison because all stages see the same batches in identical order.

Dataloader is purposefully simple and included for transparency. On first run it will download the shard, tokenize it with tiktoken GPT-2 tokenizer and save as `np.array`. This pre-tokenization simplifies training loop. During operation whole shard is loaded on each rank, then each rank consumes different subset of tokens. For simplicity, no effort is made to align document boundaries.

```python
# train.py

def main():
    # DDP Init
    torch.distributed.init_process_group(...)
    # ...

    # Data Loader
    train_loader = DataLoader(...)
    # ...

    # Model
    model = GPTModel(...)
    model = torch.compile(model)
    # ...

    # LR Scheduler
    # ...

    # Optimizers - dynamically loaded
    optim_module = importlib.import_module(f"optim_{args.stage}")
    optimizers = optim_module.setup_optimizers(model, ...)

    # Builtin and single-GPU optimizer version require DDP wrapping
    if not optim_module.OPTIMIZERS_OWN_COMMS:
        model = DDP(model, device_ids=[ddp_local_rank])
    # ...

    # Print group inventory
    for opt_name, opt in zip(["AdamW", "Muon"], optimizers):
        for group in opt.param_groups:
            print(...)

    # Profiler setup
    # ...

    # Train loop
    # ...
```

The `main()` is fairly standard: initialize process group, instantiate dataloader and model (and compile it). LR scheduler changes LR every step, this exposes need for careful param passing to fused kernels in stages 3-5.

The interesting part is the stage selection: it is selected with `--stage` param and corresponding `optim_*.py` module is loaded dynamically with `importlib`. Then `setup_optimizers` is called to wire up given stage optimizers implementations with the model. Stages `0_builtin` and `1_basic` depend on DDP wrapper for the comms, while other stages handle comms inside optimizers. For this reason modules expose `OPTIMIZERS_OWN_COMMS` which is used to enable/disable DDP wrapper depending on a stage loaded.

Notably the DDP wrapper is called _after_ optimizer setup. This is ok because DDP broadcasts tensors in place and preserves params identity (otherwise AdamW/Muon would operate on stale references).

The param group inventory shows exactly which optimizer is doing what, expected output is

```txt
AdamW:  2 tensors,   39.42M params, lr=0.3,   shapes=[(1024, 768), (50304, 768)]
AdamW:  1 tensors,   38.63M params, lr=0.003, shapes=[(50304, 768)]
Muon:  48 tensors,   28.31M params, lr=0.02,  shapes=[(768, 768)]
Muon:  12 tensors,   28.31M params, lr=0.02,  shapes=[(768, 3072)]
Muon:  12 tensors,   28.31M params, lr=0.02,  shapes=[(3072, 768)]
```

As we can see AdamW is responsible for embeddings (`wpe` 1024x768, `wte` 50304x768) and LM head (`lm_head` 50304x768), while Muon for attention Q,K,V and `attn.c_proj` (768x768) and MLP projections (`c_proj` 768x3072, `c_fc` 3072x768).

Moving onto the training loop:

```python
    # Training loop
    for i in range(max_steps):
        # Zero grad
        # ...

        # Forward / Backward
        x, y = train_loader.get_batch()
        with torch.profiler.record_function("fwd_bwd"):
            with torch.autocast(..., dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()

        # LR Scheduler
        # ...
        
        # Optimizers step
        with torch.profiler.record_function("adamw_step"):
            optimizers[0].step()
        with torch.profiler.record_function("muon_step"):
            optimizers[1].step()

        # Logs
        # ...

    # Final results
    print(f"RESULTS: {json.dumps(results_dict)}")

    # DDP Cleanup
    torch.distributed.destroy_process_group()
```

The training loop is fairly standard as well: zero grad, get batch, forward/backward and optimizer step. The fwd/bwd and optimizer steps are tagged for easier exploration in Perfetto viewer. The forward pass is done with autocast to `bfloat16`.

All logging is done on rank zero only. The most important is the `RESULTS: ` line: it includes run arguments, ddp setup info and actual results - a single line with an JSON object that contains everything we need to process results easily.

As for the `optim_*.py` files: they all follow roughly similar template, here I'm showing redacted `optim_1_basic.py` as an example:

```python
import torch

OPTIMIZERS_OWN_COMMS = False  # Trainer script needs to wrap model in DDP

class AdamWBasic(torch.optim.Optimizer):
    def __init__(self, params, ...):
        # ...
    def step(self):
        # ...

class MuonBasic(torch.optim.Optimizer):
    def __init__(self, params, ...):
        # ...
    def step(self):
        # ...
```

First comes the `OPTIMIZERS_OWN_COMMS` which is used by `train.py` to enable/disable DDP wrapper (enable for stages `0_builtin`, `1_basic`, disable for `2_dist`, `3_fused`, `4_async`, `5_nanochat`). The AdamW and Muon definitions are present in stages `1_basic` onward (stage 0 uses PyTorch builtin optimizers)

```python
def setup_optimizers(model, embedding_lr=0.3, unembedding_lr=0.003, matrix_lr=0.02):
    """Prepare param groups and setup optimizers."""

    # Separate parameters into groups for different optimizers and learning rates
    params_matrix = list(model.transformer.h.parameters())
    params_embedding = ...
    params_lm_head = ...

    # AdamW for dense params (params_embedding, params_lm_head)
    adam_groups = [
        dict(params=params_embedding, ...),
        dict(params=params_lm_head, ...)
    ]
    adamw_optimizer = AdamWBasic(adam_groups)

    # Muon for large matrix params, group by shape
    muon_groups = []
    for shape in sorted({p.shape for p in params_matrix}):
        group_params = [p for p in params_matrix if p.shape == shape]
        muon_groups.append({'params': group_params})
    muon_optimizer = MuonBasic(muon_groups, ...)
   
    optimizers = [adamw_optimizer, muon_optimizer]
    return optimizers
```

The `setup_optimizers()` function is responsible for grouping params, creating param groups and instantiating optimizers. For Muon this means collecting all hidden-block 2D tensors to `params_matrix` and then grouping them by shape, so they can be processed together.

<span style="color: red;">TODO: FIX/CONFIRM LINKS</span>

