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
- other params for Muon (`momentum=0.95`, `ns_steps=5`), AdamW (`betas=(0.9, 0.95)`), LR scheduler (50 step warmup, hold, 40% linear warmdown to 0.1) come from PyTorch conventions.

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

## Stage 0 - PyTorch Reference

Stage [optim_0_builtin.py](optim_0_builtin.py) is using PyTorch built-in implementations of AdamW and Muon.

```python
import torch

OPTIMIZERS_OWN_COMMS = False  # Trainer script needs to wrap model in DDP

def setup_optimizers(model, embedding_lr=0.3, unembedding_lr=0.003, matrix_lr=0.02):
    # ...

    adamw_optimizer = torch.optim.AdamW(...)
    # ...
    
    # pass weight_decay=0.0 explicitly: torch.optim.Muon defaults to 0.1
    muon_optimizer = torch.optim.Muon(muon_groups, lr=matrix_lr, momentum=0.95, ns_steps=5, weight_decay=0.0)
    # ...
    
    optimizers = [adamw_optimizer, muon_optimizer]
    return optimizers
```

The `optim_0_builtin.py` contains two things: the `OPTIMIZERS_OWN_COMMS` flag telling train script to wrap model in PyTorch DDP, and `setup_optimizers()` which instantiates `torch.optim.AdamW` and `torch.optim.Muon`.

Let's have a look at the Perfetto trace:

![](assets/stage_0a_full.png)

_Image: 8xH200, d26, single step, rank 0 - other 7 ranks look the same_

On the CPU side we have: `thread 50702` is Python main thread (code we write), `thread 50858` is PyTorch autograd worker (roughly, handles backward pass), `adamw_step` (too thin to see) and `muon_step`. These are async dispatch operations to the GPU. Interestingly `muon_step` dominates the CPU side with 1,274 CUDA kernel launches in ~305ms - we will improve that in section `3_fused`.

On GPU side we see `stream 7 7` (compute) and `stream 23 23` (comms). Notice how communication starts almost as soon as backward pass compute starts and overlaps through the whole backward pass. DDP wrapper is doing this for us through hooks. Thanks to that AdamW (very thin) can start right after backward completes and Muon follows after AdamW.

What is not obvious is that optimizers across GPUs redundantly compute same update, using exactly identical gradient inputs.

Baseline recorded on 8xH200:

| Stage | Step Time | Throughput | Max Memory | Final Loss (avg 10 steps) | Comment |
|---|---:|---:|---:|---:|---|
| 0_builtin | 529.2 ms | 495K tok/s | 67.8 GB | 6.1640 | PyTorch DDP with built-in AdamW/Muon |

_Table: recorded on 8xH200, model depth 26 (~1.03B), batch 32x1024 per GPU (262K tok/step)_

<span style="color: red;">TODO: FIX LINKS, THREAD IDS if image swapped, consider cropping or addressing remaining rows on the image, update launched kernels number</span>

## Stage 1 - AdamW/Muon Basic Implementations

In [optim_1_basic.py](optim_1_basic.py) we swap-in PyTorch optimizers with our own first. AdamW first:

```text
adamw_step():
    p = p - lr * wd * p                # decoupled weight decay
    v = B1 * v + (1 - B1) * g          # first moment (direction)
    s = B2 * s + (1 - B2) * g^2        # second moment (scaling)
    v_corrected = v / (1 - B1^t)       # bias correction
    s_corrected = s / (1 - B2^t)
    p = p - lr * v_corrected / (sqrt(s_corrected) + eps)
```

AdamW update is standard momentum `v` divided by square root of second moment `s`. Bias corrections avoid initial updates being biased towards zero. Weight decay is applied directly to the params. AdamW buffers (`v` and `s`) take 2x memory size of the params - since all GPUs hold a copy, it is a substantial amount wasted.

```python
# optim_1_basic.py

OPTIMIZERS_OWN_COMMS = False  # Trainer script needs to wrap model in DDP

class AdamWBasic(torch.optim.Optimizer):
    # ...
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                # Init Buffers
                # ...

                # Weight Decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # AdamW Update
                # ...
                update = ...

                p.add_(update, alpha=-1.0)
```

AdamW implementation is fairly simple: loop over param groups and params, apply two updates: decoupled weight decay and optimizer update. I'm skipping details for brevity.

Let's have a look at the Muon next:

```text
muon_step():
    v = B * v + (1-B) * g            # momentum 
    vv = B * v + (1-B) * g           # Nesterov look-ahead (just lerp again)
    U = newton_schulz(vv)            # orthogonalize
    lr_adj = lr * sqrt(max(1, m/n))  # adjust for aspect ratio
    p = p - lr_adj * U               # update weights

newton_schulz(vv):
    X = vv / ||vv||                        # scale so singular values < 1
    repeat 5 times:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X
```

Similar to AdamW, Muon also uses standard momentum with Nesterov look-ahead, but the AdamW per-element second moment normalization (`1/sqrt(s)`) is replaced with matrix orthogonalization step. **This is why Muon operates on 2D matrices only**, and not individual elements like AdamW. The rough intuition is: raw gradient/momentum matrices are dominated by few large directions and orthogonalization counter that, so "small" directions also get meaningful "step size". Due to computational cost exact SVD is replaced with Newton–Schulz iteration. The aspect ratio adjustment keeps the update magnitude consistent across matrix aspect ratios. I won't pretend to understand this deeply enough to be able to explain, I take update math as given, and leave explanation to [Jordan Post](https://kellerjordan.github.io/posts/muon/).

Below I'm including full Newton-Schulz and Muon code:

```python
@torch.compile
def zeropower_via_newtonschulz(grad, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315    # Keller Jordan's coefficients 
    X = grad.bfloat16()                  # sufficient and faster
    if grad.size(0) > grad.size(1):      # transpose if tall
        X = X.T

    # Scale down to norm at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if grad.size(0) > grad.size(1):
        X = X.T
    return X
```

The `zeropower_via_newtonschulz()` is pure math and `ns_step=5` is held constant across training run. Because of this it is an easy target for `@torch.compile()` decorator. The same is not trivially true for `step()` function and we will address it in stage `3_fused`. We run orthogonalization in bfloat16 for speed. The 'transpose-if-tall' is to ensure `X @ X.mT` is the smaller Gram matrix.

```python
class MuonBasic(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Lazy Init buffers
                if p not in self.state:
                    self.state[p] = {'momentum_buffer': torch.zeros_like(p)}

                # Update momentum v
                # v = B1 * v + (1-B) * g
                v = self.state[p]['momentum_buffer']
                v.lerp_(p.grad, 1 - group['momentum'])

                # Nesterov look-ahead
                # vv = B*v + (1-B)*g
                vv = p.grad.lerp(v, group['momentum'])

                # Calculate update
                update = zeropower_via_newtonschulz(vv, group['ns_steps'])

                # Update params
                lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5
                p.add_(update, alpha=-lr)
```

Muon code has same structure as AdamW: loop over param groups and params, calculate and apply update. Weight decay is omitted for simplicity and will be introduces as Cautious Weight Decay in `5_nanochat`

The most important part, worth reiterating, is that Muon **must** operate on full 2D matrices. This will be critical difference from AdamW when implementing distributed version.

Let's see how we are doing compared to `0_builtin`:

| Stage | Step Time | Throughput | Max Memory | Final Loss (avg 10 steps) | Comment |
|---|---:|---:|---:|---:|---|
| 0_builtin | 529.2 ms | 495K tok/s | 67.8 GB | 6.1640 | PyTorch DDP with built-in AdamW/Muon |
| 1_basic | 537.9 ms | 487K tok/s | 67.8 GB | 6.1637 | PyTorch DDP with hand-written AdamW/Muon |

_Table: recorded on 8xH200, model depth 26 (~1.03B), batch 32x1024 per GPU (262K tok/step)_

First let's notice the loss matches between stages to ~3 decimal places - this is a good correctness check. Secondly the step time increased from 529.2->537.9ms (8.7ms, 1.6%). The regression is due to hand-written optimizers:

- in stage `0_builtin` we create AdamW with `fused=True`
- PyTorch version of Newton-Schultz combines matrix operations like `B = b * A + c * A @ A` into fused GEMMs via `addmm`.

Because of that our `1_basic` versions result in more kernel launches (bad). Interestingly, comparing to model depth 12:

<span style="color: red;">INSERT TABLE D12 on 8xH200 stages 0-1 only, update numbers in prose</span>

It seems regression is roughly equal in absolute terms (8.7ms vs 9.6ms), but relative slowdown is much greater at lower depth because shorter steps expose it more: 53.0->62.6 (9.6ms, 18%).

<span style="color: red;">TODO: FIX LINKS, THREAD IDS if image swapped, consider cropping or addressing remaining rows on the image, update launched kernels number</span>

## Stage 2 - Distributed Implementation

In the [optim_2_dist.py](optim_2_dist.py) we are switching from PyTorch DDP to custom implementation of cross-GPU communication in the optimizer code. 

```diff
# diff optim_1_basic.py optim_2_dist.py
-OPTIMIZERS_OWN_COMMS = False  # Trainer script needs to wrap model in DDP
+OPTIMIZERS_OWN_COMMS = True  # Muon handles distributed comms, model should not be wrapped in DDP
```

With DDP magic disabled our starting point is this: each GPU did a forward/backward pass on different data batch. Params are in sync (same init seed, no broadcast anymore), but **gradients differ between GPUS**. In earlier stages DDP did gradient all-reduce (overlapped with backward) for us - this is now gone.

Here is what we want to achieve:

- assign to each GPU which params does it own exactly
- take gradients and reduce-scatter them, so each GPU gets true gradient (averaged from all GPUs) for it's owned param chunk
- on each GPU compute optimizer update for it's owned param chunk, and apply it to params chunk
- all-gather params across GPUs, so every rank has all params updated and identical before next step

The key part is how to assign params between GPUs:

For AdamW it's easy, since each individual parameter scalar is independently updated, we can shard param tensor across GPUs. Gradient for embedding `wte` is sliced 8-ways, `lm_head` is sliced 8-ways and so on. Lets have a look at the most important difference compared to stage `1_basic`:

```diff
 def step(self):
     rank = torch.distributed.get_rank()
     world_size = torch.distributed.get_world_size()

     for group in self.param_groups:
         for p in group['params']:

             # Establish owned slice
             assert p.size(0) % world_size == 0, f"Param shape {p.shape} not divisible by world size"
+            slice_width = p.size(0) // world_size
+            slice_start = rank * slice_width
+            slice_end = slice_start + slice_width
+            p_slice = p[slice_start:slice_end]  # view

             # Lazy buffer init - only for the slice
             if p not in self.state:
                 self.state[p] = {
                     'step': torch.tensor(0, dtype=torch.int64, device=p.device),
+                    'exp_avg': torch.zeros_like(p_slice),
+                    'exp_avg_sq': torch.zeros_like(p_slice),
                 }

             # Weight Decay - only owned slice
+            p_slice.mul_(1 - group['lr'] * group['weight_decay'])

             # Sync point 1 - average grads across ranks and put in grad_slice
-            grad_slice = torch.empty_like(p.grad[:slice_width])
-            torch.distributed.reduce_scatter_tensor(grad_slice, p.grad, op=torch.distributed.ReduceOp.AVG)

             # Update - owned slice only!
             update = ...
+            p_slice.add_(update, alpha=-1.0)

             # Sync point 2 - sync updated params across ranks
-            torch.distributed.all_gather_into_tensor(p, p_slice)
```

There are two main changes:

1) The ownership - for each param tensor, each rank owns only a slice. On 8xGPU system the code slices the param tensor 8-ways, calculates start/end of the owned slice and creates `p_slice` view. The AdamW buffers are created with `p_slice` size. This is the memory saving. Then both WD update and optimizer step update params tensor via `p_slice` view. This is the compute saving.

2) Comms - The `Sync point 1` does reduce-scatter to populate `grad_slice` with correct averaged gradient based on all 8 ranks. The `Sync point 2` all-gathers param slices across all ranks, ensuring param tensor is in sync across ranks after step completes. 

In this stage both communication primitives are called synchronously - no overlap with compute (and definitely not with the backward pass). This is deliberate at this stage and will be addressed in stage `4_async`. Notably reduce-scatter followed by all-gather has same cost as all-reduce, which was done by DDP wrapper in earlier stages. So we did not add communication cost, but we moved it out of backward pass naively (for now) into optimizer step.

Now let's have a look how to implement distributed Muon, and how it differs from AdamW:

```diff
 def step(self):
     rank = torch.distributed.get_rank()
     world_size = torch.distributed.get_world_size()

     # Reduce Scatter Grads - each rank gets full averaged grad for owned param
     for group in self.param_groups:
+        for i in range(0, len(group['params']), world_size):                # iterate 'world_size' params at a time

             # Prepare tensors - handle padded/non-padded params differently
             if i+rank < len(group['params']):
                 out_tensor = group['params'][i+rank].grad                   # reduce_scatter will write into the owned param's grad
             else:
                 out_tensor = torch.zeros_like(group['params'][0].grad)      # reduce_scatter will write into scratch tensor for padded params
             input_grads = [p.grad for p in group['params'][i:i+world_size]]
             input_grads_padded = pad_to_world_size(input_grads, world_size)

             # Sync point 1
-            torch.distributed.reduce_scatter(out_tensor, input_grads_padded, op=torch.distributed.ReduceOp.AVG)
        
             # Prepare tensors - handle padded/non-padded params differently
             if i+rank < len(group['params']):
                 # Get owned matrix
+                p = group['params'][i+rank]

                 # Lazy Init - create buffer for owned params matrix
                 if p not in self.state:
+                    self.state[p] = {
+                        'momentum_buffer': torch.zeros_like(p),
+                    }

                 # Update - owned matrix only!
                 update = ...
                 lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5
+                p.add_(update, alpha=-lr)
                 input_tensor = p
             else:
                 input_tensor = torch.zeros_like(group['params'][0])  # dummy tensor for padded params
             output_params = [p for p in group['params'][i:i+world_size]]
             output_params_padded = pad_to_world_size(output_params, world_size)
             # Sync point 2
-            torch.distributed.all_gather(output_params_padded, input_tensor)
```

Let's consider how ownership and comms look different:

1) The ownership - iterate param group in W-size chunks (world size), each rank owns one whole param matrix from that chunk. Pad last chunk as needed. For example a depth 12 model on a 8xGPU system has a param group of 12x `c_fc` matrices. Chunks are: [0,1,2,3,4,5,6,7] and [8,9,10,11,zeros,zeros,zeros,zeros]. Rank 0 owns params [0,8] and so on. Params `p` and momentum buffer, update are full not-sliced matrices. Note that if padding on last chunk is needed, then some "tail" ranks will do no work.

2) Comms - The tricky part is padding on the last chunk. Let's take our example from above: [8,9,10,11,zeros,zeros,zeros,zeros]. In the `Sync point 1` the function `reduce_scatter()` requires 8-element list, so `input_grads` needs to be padded with `pad_to_world_size()`. On ranks 0-3 the `out_tensor` is corresponding param `.grad`, but on ranks 4-7 we need to "invent" a dummy buffer. Also note that `out_tensor` aliases one of tensors in `input_grads_padded` - this is a standard in-place pattern. Similarly in `Sync point 2` the `all_gather()` needs full output list of 8-elements. We pad `output_params` and "invent" `input_tensor` on ranks that have no work to do.

Let's see how it looks in Perfetto now:

![](assets/stage_0a_full.png)

In the profiled step (this is separate from table below, slightly slower) improved from 533.2->434.6 (~99 ms) between stages `0_builtin` and `2_dist`. We gained in two places:

1) Optimizers - combined optimizer time reduced from 101.2->46.6 ms (~54.6 ms). This is due to reducing redundant work, mostly in Muon (comparatively AdamW takes little time)
2) Backward Pass - 297.7->259.5ms (38.2ms). PyTorch DDP deliberately partitions the compiled backward pass so it can run comms alongside. Without DDP backward is single clean compiled graph.

The results table (w/o profiler, slightly faster) looks now as follows:

| Stage | Step Time | Throughput | Max Memory | Final Loss (avg 10 steps) | Comment |
|---|---:|---:|---:|---:|---|
| 0_builtin | 529.2 ms | 495K tok/s | 67.8 GB | 6.1640 | PyTorch DDP with built-in AdamW/Muon |
| 1_basic | 537.9 ms | 487K tok/s | 67.8 GB | 6.1637 | PyTorch DDP with hand-written AdamW/Muon |
| 2_dist | 445.6 ms | 588K tok/s | 57.6 GB | 6.1636 | Naive ZeRO-2 style optimizers, comms inside optimizers |

_Table: recorded on 8xH200, model depth 26 (~1.03B), batch 32x1024 per GPU (262K tok/step)_

As expected. Table presents both speed gain and ~10GB peak step memory reduction. The prof of concept sharded optimizer works and doing it's job. Also loss matches between stages to ~3 decimal places, further confirming sanity of our implementation

<span style="color: red;">TODO: FIX LINKS, SWAP IMAGE, image THREAD IDS in prose, update numbers </span>

## Stage 3 - Fused Optimizer Compute Graph

Looking at the Perfetto plot from stage `2_dist` we have a problem:

![](assets/stage_0a_full.png)    // dummy link for now, replace with copy of 2_dist with clearly indicated CPU path nearly as long as GPU path

Currently AdamW performs 63 kernel launches (57 compute, 6 NCCL) and Muon whopping 694 (652 compute, 42 NCCL). The annotated CPU time on the plot does not necessarily mean CPU is 100% busy. Also, GPU side is already saturated. Because of that we don't expect huge improvement. Having said that, it is unnecessary overhead and looks "ugly" so lets reduce it.

We will approach it from two sides:

1) In both AdamW/Muon we will refactor compute into fused optimizer step - wrap as much of the graph under explicit `@torch.compile` as feasible
2) In Muon we will stack params/grads into one larger staging buffer - replace many small kernel launches with fewer larger ones, at the cost of creating staging buffer.

First, let's rewrite Muon math into fused step function:

```python
@torch.compile(dynamic=False, fullgraph=True)
def fused_muon_step(params, grad, momentum_buffer, momentum, lr, steps=5):
    # Update with Nesterov look-ahead
    v = momentum_buffer
    v.lerp_(grad, 1 - momentum)
    vv = grad.lerp(v, momentum)
    # Zeropower via Newton-Schulz
    a, b, c = 3.4445, -4.7750, 2.0315
    X = vv.bfloat16()
    if vv.size(-2) > vv.size(-1):  # replaced from hard-coded 0 and 1
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if vv.size(-2) > vv.size(-1):
        X = X.mT
    update = X
    # Update Params
    params.sub_(lr * update)    # replaced from: p.add_(update, alpha=-lr)
```

Firstly, the `@torch.compile(dynamic=False, fullgraph=True)` means one graph per shape, and no graph breaks, respectively. Both are good for performance and catching unintended graph breaks early.

Secondly, the function parameters: `params`, `grad` and `momentum_buffer` are tensors, but `momentum` and `lr` **are 0-D tensors as well**. `lr` specifically changes every time step due to LR scheduler. A changing float param would trigger torch recompile every step, which would be a disaster for performance. This is why we are passing 0-D tensors instead: to avoid recompiles every step. Simple way to confirm there is no excessive re-compilation is to run script with `TORCH_LOGS=recompiles`. The expectation is one compile per Muon param group (three here). Also note that passing 0-D tensor to `alpha=-lr` causes type error so the line was migrated from `p.add_(update, alpha=-lr)` in stage `2_dist` to `params.sub_(lr * update)` here.

Thirdly, hardcoded dimensions `0,1` are replaced with `-2,-1` to allow function to handle 3D stacked tensors for `prams` and `grads`. We will explore it shortly.

Let's look at the actual Muon step function:

```diff
     def step(self):
         assert all(p.grad is not None for group in self.param_groups for p in group["params"])
         rank = torch.distributed.get_rank()
         world_size = torch.distributed.get_world_size()

         # Reduce Scatter Grads - each rank gets full averaged grad for owned param
         for group in self.param_groups:
             # for i in range(0, len(group['params']), world_size):            # <- inner loop now removed

             p = group['params'][0]  # 'p' variable is used mainly as handle

             # Staging buffer 1 - grads
+            num_real = len(group['params'])                                   # real params, before padding
+            chunk_size = (num_real + world_size - 1) // world_size            # params per rank (ceil div)
+            stacked_all_grads = torch.stack(pad_to_world_size([p.grad for p in group['params']], world_size))
+            stacked_grads = torch.empty_like(stacked_all_grads[:chunk_size])  # this rank's chunk

             # Sync point 1
-            torch.distributed.reduce_scatter_tensor(
-                output=stacked_grads,
-                input=stacked_all_grads,
-                op=torch.distributed.ReduceOp.AVG
-            )

             # NOTE: We will break into two loops here later, when implementing async comms

             # Staging buffer 2 - params
+            idx_start = chunk_size * rank
+            padded_all_params = pad_to_world_size(list(group['params']), world_size)
+            stacked_params = torch.stack(padded_all_params[idx_start:idx_start+chunk_size])  # this rank's chunk

             # Lazy init momentum buffers
             if 'momentum_buffer' not in self.state[p]:
+                self.state[p]['momentum_buffer'] = torch.zeros_like(stacked_params)

             num_params_this_rank = min(chunk_size, max(0, num_real - idx_start))
             if num_params_this_rank > 0:

                 # Scale LR
                 lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5

                 # Convert to 0-D CPU tensors to avoid re-compilation when values change
                 lr = torch.tensor(lr, device='cpu', dtype=torch.float32)
                 momentum = torch.tensor(group['momentum'], device='cpu', dtype=torch.float32)

                 # Call Fused Step
                 fused_muon_step(
                     params=stacked_params[:num_params_this_rank],
                     grad=stacked_grads[:num_params_this_rank],
                     momentum_buffer=self.state[p]['momentum_buffer'][:num_params_this_rank],
                     lr=lr,
                     momentum=momentum,
                     steps=group['ns_steps']
                 )

             # Sync point 2 - Reuse the stacked_all_grads buffer for params
-            torch.distributed.all_gather_into_tensor(stacked_all_grads, stacked_params)

             # Copy Back - copy back to actual params
+            torch._foreach_copy_(group["params"], list(stacked_all_grads[:num_real].unbind(0)))
```

In `Staging buffer 1` we take **all** grads from the group, pad to a multiple of world_size, and stack into 3D `stacked_all_grads` tensor. All grads need to be stacked because this is what reduce-scatter requires.

The `stacked_grads` is created to hold rank owned grads as a 3D stacked tensor (which may be partially padded on some ranks)

Then in `Sync point 1` reduce-scatter operation is performed populating `stacked_grads` with true gradients averaged from all ranks. Let's take previous example of depth 12 model on 8xGPU having 12x `c_fc` matrices. This time `stacked_all_grads` holds [0,1,2,3,4,5,6,7,8,9,10,11,zeros,zeros,zeros,zeros], rank 0 `stacked_grads` chunk is [0,1] and so on up to [zeros,zeros]. In this example ranks 6 and 7 are fully idle (for that param group at least). Same math, different rank assignment. Note that variable `stacked_all_grads` is "consumed" in a sense values it holds are no longer necessary.

`Staging buffers 2` does somewhat analogous stacking for the params - with the difference that **only** owned params are stacked (all we need).  Rank owned params placed in `stacked_params`. A single contiguous `momentum_buffer` is allocated using `p` (group first param object) as a "handle".

Then `lr` and `momentum` scalars are converted to 0-D tensors and `fused_muon_step` is called, resulting in single compiled graph call (per rank). The `num_params_this_rank > 0` is a bypass on idle ranks.

After the muon step, in `Sync point 2` we all-gather params back - so all updated and synchronized params land in (confusingly) `stacked_all_grads`. Since `stacked_all_grads` was "consumed" earlier, and we need a tensor of exactly that shape/dtype, we reuse it.

Since the muon step operated on a copy of grads/params, in `Copy Back` section we need to copy back all params back from `stacked_all_grads` (which now holds params!) to individual locations.

Regarding AdamW, similarly to Muon we implement fused step function, but we **do not** implement stacked tensors or change the communication. In our model Adam handles only three params: `wte`, `wpe` in param group 1 and `lm_head` in param group 2. The `wte` and `wpe` could be flattened (shapes 50304x768 and 1024x768 are not good fit for stacking and AdamW is element-wise), but likely benefit is not there to justify memory and code complexity cost (AdamW performs 63 kernel launches vs 694 Muon, so less to gain). Nanochat doesn't stack/flatten params in AdamW either.

Let's investigate the Perfetto trace:

![](assets/stage_0a_full.png)    // dummy link for now, replace with copy of 3_fused showing reduced CPU overhead

We can see CPU side optimizer blocks occupy much less time (specifically Muon). In a profiled run (slower than table runs), the number of kernel launches in AdamW reduced 63->9 (57 compute, 6 NCCL -> 3, 6) and for Muon reduced 694->157 (652 compute, 42 NCCL -> 151, 6).

AdamW CPU side time reduced from 12.10->1.36ms (10.74ms, -88.8%) and Muon time reduced 312.37->70.09ms (242.28ms, -77.6%). There are small improvements on the GPU side as well: AdamW 4.40->3.84 (0.56ms, -12.7%) and Muon 42.57->37.57ms (5.00ms, -11.7%).

In benchmark runs (different runs, slightly faster):

| Stage | Step Time | Throughput | Max Memory | Final Loss (avg 10 steps) | Comment |
|---|---:|---:|---:|---:|---|
| 0_builtin | 529.2 ms | 495K tok/s | 67.8 GB | 6.1640 | PyTorch DDP with built-in AdamW/Muon |
| 1_basic | 537.9 ms | 487K tok/s | 67.8 GB | 6.1637 | PyTorch DDP with hand-written AdamW/Muon |
| 2_dist | 445.6 ms | 588K tok/s | 57.6 GB | 6.1636 | Naive ZeRO-2 style optimizers, comms inside optimizers |
| 3_fused | 442.2 ms | 593K tok/s | 57.6 GB | 6.1635 | ZeRO-2 style optimizers with fused compute sections (torch.compile) |

The throughput increased from 588K->593K tok/s (+X.XX%). As mentioned earlier, because GPU was already saturated, the benefit is not groundbreaking, but not nothing either. Notably we do not see peak memory increase. Likely staging buffers do not push memory above peak that was established in forward/backward pass, at least in synchronous case.

Stage `3_fused` loss is close to previous stages, providing sanity check for the implementation.

## Stage 4 - Async Communications

Unfortunately we have another problem: comms in optimizers so far are synchronous and do not overlap with compute.

![](assets/stage_0a_full.png)    // dummy link for now, zoom in of 3_fused showing Muon GPU side with compute/comms not overlapping

Inspection of Perfetto traces for stage `3_fused` shows that on the GPU side Muon time in compute vs comms is 16.73 ms and 20.31 ms (44.5% and 54.1%) respectively. For AdamW we see 0.15 ms and 3.67 ms (3.9% and 95.5%). Let's see how much we can recover.

```diff
 def step(self):
     rank = torch.distributed.get_rank()
     world_size = torch.distributed.get_world_size()

+    temp_buffers = {}

     # Reduce Scatter Grads - each rank gets full averaged grad for owned param
+    for i, group in enumerate(self.param_groups):

         # Staging buffer 1
         # ...
        stacked_all_grads = ...
        stacked_grads = ...       # this ranks chunk

         # Sync point 1
-        reduce_scatter_future = torch.distributed.reduce_scatter_tensor(
-            output=stacked_grads,
-            input=stacked_all_grads,
-            async_op=True
-        ).get_future()

         # Temp buffers
+        temp_buffers[i] = {
+            'reduce_scatter_future': reduce_scatter_future,
+            'stacked_grads': stacked_grads,
+            'stacked_all_grads': stacked_all_grads,
+        }
    
     # NOTE: We have broken into two loops here, now that async comms is implemented

     # Do fused step for each param group
+    for i, group in enumerate(self.param_groups):
         p = group['params'][0]  # 'p' variable is used mainly as handle
         num_real = len(group['params'])
         chunk_size = (num_real + world_size - 1) // world_size
 
         # Wait and get buffers
-        temp_buffers[i].pop('reduce_scatter_future').wait()
+        stacked_grads = temp_buffers[i].pop('stacked_grads')
 
         # Staging buffer 2
         # ...
         stacked_params = ...  # this ranks chunk
 
         # Lazy init momentum buffers
         # ...
 
         num_params_this_rank = ...
         if num_params_this_rank > 0:

             # Scale LR
             # ...
 
             # Convert to 0-D CPU tensors to avoid re-compilation when values change
             # ...
 
             # Call Fused Step
             fused_muon_step(
                 params=stacked_params[:num_params_this_rank],
                 grad=stacked_grads[:num_params_this_rank],
                 ...
             )
 
         # Sync point 2 - Reuse the stacked_all_grads buffer for params
+        stacked_all_grads = temp_buffers[i]['stacked_all_grads']
-        all_gather_future = torch.distributed.all_gather_into_tensor(
-            stacked_all_grads,
-            stacked_params,
-            async_op=True
-        ).get_future()
+        temp_buffers[i]['all_gather_future'] = all_gather_future
 
     # Copy back to actual params
+    for i, group in enumerate(self.param_groups):
+        num_real = len(group['params'])
+        temp_buffers[i].pop('all_gather_future').wait()
+        stacked_all_grads = temp_buffers[i].pop('stacked_all_grads')
+        torch._foreach_copy_(group["params"], list(stacked_all_grads[:num_real].unbind(0)))
```

Previously (`3_fused`) we had single loop: reduce-scatter, optimizer step, all-gather, param copy. This is now replaced with three loops and asynchronous operations. The `temp_buffers` is introduced to hold futures and buffer references between the loops. Let's look in detail:

- loop 1: prepares stacked buffers and initiates reduce-scatter. Note `async_op=True` and `.get_future()` to capture future for later use. Function call is non-blocking and moves on immediately. We need to track and later retrieve buffers create in all loop iterations, so we stash them in `temp_buffers`
- loop 2: waits for its group reduce-scatter to complete (`.wait()`) and retrieves buffer from `temp_buffer`. Ten performs optimizer step. While doing so other comms are possibly still in-flight in the background. This is the overlap we want. After the step it issues all-gather, again in async mode, allowing it to possibly overlap with next iteration of compute. At the end we capture buffer reference for later. As before in stage `3_fused` we reuse `stacked_all_grads` to store params.
- loop 3: as in `3_fused` we need to copy back the params, this loop captures comms results and does the copy.

In other words: first loop dispatches all reduce-scatters. Second loop, as chunks arrive, does the step and dispatches results. Third loop collects the results. AdamW is treated with analogous refactor.

![](assets/stage_0a_full.png)    // dummy link for now, replace with copy of 4_async showing Muon GPU side with compute/comms overlapping

Looking at the perfetto plot the comms and compute in Muon indeed overlap.

In the profiled run, Muon GPU time decreased from 37.57 to 27.20 ms (~10 ms gain, 27.6%). Notably in Muon compute adds up to 21.13 ms and comms to 25.82 ms, which now both fit in 27.2 ms total time, thanks to overlap. While overlap isn't free, it still wins (27.2 total < 21.31 compute + 28.82 comms).

<span style="color: red;">TODO: The H200 runs need to be re-done with barrier, below i assume numbers land just fine</span>

As for AdamW, the time stays at approximately ~3.84 (X.XX, Y.YY stages `3_fused` and `4_async` respectively). This makes sense since AdamW compute is only ~0.15ms, so there is nothing to hide comms under.

Footgun warning: Backward pass ends at slightly different time between ranks, introducing around ~10ms jitter. This is more than AdamW step time. To alleviate the jitter effect on AdamW measurement, I added `cuda.synchronize()` along with `dist.barrier()` after backward pass for `--profile` runs. Details in [LOG.md](LOG.md)

Let's have a look at the table runs (again, these are w/o `--profile` so timings differ):

| Stage | Step Time | Throughput | Max Memory | Final Loss (avg 10 steps) | Comment |
|---|---:|---:|---:|---:|---|
| 0_builtin | 529.2 ms | 495K tok/s | 67.8 GB | 6.1640 | PyTorch DDP with built-in AdamW/Muon |
| 1_basic | 537.9 ms | 487K tok/s | 67.8 GB | 6.1637 | PyTorch DDP with hand-written AdamW/Muon |
| 2_dist | 445.6 ms | 588K tok/s | 57.6 GB | 6.1636 | Naive ZeRO-2 style optimizers, comms inside optimizers |
| 3_fused | 442.2 ms | 593K tok/s | 57.6 GB | 6.1635 | ZeRO-2 style optimizers with fused compute sections (torch.compile) |
| 4_async | 436.5 ms | 601K tok/s | 62.9 GB | 6.1636 | ZeRO-2 style, fused compute, hide comms behind compute |

_Table: recorded on 8xH200, model depth 26 (~1.03B), batch 32x1024 per GPU (262K tok/step)_

It looks like we moved from 442.2->436.5 ms (`3_fused`->`4_async`, 5.7 ms, ~1%). It's a small but honest gain.

Notably, the memory jumped 57.6->62.9 GB (still below DDP 67.8 GB). This is because in `3_fused` the intermediate buffers in the optimizers were created and destroyed in each loop iteration. In `4_async` optimizers create all buffers in the first loop, and pass them to second loop.

As earlier, the loss 6.1636 sits in the middle of cluster of runs so far, sanity-confirming current code.

#### Side Note

When originally writing code for this section, I wrongly indented the code in Muon `step()`.

This was fixed with following patch:

```diff
     # Copy back to actual params
     for i, group in enumerate(self.param_groups):
         num_real = len(group['params'])
         temp_buffers[i].pop('all_gather_future').wait()
         stacked_all_grads = temp_buffers[i].pop('stacked_all_grads')
-    torch._foreach_copy_(group["params"], list(stacked_all_grads[:num_real].unbind(0)))
+        torch._foreach_copy_(group["params"], list(stacked_all_grads[:num_real].unbind(0)))
```

With `_foreach_copy_` after the for loop, only the last param group would copy back. All-but-last param groups were left untrained. The model as a whole _was_ training, loss was going down, but code was clearly not doing what it supposed to. The reason this was caught is because loss was noticeably under-trained compared to previous stages - immediate red flag.

---
END OF DOC


