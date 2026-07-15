

## 2026.07.15 Stages Sweep and Weight-Decay Issue

A sweep across stages (`run_stages_sweep.sh`) with weight decay enabled across stages revealed a small issue. My expectation was that stages 0_builtin, 1_basic, 2_dist, 3_fused, 4_async would have closely overlapping loss curves (they implement same math). Stage 5_nanochat would have lower loss due to Muon extensions. Instead of uniform loss across 0-4, I observed is two sub-clusters: stages 0-2 landed at ~4.377–4.378 and 3-4 at ~4.386 (a +0.008 offset, ~40x the rerun noise of ~0.0002). The cause is stages 0-2 came from PyTorch built-in Muon, while stages 3-4 were achieved by pruning `nanochat` implementation, which uses slightly different convention on combining LR with WD

PyTorch convention uses LR directly:

```python
# weight decay using 'raw' LR
p.mul_(1 - group['lr'] * group['weight_decay'])
# Later, LR scaled for params update only
lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5
p.add_(update, alpha=-lr)
```

Nanochat convention 
```python
# LR scaling FIRST
lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5
# Then, in fused kernel, scaled LR used for both weight decay and param update
update_full = lr * update + lr * wd * params * mask
params.sub_(update_full)
```

I think clearest educational value for this post is: "Stages 0-4 present same math, but progressively faster distributed implementation, while stage 5 presents Muon algorithmic extension". As such I decided to drop weight decay from stages 0-4 completely and hard-code value WD=0.1 for stage 5.

After WD cleanup, we get clean table (4x3090, depth 12):

| Stage | Step Time | Throughput | Max Memory | Final Loss (avg 10 steps) | NProc | Comment |
|---|---:|---:|---:|---:|---:|---|
|  0_builtin | 578.9 ms | 113K tok/s | 8.85 GB | 4.4079 | 4 | PyTorch DDP with built-in AdamW/Muon |
|    1_basic | 582.9 ms | 112K tok/s | 8.85 GB | 4.4062 | 4 | PyTorch DDP with hand-written AdamW/Muon |
|     2_dist | 663.8 ms |  99K tok/s | 7.85 GB | 4.4075 | 4 | Naive ZeRO-2 style optimizers, comms inside optimizers |
|    3_fused | 660.2 ms |  99K tok/s | 7.85 GB | 4.4074 | 4 | ZeRO-2 style optimizers with fused compute sections (torch.compile) |
|    4_async | 648.9 ms | 101K tok/s | 9.09 GB | 4.4056 | 4 | ZeRO-2 style, fused compute, hide comms behind compute |
| 5_nanochat | 648.7 ms | 101K tok/s | 9.09 GB | 4.3542 | 4 | Muon extended with Polar Express, NorMuon, Cautious Weight Decay |

Loss for stages 0-4 is closely clustered, while loss for stage drops down as expected. Notably throughput numbers confirm what was established earlier: ZeRO-2 does not provide improvement on 4x3090 and instead it is slower.

## 2026.07.14 Rerun Hyperparameter Sweep

Rerun hyperparam sweep, changes since last run:
- added `RESULTS:` lines for easy parsing
- we are now on 4x3090 vs 2x up to this point
- dataset changed from tiny Shakespeare to first shard of ClimbMix

| label | matrix_lr | embedding_lr | unembedding_lr | weight_decay | final loss (mean last 10) |
|---|---:|---:|---:|---:|---:|
|              default |   0.02 |    0.3 |  0.003 |    0.0 | 4.3928 |
|      matrix_lr=0.005 |  0.005 |    0.3 |  0.003 |    0.0 | 4.9026 |
|       matrix_lr=0.01 |   0.01 |    0.3 |  0.003 |    0.0 | 4.5555 |
|       matrix_lr=0.04 |   0.04 |    0.3 |  0.003 |    0.0 | 4.7181 |
|    embedding_lr=0.03 |   0.02 |   0.03 |  0.003 |    0.0 | 4.4582 |
|     embedding_lr=0.6 |   0.02 |    0.6 |  0.003 |    0.0 | 4.4507 |
| unembedding_lr=0.001 |   0.02 |    0.3 |  0.001 |    0.0 | 4.5066 |
|  unembedding_lr=0.03 |   0.02 |    0.3 |   0.03 |    0.0 | 4.5810 |
|     weight_decay=0.1 |   0.02 |    0.3 |  0.003 |    0.1 | 4.3559 |

Current defaults are confirmed on every LR axis. Interestingly embedding LR barely matters.

On weight decay: changing WD 0.0->0.1 seems to slightly improve loss, at least on stage `5_nanochat` which is using Cautious Weight Decay (stages 0-4 use plain WD). `nanochat` tested this extensively and we got a nice small sanity check. I'm enabling WD=0.1 as the default for all stages. Strictly, the sweep only supports this for stage `5_nanochat`, and for stages 0-4 value remains untested. I accept it here because stages 0-4 are meant to be a reference.


## 2026.07.13 Logging Silliness

Generating summary tables is far more complex because of few issues with the logs:

- multiple ranks printing at once cause logs to be scrambled, e.g. `Args: {..., 'profile': False}ddp_rank=1, ddp_local_rank=1, ...`
- log params are on `Args: {..}` line, results need to be parsed from per-step lines, some param are in custom `=== wscale stage=0_builtin nproc=1` lines

Because of that script to parse logs is unnecessarily complex. Adding single `RESULTS: ` line including params, ddp setup, and ready to use results fixes it. Can't believe I didn't do it right away after same pain when reproducing `nanochat`

## 2026.07.10 8xH200 Runs

Initial parsing of the results of d26 run on 8xH200, batch 32x1024 per GPU

| Stage      | Step Time | Throughput | Max Memory | Final Loss (avg 10 steps) | Comment |
|---|---:|---:|---:|---:|---|
| 0_builtin  | 529.2 ms  | 495K tok/s | 67.8 GB    | 6.1640 | PyTorch DDP with built-in AdamW/Muon |
| 1_basic    | 537.9 ms  | 487K tok/s | 67.8 GB    | 6.1637 | PyTorch DDP with hand-written AdamW/Muon |
| 2_dist     | 445.6 ms  | 588K tok/s | 57.6 GB    | 6.1636 | Naive ZeRO-2 style optimizers, comms inside optimizers |
| 3_fused    | 442.2 ms  | 593K tok/s | 57.6 GB    | 6.1635 | ZeRO-2 style optimizers with fused compute sections (torch.compile) |
| 4_async    | 436.5 ms  | 601K tok/s | 62.9 GB    | 6.1636 | ZeRO-2 style, fused compute, hide comms behind compute |
| 5_nanochat | 436.9 ms  | 600K tok/s | 62.9 GB    | 6.1567 | Muon extended with Polar Express, NorMuon, Cautious Weight Decay |

Treating stage 0_builtin as a baseline and a starting point. Stage 1_basic expectedly regresses on step time (non fused). 2_dist provides step time and memory improvement, 3_fused further step time reduction, 4_async improves step time and slightly regresses on memory (still below 0_builtin, due to extra tensor buffers), 5_nanochat stays flat step-time/memory, but lowers loss curve.

All above improvements (and regressions) are roughly expected and seem sane on initial inspection. Perfetto traces also show what we expect.

## ~2026.07.09 Hyperparameter sweep

I've ran a quick hyperparameter sweep (`sweep.sh`) on my 2x3090. The dataset is Tiny Shakespeare, stage `5_nanochat`, d12, 500 iterations, device batch 16. Note on this dataset we are in heavy memorization regime, for now should be ok.

Outcomes:
- embedding_lr 0.1 -> 0.3
- unembedding_lr 0.01 -> 0.003
- matrix_lr kept at 0.02 (flat basin 0.005-0.02, degrades at 0.04)
- weight_decay kept 0.0 (no difference at this training horizon)

Dataset: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## ~2026.07.07 Stages 1-5 Complete

All versions now work and produce sane results, for now sanity tested on 2x3090:

- 0_builtin - `torch.optim.AdamW/Muon` with `torch.nn.parallel.DistributedDataParallel`
- 1_basic - handcrafted AdamW/Muon with `torch.nn.parallel.DistributedDataParallel`
- 2_dist - comms transferred handcrafted AdamW/Muon, no PyTorch DDP anymore
- 3_fused - add fused `torch.compile` kernels in both optimizers
- 4_async - overlap comms with compute in optimizers
- 5_nanochat - expand Muon with Polar Express, NorMuon and Cautious Weight Decay

Stages 0-4 have nearly identical initial loss curves to ~4 decimal places. Stage 5_nanochat lowers loss curve.

Since performance on 2x3090 is not indicative next steps are to do hyperparameter sweep and test proper compute node.

## ~2026.07.07 Unexpected Results on 2x3090

**Training d12 on 2x3090** caused unexpected regression. Fancy hand-crafted ZoRO-2 is slower than PyTorch built-in Muon/AdamW/DDP, and not by small margin. Agents initial theory was that PyTorch `DistributedDataParallel` overlaps comms with backward pass, which neither of hand-crafted stages code do.

Why I was initially skeptical:

- I remember my past experiments in my [nanochat-repro](https://github.com/marcinbogdanski/nanochat-repro) (same hand-crafted code as this repo) vs Andrej [nanochat](https://github.com/karpathy/nanochat) and produced nearly identical results.
- Andrej `nanochat` is obviously faster than built-in, otherwise whole project wouldn't make sense

I got my agent to comb through my older nanochat runs, and re-profile `nanochat-repro` vs `nanochat` in separate harness.

The result was that (sparing details, since they are not instrumental to the blog post):

- on 2x3090 comm channel is saturated, ~240ms wire time vs ~35-40ms optimizer compute time. (noteworthy my 2x3090 are on 8x PCIe 3.0, which really doesn't help)
- in that regime, DDP hiding comms behind backward pass compute is materially important, loosing it cases regression
- results between `nanochat-repro` and `nanochat` were almost identical

The conclusion is that on H100/H200 SXM comms speed vs compute speed ratio is significantly different than 2x3090 on 8x PCIe 3.0. Very long way to discover that "different hardware, different optimizations". The second conclusion is PCIe sucks more than I thought.


## ~2026.07.06 Silly Bugs Caught Early

I'm including this because both bugs were caught by comparing loss/step-time to the expected reference. In both cases loss decreased and looked plausible but code was clearly not doing what it should.

1) In stage `2_dist` I defined AdamWDist but forgot to actually call it (facepalm, fixed in `93b695e7`). Caught because stage timings did not change (and agent immediately called out why).

```diff
-    adamw_optimizer = torch.optim.AdamW(adam_groups, fused=True)
+    adamw_optimizer = AdamWDist(adam_groups, fused=True)
```

2) Stage 4 _foreach_copy_ indented incorrectly outside the loop. This caused only the last param group to sync in Muon. This was caught because loss curve immediately and unexpectedly diverged between reference (stages 1-3) and new stage 4. The model still trained, loss improved (mostly dude to AdamW) and everything seemed _fine_.

```diff

         # Copy back params
         for i, group in enumerate(self.param_groups):
             num_params = len(group['params'])
             temp_buffers[i].pop('all_gather_future').wait()
             stacked_all_grads = temp_buffers[i].pop('stacked_all_grads')
-        torch._foreach_copy_(group["params"], list(stacked_all_grads[:num_params].unbind(0)))
+            torch._foreach_copy_(group["params"], list(stacked_all_grads[:num_params].unbind(0)))
```
