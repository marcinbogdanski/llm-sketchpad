

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
