import os
import time
import json
import importlib
import argparse
from urllib.request import urlopen
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken
import pyarrow.parquet as pq

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention layer with a projection at the end"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()

        # Compute query, key, value
        q = self.c_q(x)  # B,T,C
        k = self.c_k(x)  # B,T,C
        v = self.c_v(x)  # B,T,C

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        k = k.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        v = v.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        q = q.transpose(1, 2)  # B,nh,T,hs
        k = k.transpose(1, 2)  # B,nh,T,hs
        v = v.transpose(1, 2)  # B,nh,T,hs

        # Scaled dot-product attention with causal masking
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Merge heads
        y = y.transpose(1, 2)  # B,T,nh,hs
        y = y.contiguous()
        y = y.view(B,T,C)

        # Output projection
        out = self.c_proj(y)
        return out

class MLP(nn.Module):
    """Linear transform and activation"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

def norm(x):
    """Param-free replacement for LayerNorm"""
    return F.rms_norm(x, (x.size(-1),))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(norm(x))        # B,T,E pre-norm
        x = x + self.mlp(norm(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Init Params
        self.init_weights()

    def init_weights(self):
        """Initialize the weights of the model."""
        # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        s = 3**0.5 * self.config.n_embd**-0.5

        for block in self.transformer.h:
            # Attention
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            # MLP
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s*0.4, s*0.4)  # smaller init for feedforward
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

    def forward(self, idx, targets):
        B, T = idx.shape
        assert T <= self.config.block_size
        
        # Embeddings
        pos = torch.arange(T, device=idx.device)  # T
        pos_emb = self.transformer.wpe(pos)       #   T,E <- T
        tok_emb = self.transformer.wte(idx)       # B,T,E <- B,T
        x = tok_emb + pos_emb                     # B,T,E

        # Transformer
        for block in self.transformer.h:
            x = block(x)
        x = norm(x)
        logits = self.lm_head(x)   # B,T,V <- B,T,E

        # Loss
        B, T, C = logits.shape
        logits_ = logits.view(B*T, C)  # B*T, C
        targets_ = targets.view(B*T)   # B*T
        loss = F.cross_entropy(logits_, targets_)
        return loss


class DataLoader:
    """Simple data loader for first shard of ClimbMix dataset
    
    On first run, rank 0 downloads the first shard, tokenizes it and saves as .npy file.
    For testing we need roughly micro_batch=64, world_size=8, block_size=1024.
    That is 524,288 tokens per step on 8xH100 node. More than enough for perf testing.
    """
    def __init__(self, batch_size, block_size, proc_rank, world_size):
        self.batch_size = batch_size
        self.block_size = block_size
        self.proc_rank = proc_rank
        self.world_size = world_size
        self.pos = self.batch_size * self.block_size * self.proc_rank

        tokens_filename = "shard_00000.npy"
        parquet_filename = "shard_00000.parquet"  # temp: download, process, delete
        if proc_rank == 0 and not os.path.exists(tokens_filename):
            # Download the first shard
            base_url = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/"
            remote_url = base_url + parquet_filename
            print(f"Rank {proc_rank} downloading {remote_url}")
            with urlopen(remote_url, timeout=30) as r, open(parquet_filename, "wb") as f:
                f.write(r.read())
            print(f"Rank {proc_rank} downloaded {parquet_filename}")

            # Read the text from the parquet file
            documents = []  # list of strings
            pf = pq.ParquetFile(parquet_filename)
            for rg_index in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_index)
                rg_documents = rg.column('text').to_pylist()
                documents.extend(rg_documents)

            # Tokenize and save as .npy ready for training            
            tokenizer = tiktoken.get_encoding("gpt2")
            doc_tokens = tokenizer.encode_ordinary_batch(documents, num_threads=16)
            tokens = []
            for dt in doc_tokens:
                tokens.extend([tokenizer.eot_token] + dt)  # prepend EOT to each document
            np.save("shard_00000.npy", np.array(tokens, dtype=np.uint16))  # gpt2 vocab fits in uint16
            os.remove(parquet_filename)
            print(f"Rank {proc_rank} loaded {len(documents)} docs, {len(tokens)} tokens, saved to {tokens_filename}")

        # Other ranks wait for rank 0 to finish downloading
        torch.distributed.barrier()

        self.tokens = torch.from_numpy(np.load(tokens_filename)).to(torch.int64)



    def get_batch(self):
        buff = self.tokens[self.pos:self.pos+self.batch_size*self.block_size+1]
        x = buff[:-1].view(self.batch_size, self.block_size)
        y = buff[1:].view(self.batch_size, self.block_size)

        self.pos += self.batch_size * self.block_size * self.world_size
        if self.pos + self.batch_size * self.block_size * self.world_size + 1 > len(self.tokens):
            self.pos = self.batch_size * self.block_size * self.proc_rank

        return x, y

def main():
    assert torch.cuda.is_available(), "CUDA is required for this training script."
    assert 'RANK' in os.environ, "Must be run with torchrun, --nproc_per_node=1 falls back to single-GPU training"

    stage_choices = [
        "0_builtin",  # baseline, built-in AdamW + Muon, model wrapped in DDP
        "1_basic",    # AdamW/Muon implemented in Python, model wrapped in DDP
        "2_dist",     # ZeRO-2 version of AdamW/Muon, optimizers handle distributed comms, model not wrapped in DDP
        "3_fused",    # Implement fused kernel in AdamW/Muon
        "4_async",    # Implement async comms in AdamW/Muon
        "5_nanochat"  # Nanochat-compatible version, Polar Express, NorMuon, Cautions Weight Decay
    ]
    parser = argparse.ArgumentParser(description="Train a GPT model with various versions of AdamW/Muon optimizer.")
    parser.add_argument('--depth', type=int, default=12, help='Number of transformer layers. Drives n_head and n_embd.')
    parser.add_argument('--stage', type=str, choices=stage_choices, required=True, help='Which AdamW/Muon version to use.')
    parser.add_argument('--num-iterations', type=int, default=500, help='Maximum number of training steps.')
    parser.add_argument('--device-batch-size', type=int, default=16, help='Batch size per device.')
    parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (AdamW)")
    parser.add_argument("--unembedding-lr", type=float, default=0.003, help="learning rate for unembedding parameters (AdamW)")
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="Learning rate for matrix parameters (Muon)")
    parser.add_argument("--profile", action="store_true", help="Trace training step 10 with torch.profiler, export chrome trace per rank (open in ui.perfetto.dev)")

    args = parser.parse_args()

    # DDP Init
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    ddp_master = ddp_rank == 0  # is this a master?
    device = f'cuda:{ddp_local_rank}'
    device_type = 'cuda'
    ddp_spec = {'rank': ddp_rank, 'local_rank': ddp_local_rank, 'world_size': ddp_world_size, 'master': ddp_master, 'device': device}
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend='nccl', device_id=ddp_local_rank)  # device_id= to suppress barrier warning

    # Print args
    if ddp_master:
        print(f"DDP: {ddp_rank=}, {ddp_local_rank=}, {ddp_world_size=}, {ddp_master=}, "
              f"{device=}, gpu={torch.cuda.get_device_name(0)}, torch={torch.__version__}")
        print(f"Args: {vars(args)}")

    # Enable TF32 for matmul
    torch.backends.cuda.matmul.fp32_precision = 'tf32'  # newer api

    # Reproducibility
    # Model init relies on identical random seeds, will address later
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Batching
    batch_size = args.device_batch_size             # what fits in GPU
    block_size = 1024
    
    if ddp_master:
        print(f"{block_size=}, {batch_size=}, {ddp_world_size=}")

    # Data Loader
    train_loader = DataLoader(
        batch_size=batch_size,
        block_size=block_size,
        proc_rank=ddp_rank,
        world_size=ddp_world_size,
    )

    n_layer = args.depth
    n_head = args.depth    # head size stays at 64 (n_embd/n_head = 64)
    n_embd = 64 * n_layer
    
    # Model
    # NOTE: because vocab_size is expanded model may in theory generate invalid tokens
    model = GPTModel(GPTConfig(
        block_size=1024,     # max context length, max len feed into the model,
        vocab_size=50304,    # 50304 is 'nicer', original was 50257
        n_layer=n_layer,
        n_head=n_head,           # head size n_embd/n_head = 64,
        n_embd=n_embd,          # size of embeddings, i.e. 'first layer',
    ))
    model.to(device)
    model = torch.compile(model)

    # LR Scheduler params
    final_lr_frac = 0.1
    warmup_steps = 50
    warmdown_ratio = 0.4
    max_steps = args.num_iterations

    # LR Scheduler function
    def get_lr(step: int):
        warmdown_steps = round(warmdown_ratio * max_steps)
        if step < warmup_steps:
            return (step+1) / warmup_steps
        if step <= max_steps - warmdown_steps:
            return 1.0
        else:
            progress = (max_steps - step) / warmdown_steps
            return (progress * 1.0) + (1.0 - progress) * final_lr_frac

    # Optimizer
    # AdamW LRs below were tuned at depth=12, n_embd=768
    # This rescale is a muP-flavored heuristic, similar to nanochat. Real HP transfer is out of scope here.
    dmodel_lr_scale = (n_embd/768) ** -0.5
    # Passing compiled model to optimizer is ok because of __getattr__ forwarding
    optim_module = importlib.import_module(f"optim_{args.stage}")
    optimizers = optim_module.setup_optimizers(
        model,
        embedding_lr=args.embedding_lr * dmodel_lr_scale,
        unembedding_lr=args.unembedding_lr * dmodel_lr_scale,
        matrix_lr=args.matrix_lr,
    )

    # Builtin and single-GPU optimizer version require DDP wrapping
    # DDP wrap after optimizer setup is ok: DDP syncs params in place and tensor references are preserved
    if not optim_module.OPTIMIZERS_OWN_COMMS:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Print group inventory
    if ddp_master:
        for opt_name, opt in zip(["AdamW", "Muon"], optimizers):
            for group in opt.param_groups:
                num_tensors = len(group['params'])
                num_el = sum(p.numel() for p in group['params'])
                shapes = sorted({tuple(p.shape) for p in group['params']})
                print(f"{opt_name}: {num_tensors:3d} tensors, {num_el/1e6:7.2f}M params, lr={group['initial_lr']:.3g}, shapes={shapes}")

    # Profiler setup
    profiler = None
    if args.profile:
        assert max_steps > 10, "Need more than 10 steps to profile step 10"
        def export_trace(prof):
            trace_path = f"trace_stage_{args.stage}_rank{ddp_rank}.json.gz"
            prof.export_chrome_trace(trace_path)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=9, warmup=1, active=1, repeat=1),
            on_trace_ready=export_trace,
        )
        profiler.start()

    # Training loop
    model.train()
    metrics = {'loss': [], 'dt': [], 'tps': [], 'max_mem': []}
    for i in range(max_steps):
        torch.cuda.reset_peak_memory_stats()
        ts = time.time()

        # Zero grad
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        # Calc gradient
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.profiler.record_function("fwd_bwd"):
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()
        
        # Profiler Guard
        if args.profile:
            torch.cuda.synchronize()
            torch.distributed.barrier()  # align ranks before profiling AdamW/Muon step
        
        # Optimizer step
        lrm = get_lr(i)
        for optimizer in optimizers:
            for pg in optimizer.param_groups:
                pg['lr'] = lrm * pg['initial_lr']
        with torch.profiler.record_function("adamw_step"):
            optimizers[0].step()
        with torch.profiler.record_function("muon_step"):
            optimizers[1].step()

        # Logs
        torch.cuda.synchronize() # wait for the GPU to finish work
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        dt = (time.time() - ts)
        ntok = (batch_size * block_size * ddp_world_size)
        tps = ntok / dt
        if ddp_master:
            val_loss = loss.item()
            metrics['loss'].append(val_loss)
            metrics['dt'].append(dt)
            metrics['tps'].append(tps)
            metrics['max_mem'].append(max_mem)
            print(f"{i:4d}: loss(rank0)={val_loss:.6f}, lr={lrm:.4e}, dt={dt*1e3:.2f}ms, tps={tps:.2f}, max_mem={max_mem:.2f}GB")
        
        if profiler is not None:
            profiler.step()

    if ddp_master:
        dt_median = float(np.median(metrics['dt'][2:]))  # 0 - compile, 1 - allocator warmup
        tps_median = float(np.median(metrics['tps'][2:]))
        max_mem_max = float(max(metrics['max_mem']))
        loss_mean = float(np.mean(metrics['loss'][-10:]))
        results_dict = {
            "args": vars(args),
            "ddp_spec": ddp_spec,
            "results": {
                "dt_median_ms": dt_median*1e3,
                "tps_median": tps_median,
                "max_mem_GB": max_mem_max,
                "loss_mean_last10": loss_mean
            }
        }
        print(f"RESULTS: {json.dumps(results_dict)}")
    
    if profiler is not None:
        profiler.stop()
        if ddp_master:
            print(f"Rank {ddp_rank} trace exported, open in ui.perfetto.dev")
        
    torch.distributed.destroy_process_group()
    if ddp_master:
        print("Bye")


if __name__ == "__main__":
    main()
