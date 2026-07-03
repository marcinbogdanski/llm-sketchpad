import os
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken
from pathlib import Path

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

    def setup_optimizer(self, embedding_lr=0.1, unembedding_lr=0.01, matrix_lr=0.02):
        """Prepare param groups and setup optimizers. Scale learning rates based on parameter counts"""

        # Separate parameters into groups for different optimizers and learning rates
        params_matrix = list(self.transformer.h.parameters())
        params_embedding = list(self.transformer.wte.parameters()) + list(self.transformer.wpe.parameters())
        params_lm_head = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(params_matrix) + len(params_embedding) + len(params_lm_head)

        # AdamW for dense params
        adam_groups = [
            dict(params=params_embedding, lr=embedding_lr, betas=(0.9, 0.95), weight_decay=0.0),
            dict(params=params_lm_head, lr=unembedding_lr, betas=(0.9, 0.95), weight_decay=0.0)
        ]
        adamw_optimizer = torch.optim.AdamW(adam_groups, fused=True)

        # Muon for large matrix params
        muon_groups = []
        for shape in sorted({p.shape for p in params_matrix}):
            group_params = [p for p in params_matrix if p.shape == shape]
            muon_groups.append({'params': group_params})
        
        muon_optimizer = torch.optim.Muon(
            muon_groups,
            lr=matrix_lr,
            momentum=0.95,
            ns_steps=5,
            weight_decay=0.0,
        )
        
        # Set initial_lr in param groups for proper LR scaling
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        
        # [0] is AdamW, [1] is Muon
        return optimizers

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
    """Simple data loader for Shakespeare dataset"""
    def __init__(self, data_path, batch_size, block_size, proc_rank, world_size):
        self.batch_size = batch_size
        self.block_size = block_size
        self.proc_rank = proc_rank
        self.world_size = world_size
        self.pos = self.batch_size * self.block_size * self.proc_rank

        with open(data_path, 'r') as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)

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
    assert 'RANK' in os.environ, "Must be run with torchrun for DDP."

    # DDP Init
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    ddp_master = ddp_rank == 0  # is this a master?
    device = f'cuda:{ddp_local_rank}'
    device_type = 'cuda'
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend='nccl', device_id=ddp_local_rank)  # device_id= to suppress barrier warning
    print(f"{ddp_rank=}, {ddp_local_rank=}, {ddp_world_size=}, {ddp_master=}, {device=}")

    # Enable TF32 for matmul
    torch.backends.cuda.matmul.fp32_precision = 'tf32'  # newer api

    # Reproducibility
    # Model init relies on identical random seeds, will address later
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Batching
    batch_size = 16             # what fits in GPU
    block_size = 1024
    
    if ddp_master:
        print(f"{block_size=}, {batch_size=}, {ddp_world_size=}")

    # Data Loader
    data_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "tinyshakespeare.txt"
    train_loader = DataLoader(
        data_path=data_path,
        batch_size=batch_size,
        block_size=block_size,
        proc_rank=ddp_rank,
        world_size=ddp_world_size,
    )

    # Model
    # NOTE: because vocab_size is expanded model may in theory generate invalid tokens
    model = GPTModel(GPTConfig(
        block_size=1024,     # max context length, max len feed into the model,
        vocab_size=50304,    # 50304 is 'nicer', original was 50257
        n_layer=12,
        n_head=12,           # head size 768/12=64,
        n_embd=768,          # size of embeddings, i.e. 'first layer',
    ))
    model.to(device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_local_rank])

    # LR Scheduler params
    final_lr_frac = 0.1
    warmup_steps = 50
    warmdown_ratio = 0.4
    max_steps = 500

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
    optimizers = model.module.setup_optimizer(embedding_lr=0.1, unembedding_lr=0.01, matrix_lr=0.02)

    # Print group inventory
    if ddp_master:
        for opt_name, opt in zip(["AdamW", "Muon"], optimizers):
            for group in opt.param_groups:
                num_tensors = len(group['params'])
                num_el = sum(p.numel() for p in group['params'])
                shapes = sorted({tuple(p.shape) for p in group['params']})
                print(f"{opt_name}: {num_tensors:3d} tensors, {num_el/1e6:7.2f}M params, lr={group['initial_lr']:.3g}, shapes={shapes}")

    # Training loop
    model.train()
    for i in range(max_steps):
        ts = time.time()

        # Zero grad
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        # Calc gradient
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        
        # Optimizer step
        lrm = get_lr(i)
        for optimizer in optimizers:
            for pg in optimizer.param_groups:
                pg['lr'] = lrm * pg['initial_lr']
            optimizer.step()

        # Logs
        torch.cuda.synchronize() # wait for the GPU to finish work
        dt = (time.time() - ts)
        ntok = (batch_size * block_size * ddp_world_size)
        tps = ntok / dt
        if ddp_master:
            print(f"{i:4d}: loss(rank0)={loss.item():.6f}, lr={lrm:.4e}, dt={dt*1e3:.2f}ms, tps={tps:.2f}")
        
    torch.distributed.destroy_process_group()
    print("Bye")


if __name__ == "__main__":
    main()
