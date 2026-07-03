import os
import math
import time
import inspect
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

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))        # B,T,E pre-norm
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Sharing
        self.transformer.wte.weight = self.lm_head.weight

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
        # NOTE: lm_head is tied to wte, so no need to init it separately

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
        x = self.transformer.ln_f(x)
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
        self.data_path = data_path
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


class LRScheduler:
    def __init__(self, max_lr, min_lr, warmup_steps, max_steps):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step+1) / self.warmup_steps
        elif self.warmup_steps <= step < self.max_steps:
            decay_ratio = (step-self.warmup_steps) / (self.max_steps-self.warmup_steps)
            cf = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.min_lr + cf * (self.max_lr - self.min_lr)
        else:
            return self.min_lr

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
    assert torch.cuda.is_available()
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

    # LR Scheduler
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 50
    max_steps = 500
    lr_scheduler = LRScheduler(max_lr, min_lr, warmup_steps=warmup_steps, max_steps=max_steps)

    # Optimizer
    weight_decay = 0.1
    # All params that require grad
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # weight decay 2D params (matmul, embd), skip 1D (biases, layernorms)
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    if ddp_master:
        print(f"Decay {len(decay_params)} tensors with {num_decay} params")
        print(f"Nodecay {len(nodecay_params)} tensors with {num_nodecay} params")
    # Check if fused adam
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )

    ########################################
    # Training step
    for i in range(max_steps):
        model.train()
        ts = time.time()

        # Zero grad
        optimizer.zero_grad(set_to_none=True)

        # Calc gradient
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        
        # Optimizer step
        lr = lr_scheduler.get_lr(i)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        # Logs
        if device.startswith('cuda'):
            torch.cuda.synchronize() # wait for the GPU to finish work
        dt = (time.time() - ts)
        ntok = (batch_size * block_size * ddp_world_size)
        tps = ntok / dt
        if ddp_master:
            print(f"{i:4d}: loss(rank0)={loss.item():.6f}, lr={lr:.4e}, dt={dt*1e3:.2f}ms, tps={tps:.2f}")
        
    torch.distributed.destroy_process_group()
    print("Bye")


if __name__ == "__main__":
    main()
