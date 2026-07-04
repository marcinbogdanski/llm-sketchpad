import torch

OPTIMIZERS_OWN_COMMS = True  # Muon handles distributed comms, model should not be wrapped in DDP


class AdamWDist(torch.optim.AdamW):
    """Minimal modification to AdamW to handle distributed comms, now that DDP is disabled.
    
    Every rank averages gradients before the optimizer step, then applies same optimization
    redundantly on all ranks. Params stay in sync because all ranks started with
    the same model (same seed) and see the same gradients (averaged across ranks).
    """
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
        super().step()



@torch.compile
def zeropower_via_newtonschulz(grad, steps=5):
    """Newton-schulz orthogonalization
    
    Algorithm:
        X = G / ||G||                        # scale so singular values < 1
        repeat 5 times:
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        return X
    """
    assert grad.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = grad.bfloat16()
    if grad.size(0) > grad.size(1):
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


class MuonDist(torch.optim.Optimizer):
    """ZeRO-2 inspired version of Muon optimizer
    
    Algorithm:
        p = p - lr * wd * p              # decoupled weight decay
        v = B * v + (1-B) * g            # momentum 
        vv = B * v + (1-B) * g           # optional, Nesterov look-ahead (just lerp again)
        U = newton_schulz(vv)            # orthogonalize
        lr_adj = lr * sqrt(max(1, m/n))  # adjust for aspect ratio
        p = p - lr_adj * U               # update weights
    """
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Assert all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"])

        # Sync point 1
        # This will reduce scatter grads, such that each rank gets full averaged grad for owned param
        for group in self.param_groups:
            if len(group['params']) % world_size != 0:
                group['zero_buffer'] = torch.zeros_like(group['params'][0].grad)
            for i in range(0, len(group['params']), world_size):
                input_grads = [p.grad for p in group['params'][i:i+world_size]]
                if len(input_grads) < world_size:
                    input_grads.extend([group['zero_buffer']] * (world_size - len(input_grads)))
                out_tensor = group['params'][i+rank].grad if i+rank < len(group['params']) else torch.zeros_like(group['zero_buffer'])
                torch.distributed.reduce_scatter(
                    out_tensor,
                    input_grads,
                    op=torch.distributed.ReduceOp.AVG
                )
            
        for group in self.param_groups:
            for i in range(0, len(group['params']), world_size):
                if i+rank < len(group['params']):
                
                    p = group['params'][i+rank]

                    # Lazy Init
                    if p not in self.state:
                        self.state[p] = {
                            'momentum_buffer': torch.zeros_like(p),
                        }

                    # Decoupled Weight Decay
                    if group['weight_decay'] != 0:
                        p.mul_(1 - group['lr'] * group['weight_decay'])

                    # Update v
                    # v = B1 * v + (1-B) * g
                    v = self.state[p]['momentum_buffer']
                    v.lerp_(p.grad, 1 - group['momentum'])

                    # Optional Nesterov look-ahead
                    # vv = B*v + (1-B)*g
                    vv = p.grad.lerp(v, group['momentum']) if group['nesterov'] else v

                    # Update
                    update = zeropower_via_newtonschulz(vv, group['ns_steps'])
                    lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5
                    p.add_(update, alpha=-lr)
                    input_tensor = p
                else:
                    input_tensor = torch.zeros_like(group['zero_buffer'])

                # Sync point 2
                output_params = [p for p in group['params'][i:i+world_size]]
                if len(output_params) < world_size:
                    output_params.extend(torch.zeros_like(group['zero_buffer']) for _ in range(world_size - len(output_params)))
                torch.distributed.all_gather(output_params, input_tensor)




def setup_optimizers(model, embedding_lr=0.1, unembedding_lr=0.01, matrix_lr=0.02):
    """Prepare param groups and setup optimizers. Scale learning rates based on parameter counts"""
    assert isinstance(model, torch.nn.Module)

    # Separate parameters into groups for different optimizers and learning rates
    params_matrix = list(model.transformer.h.parameters())
    params_embedding = list(model.transformer.wte.parameters()) + list(model.transformer.wpe.parameters())
    params_lm_head = list(model.lm_head.parameters())
    assert len(list(model.parameters())) == len(params_matrix) + len(params_embedding) + len(params_lm_head)

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
    
    muon_optimizer = MuonDist(
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
