import torch

OPTIMIZERS_OWN_COMMS = True  # Muon handles distributed comms, model should not be wrapped in DDP


class AdamWDist(torch.optim.Optimizer):
    """ZeRO-2 version of AdamW optimizer"""
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Lazy Init
                assert p.size(0) % world_size == 0, f"Param shape {p.shape} not divisible by world size"
                slice_width = p.size(0) // world_size
                slice_start = rank * slice_width
                slice_end = slice_start + slice_width

                if p not in self.state:
                    self.state[p] = {
                        'step': torch.tensor(0, dtype=torch.int64, device=p.device),
                        'exp_avg': torch.zeros_like(p[:slice_width]),
                        'exp_avg_sq': torch.zeros_like(p[:slice_width]),
                    }
                self.state[p]['step'] += 1

                # Weight Decay
                if group['weight_decay'] != 0.0:
                    # AdamW
                    # p = p - lr * weight_decay * p
                    p[slice_start:slice_end].mul_(1 - group['lr'] * group['weight_decay'])

                # Sync point 1
                grad_slice = torch.empty_like(p.grad[:slice_width])
                torch.distributed.reduce_scatter_tensor(grad_slice, p.grad, op=torch.distributed.ReduceOp.AVG)

                # Update v
                # v = B1 * v + (1-B1) * g
                v = self.state[p]['exp_avg']
                v.mul_(group['betas'][0]).add_(grad_slice, alpha=1-group['betas'][0])

                # Update s
                # s = B2 * s + (1-B2) * g**2
                s = self.state[p]['exp_avg_sq']
                s.mul_(group['betas'][1])
                s.addcmul_(grad_slice, grad_slice, value=1-group['betas'][1])

                # Correction
                # Somewhat convoluted way to do:
                # v_corrected = v / (1-B1**t)
                # s_corrected = s / (1-B2**t)
                # p = p - lr * v_corrected / (sqrt(s_corrected)+eps)
                t = self.state[p]['step']
                bias1 = 1-group['betas'][0]**t
                bias2 = 1-group['betas'][1]**t
                denom = (s / bias2).sqrt().add_(group['eps'])
                update = v.div(denom).mul_(-group['lr'] / bias1)
                p_slice = p[slice_start:slice_end] + update

                # Sync point 2
                torch.distributed.all_gather_into_tensor(p, p_slice)




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
    adamw_optimizer = AdamWDist(adam_groups)

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
