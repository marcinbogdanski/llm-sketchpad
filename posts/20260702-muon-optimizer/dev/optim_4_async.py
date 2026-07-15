import torch

OPTIMIZERS_OWN_COMMS = True  # Muon handles distributed comms, model should not be wrapped in DDP


@torch.compile(dynamic=False, fullgraph=True)
def fused_adamw_step(
    params,
    grad,
    exp_avg,
    exp_avg_sq,
    step,
    lr,
    beta1,
    beta2,
    eps,
    wd,
):
    # Weight Decay
    # p = p - lr * weight_decay * p
    params.mul_(1 - lr * wd)

    # Update v
    # v = B1 * v + (1-B1) * g
    exp_avg.lerp_(grad, 1-beta1)

    # Update s
    # s = B2 * s + (1-B2) * g**2
    exp_avg_sq.lerp_(grad.square(), 1-beta2)
    
    # Correction
    # Somewhat convoluted way to do:
    # v_corrected = v / (1-B1**t)
    # s_corrected = s / (1-B2**t)
    # p = p - lr * v_corrected / (sqrt(s_corrected)+eps)
    bias1 = 1-beta1**step
    bias2 = 1-beta2**step
    denom = (exp_avg_sq / bias2).sqrt().add_(eps)
    update = exp_avg.div(denom).mul_(lr / bias1)
    params.add_(update, alpha=-1.0)


@torch.compile(dynamic=False, fullgraph=True)
def fused_muon_step(
    params,
    grad,
    momentum_buffer,
    momentum,
    lr,
    steps=5
):
    # Update v: v = B1 * v + (1-B) * g
    v = momentum_buffer
    v.lerp_(grad, 1 - momentum)
    # Nesterov look-ahead: vv = B*v + (1-B)*g
    grad = grad.lerp(v, momentum)

    ###################################
    # Zeropower via Newton-Schulz
    a, b, c = 3.4445, -4.7750, 2.0315
    X = grad.bfloat16()
    if grad.size(-2) > grad.size(-1):
        X = X.mT
    # Scale down to norm at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if grad.size(-2) > grad.size(-1):
        X = X.mT
    update = X

    # Update Params
    params.sub_(lr * update)


class AdamWAsync(torch.optim.Optimizer):
    """ZeRO-2 version of AdamW optimizer"""
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        temp_buffers = {}

        for i, group in enumerate(self.param_groups):
            for j, params in enumerate(group['params']):
                if params.grad is None:
                    continue
                # Lazy Init
                assert params.size(0) % world_size == 0
                slice_width = params.size(0) // world_size
                slice_start = rank * slice_width
                slice_end = slice_start + slice_width

                if params not in self.state:
                    self.state[params] = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(params[:slice_width]),
                        'exp_avg_sq': torch.zeros_like(params[:slice_width]),
                    }
                self.state[params]['step'] += 1

                # Sync point 1
                grad_slice = torch.empty_like(params.grad[:slice_width])
                future = torch.distributed.reduce_scatter_tensor(
                    grad_slice, params.grad, op=torch.distributed.ReduceOp.AVG, async_op=True
                ).get_future()
                params_slice = params[slice_start:slice_end]

                temp_buffers[(i,j)] = {
                    'future': future,
                    'grad_slice': grad_slice,
                    'params_slice': params_slice
                }

        for i, group in enumerate(self.param_groups):
            for j, params in enumerate(group['params']):
                temp_buffers[(i,j)].pop('future').wait()
                grad_slice = temp_buffers[(i,j)].pop('grad_slice')
                params_slice = temp_buffers[(i,j)].pop('params_slice')

                exp_avg = self.state[params]['exp_avg']
                exp_avg_sq = self.state[params]['exp_avg_sq']

                step = torch.tensor(self.state[params]['step'], device='cpu', dtype=torch.float32)
                lr = torch.tensor(group['lr'], device='cpu', dtype=torch.float32)
                beta1 = torch.tensor(group['betas'][0], device='cpu', dtype=torch.float32)
                beta2 = torch.tensor(group['betas'][1], device='cpu', dtype=torch.float32)
                eps = torch.tensor(group['eps'], device='cpu', dtype=torch.float32)
                wd = torch.tensor(group['weight_decay'], device='cpu', dtype=torch.float32)

                fused_adamw_step(
                    params=params_slice,
                    grad=grad_slice,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    step=step,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    wd=wd,
                )

                # Sync point 2
                future2 = torch.distributed.all_gather_into_tensor(
                    params, params_slice, async_op=True
                ).get_future()
                temp_buffers[(i,j)]['future2'] = future2
        
        for i, group in enumerate(self.param_groups):
            for j, params in enumerate(group['params']):
                temp_buffers[(i,j)].pop('future2').wait()
        


        



class MuonAsync(torch.optim.Optimizer):
    """ZeRO-2 inspired version of Muon optimizer
    
    Algorithm:
        v = B * v + (1-B) * g            # momentum 
        vv = B * v + (1-B) * g           # optional, Nesterov look-ahead (just lerp again)
        U = newton_schulz(vv)            # orthogonalize
        lr_adj = lr * sqrt(max(1, m/n))  # adjust for aspect ratio
        p = p - lr_adj * U               # update weights
    """
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        assert all(p.grad is not None for group in self.param_groups for p in group["params"])
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        temp_buffers = {}

        # This will reduce scatter grads, such that each rank gets full averaged grad for owned param
        for i, group in enumerate(self.param_groups):
            p = group['params'][0]  # shape, dtype, device
            num_params = len(group['params'])
            padded_num_params = ((num_params + world_size - 1) // world_size) * world_size
            num_params_per_rank = padded_num_params // world_size

            if len(group['params']) % world_size != 0:
                group['zero_buffer'] = torch.zeros_like(group['params'][0].grad)

            padded_grads = [p.grad for p in group['params']]
            if len(group['params']) % world_size != 0:
                padded_grads.extend([group['zero_buffer']] * (padded_num_params-len(group['params'])))
            stacked_all_grads = torch.stack(padded_grads)
            stacked_grads = torch.empty(num_params_per_rank, *p.shape, dtype=p.dtype, device=p.device)

            # Sync point 1
            reduce_scatter_future = torch.distributed.reduce_scatter_tensor(
                output=stacked_grads,
                input=stacked_all_grads,
                op=torch.distributed.ReduceOp.AVG,
                async_op=True
            ).get_future()

            # Temp buffers
            temp_buffers[i] = {
                'reduce_scatter_future': reduce_scatter_future,
                'stacked_grads': stacked_grads,
                'stacked_all_grads': stacked_all_grads,
            }
        
        # Do fused step for each param group
        for i, group in enumerate(self.param_groups):
            p = group['params'][0]  # shape, dtype, device
            num_params = len(group['params'])
            padded_num_params = ((len(group['params']) + world_size - 1) // world_size) * world_size
            num_params_per_rank = padded_num_params // world_size

            # Wait and get buffers
            temp_buffers[i].pop('reduce_scatter_future').wait()
            stacked_grads = temp_buffers[i].pop('stacked_grads')

            # Sync point 2
            idx_start = num_params_per_rank * rank
            padded_params = [p for p in group['params']]
            if len(group['params']) % world_size != 0:
                padded_params.extend([group['zero_buffer']] * (padded_num_params-len(group['params'])))
            stacked_params = torch.stack(padded_params[idx_start:idx_start+num_params_per_rank])

            # Create buffers
            if 'momentum_buffer' not in self.state[p]:
                self.state[p]['momentum_buffer'] = torch.zeros_like(stacked_params)

            num_params_this_rank = min(num_params_per_rank, max(0, num_params - idx_start))
            if num_params_this_rank > 0:

                # Update
                assert p.grad.ndim == 2
                lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5

                # 0-D CPU tensors to avoid re-compilation when values change
                lr = torch.tensor(lr, device='cpu', dtype=torch.float32)
                momentum = torch.tensor(group['momentum'], device='cpu', dtype=torch.float32)
                fused_muon_step(
                    params=stacked_params[:num_params_this_rank],
                    grad=stacked_grads[:num_params_this_rank],
                    momentum_buffer=self.state[p]['momentum_buffer'][:num_params_this_rank],
                    lr=lr,
                    momentum=momentum,
                    steps=group['ns_steps']
                )

            # Reuse the stacked_all_grads buffer for params
            stacked_all_grads = temp_buffers[i]['stacked_all_grads']
            all_gather_future = torch.distributed.all_gather_into_tensor(
                stacked_all_grads, stacked_params, async_op=True
            ).get_future()
            temp_buffers[i]['all_gather_future'] = all_gather_future

        # Copy back params
        for i, group in enumerate(self.param_groups):
            num_params = len(group['params'])
            temp_buffers[i].pop('all_gather_future').wait()
            stacked_all_grads = temp_buffers[i].pop('stacked_all_grads')
            torch._foreach_copy_(group["params"], list(stacked_all_grads[:num_params].unbind(0)))



def setup_optimizers(model, embedding_lr=0.3, unembedding_lr=0.003, matrix_lr=0.02):
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
    adamw_optimizer = AdamWAsync(adam_groups)

    # Muon for large matrix params
    muon_groups = []
    for shape in sorted({p.shape for p in params_matrix}):
        group_params = [p for p in params_matrix if p.shape == shape]
        muon_groups.append({'params': group_params})
    
    muon_optimizer = MuonAsync(
        muon_groups,
        lr=matrix_lr,
        momentum=0.95,
        ns_steps=5,
    )
    
    # Set initial_lr in param groups for proper LR scaling
    optimizers = [adamw_optimizer, muon_optimizer]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    # [0] is AdamW, [1] is Muon
    return optimizers