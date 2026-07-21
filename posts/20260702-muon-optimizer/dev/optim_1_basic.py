import torch

OPTIMIZERS_OWN_COMMS = False  # Trainer script needs to wrap model in DDP


class AdamWBasic(torch.optim.Optimizer):
    """AdamW optimizer
    
    Algorithm:
        v = B1 * v + (1 - B1) * g          # first moment (direction)
        s = B2 * s + (1 - B2) * g^2        # second moment (scaling)

        v_corrected = v / (1 - B1^t)       # bias correction
        s_corrected = s / (1 - B2^t)

        p = p - lr * v_corrected / (sqrt(s_corrected) + eps)
        p = p - lr * wd * p                # AdamW: decoupled weight decay
    """
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Lazy Init
                if p not in self.state:
                    self.state[p] = {
                        'step': torch.tensor(0, dtype=torch.int64, device=p.device),
                        'exp_avg': torch.zeros_like(p),
                        'exp_avg_sq': torch.zeros_like(p),
                    }
                self.state[p]['step'] += 1

                # Weight Decay
                grad = p.grad
                if group['weight_decay'] != 0.0:
                    # AdamW
                    # p = p - lr * weight_decay * p
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update v
                # v = B1 * v + (1-B1) * g
                v = self.state[p]['exp_avg']
                v.mul_(group['betas'][0]).add_(grad, alpha=1-group['betas'][0])

                # Update s
                # s = B2 * s + (1-B2) * g**2
                s = self.state[p]['exp_avg_sq']
                s.mul_(group['betas'][1])
                s.addcmul_(grad, grad, value=1-group['betas'][1])

                # Correction
                # Somewhat convoluted way to do:
                # v_corrected = v / (1-B1**t)
                # s_corrected = s / (1-B2**t)
                # p = p - lr * v_corrected / (sqrt(s_corrected)+eps)
                t = self.state[p]['step']
                bias1 = 1-group['betas'][0]**t
                bias2 = 1-group['betas'][1]**t
                denom = (s / bias2).sqrt().add_(group['eps'])
                update = v.div(denom).mul_(group['lr'] / bias1)
                p.add_(update, alpha=-1.0)



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


class MuonBasic(torch.optim.Optimizer):
    """Muon optimizer
    
    Algorithm:
        v = B * v + (1-B) * g            # momentum 
        vv = B * v + (1-B) * g           # Nesterov look-ahead (just lerp again)
        U = newton_schulz(vv)            # orthogonalize
        lr_adj = lr * sqrt(max(1, m/n))  # adjust for aspect ratio
        p = p - lr_adj * U               # update weights
    """
    def __init__(self, params, lr=0.01, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Lazy Init
                if p not in self.state:
                    self.state[p] = {
                        'momentum_buffer': torch.zeros_like(p),
                    }

                # Update v
                # v = B1 * v + (1-B) * g
                v = self.state[p]['momentum_buffer']
                v.lerp_(p.grad, 1 - group['momentum'])

                # Nesterov look-ahead
                # vv = B*v + (1-B)*g
                vv = p.grad.lerp(v, group['momentum'])

                # Update
                update = zeropower_via_newtonschulz(vv, group['ns_steps'])
                lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5
                p.add_(update, alpha=-lr)


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
    adamw_optimizer = AdamWBasic(adam_groups)

    # Muon for large matrix params
    muon_groups = []
    for shape in sorted({p.shape for p in params_matrix}):
        group_params = [p for p in params_matrix if p.shape == shape]
        muon_groups.append({'params': group_params})
    
    muon_optimizer = MuonBasic(
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
