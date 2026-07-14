import torch

OPTIMIZERS_OWN_COMMS = False  # Trainer script needs to wrap model in DDP

def setup_optimizers(model, embedding_lr=0.3, unembedding_lr=0.003, matrix_lr=0.02, weight_decay=0.1):
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
    
    muon_optimizer = torch.optim.Muon(
        muon_groups,
        lr=matrix_lr,
        momentum=0.95,
        ns_steps=5,
        weight_decay=weight_decay,
    )
    
    # Set initial_lr in param groups for proper LR scaling
    optimizers = [adamw_optimizer, muon_optimizer]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    # [0] is AdamW, [1] is Muon
    return optimizers
