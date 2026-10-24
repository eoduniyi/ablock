import numpy as np
import torch
import torch.optim as optim
from .models import CliffordScoreNet

def train_score_model(potential_fn, num_samples=12000, epochs=250, lr=2e-3, kBT=0.6, seed=42):
    """
    Trains CliffordScoreNet via multi-scale Vincent-Song Denoising Score Matching (DSM)
    on a thermalized equilibrium ensemble drawn from the 3-basin landscape.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    samples_per_well = num_samples // 3
    x1 = torch.randn(samples_per_well, 2) * 0.35 + torch.tensor([-1.0, 1.0])
    x2 = torch.randn(samples_per_well, 2) * 0.40 + torch.tensor([1.0, -0.5])
    x3 = torch.randn(samples_per_well, 2) * 0.35 + torch.tensor([-0.5, -1.0])
    x_train = torch.cat([x1, x2, x3], dim=0)
    
    for _ in range(150):
        s = potential_fn.true_score(x_train)
        x_train = x_train + 0.012 * s + np.sqrt(2 * kBT * 0.012) * torch.randn_like(x_train)
        
    model = CliffordScoreNet(hidden_dim=64)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    sigmas = [0.05, 0.1, 0.2]
    for epoch in range(epochs):
        optimizer.zero_grad()
        sigma_noise = float(np.random.choice(sigmas))
        noise = torch.randn_like(x_train) * sigma_noise
        x_perturbed = x_train + noise
        
        target_score_dsm = -noise / (sigma_noise ** 2)
        
        x_perturbed_req = x_perturbed.detach().requires_grad_(True)
        V_pred, _, _ = model(x_perturbed_req)
        score_pred = -torch.autograd.grad(V_pred.sum(), x_perturbed_req, create_graph=True)[0]
        
        loss = torch.mean((score_pred - target_score_dsm) ** 2) * (sigma_noise ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
    return model
