import torch
import torch.optim as optim
from .models import CliffordScoreNet

def train_score_network(potential_fn, num_samples=10000, epochs=200, batch_size=256, lr=1e-3, sigma_noise=0.15, device='cpu'):
    """
    Trains CliffordScoreNet via Denoising Score Matching (DSM).
    """
    # Sample ground-truth points via rejection or MCMC from Boltzmann distribution exp(-V / k_B T)
    candidates = (torch.rand(num_samples * 10, 2, device=device) * 5.0) - 2.5
    v = potential_fn(candidates)
    # Target distribution temperature k_B T = 1.0
    p = torch.exp(-(v - v.min()))
    p = p / p.max()
    accept = torch.rand_like(p) < p
    training_data = candidates[accept][:num_samples]
    
    if len(training_data) < num_samples:
        # Fallback if acceptance is slightly lower
        extra = (torch.randn(num_samples - len(training_data), 2, device=device) * 0.8)
        training_data = torch.cat([training_data, extra], dim=0)

    model = CliffordScoreNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = torch.randperm(len(training_data))
        for i in range(0, len(training_data), batch_size):
            batch_x = training_data[perm[i:i+batch_size]]
            noise = torch.randn_like(batch_x) * sigma_noise
            perturbed_x = batch_x + noise
            
            target_score = -noise / (sigma_noise ** 2)
            pred_score, _ = model(perturbed_x)
            
            loss = torch.mean(torch.sum((pred_score - target_score)**2, dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model
