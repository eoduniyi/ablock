import torch
import numpy as np
from .models import apply_clifford_rotor_dilator

def simulate_ensemble(
    method,
    grad_or_score_fn,
    x_init,
    num_steps=6000,
    dt=0.005,
    temp=1.0,
    gamma=1.0,
    kappa=0.6,
    sigma=1.2,
    device='cpu'
):
    """
    Simulates stochastic particle ensemble under different sampling dynamics.
    
    Methods:
      - 'brownian': Unconditional Brownian diffusion (f = 0)
      - 'langevin': Standard score-guided Langevin (f = -grad V)
      - 'rotor_only': Rotor ablation (kappa = 0, f = sigma * I * grad V)
      - 'ablock': Full Clifford A-Block (f = (sigma * I - kappa) * grad V)
    """
    N = x_init.shape[0]
    x = x_init.clone().to(device)
    trajectory = [x.clone().cpu()]
    
    # Random initial chirality (+1 or -1) for rotor methods
    chirality = (torch.randint(0, 2, (N, 1), device=device) * 2 - 1).float()
    
    diff_coeff = np.sqrt(2.0 * temp * gamma * dt)

    for step in range(num_steps):
        # Neural score or negative gradient
        # Note: score = -grad V, so grad V = -score
        score = grad_or_score_fn(x)
        
        if method == 'brownian':
            deterministic_force = torch.zeros_like(x)
        elif method == 'langevin':
            # Standard Langevin drift: gamma * score = -gamma * grad V
            deterministic_force = gamma * score
        elif method == 'rotor_only':
            # Rotor only: kappa = 0
            deterministic_force = apply_clifford_rotor_dilator(score, kappa=0.0, sigma=sigma, chirality=chirality)
        elif method == 'ablock':
            # Full Clifford A-Block: both dilator (-kappa grad V) and rotor (sigma I grad V)
            deterministic_force = apply_clifford_rotor_dilator(score, kappa=kappa, sigma=sigma, chirality=chirality)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        noise = torch.randn_like(x) * diff_coeff
        x = x + deterministic_force * dt + noise
        
        # Periodic recording (every 50 steps)
        if (step + 1) % 50 == 0:
            trajectory.append(x.clone().cpu())
            
    return x, trajectory
