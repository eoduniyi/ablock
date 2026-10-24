import time
import numpy as np
import torch
from .metrics import calc_basin_masks

def simulate_multiwell_ensemble(
    method,
    model,
    x_init,
    num_steps=1000,
    dt=0.015,
    kBT=0.6,
    gamma=1.0
):
    """
    Simulates a particle ensemble on the 3-basin conformational landscape.
    Supported methods:
      - 'brownian': f = 0
      - 'langevin': f = -grad V_theta
      - 'dilator_only': f = -kappa * grad V_theta (matched contraction, sigma = 0)
      - 'rotor_only': f = sigma * I * grad V_theta (kappa = 0)
      - 'ablock': f = (sigma * I - kappa) * grad V_theta
    Returns:
      x_final, trajectory, elapsed_sec, stats_dict
    """
    x = x_init.clone()
    num_agents = x.shape[0]
    noise_amp = np.sqrt(2.0 * kBT * dt / gamma)
    
    trajectory = []
    visited_BC = torch.zeros(num_agents, dtype=torch.bool)
    last_basin = torch.zeros(num_agents, dtype=torch.long)
    transitions = 0
    
    t0 = time.time()
    for step in range(num_steps):
        if step % 2 == 0:
            trajectory.append(x[0].detach().numpy().copy())
            
        if method == 'brownian':
            x = x + noise_amp * torch.randn_like(x)
        elif method == 'langevin':
            x_req = x.detach().requires_grad_(True)
            V, _, _ = model(x_req)
            gradV = torch.autograd.grad(V.sum(), x_req)[0]
            x = x - (dt / gamma) * gradV + noise_amp * torch.randn_like(x)
        elif method == 'dilator_only':
            x_req = x.detach().requires_grad_(True)
            V, _, kappa = model(x_req)
            gradV = torch.autograd.grad(V.sum(), x_req)[0]
            x = x - dt * kappa * gradV + noise_amp * torch.randn_like(x)
        elif method == 'rotor_only':
            x_req = x.detach().requires_grad_(True)
            V, sigma, _ = model(x_req)
            gradV = torch.autograd.grad(V.sum(), x_req)[0]
            rotor_gradV = torch.stack([-gradV[:, 1], gradV[:, 0]], dim=1)
            x = x + dt * (sigma * rotor_gradV) + noise_amp * torch.randn_like(x)
        elif method == 'ablock':
            x_req = x.detach().requires_grad_(True)
            V, sigma, kappa = model(x_req)
            gradV = torch.autograd.grad(V.sum(), x_req)[0]
            rotor_gradV = torch.stack([-gradV[:, 1], gradV[:, 0]], dim=1)
            flow = sigma * rotor_gradV - kappa * gradV
            x = x + dt * flow + noise_amp * torch.randn_like(x)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        inA, inB, inC = calc_basin_masks(x)
        visited_BC |= (inB | inC)
        new_basin = last_basin.clone()
        new_basin[inA] = 0
        new_basin[inB] = 1
        new_basin[inC] = 2
        transitions += int((new_basin != last_basin).sum().item())
        last_basin = new_basin
        
    elapsed = time.time() - t0
    stats = {
        'visited_bc_pct': float(visited_BC.float().mean().item() * 100.0),
        'transitions_per_particle': float(transitions / num_agents)
    }
    return x, trajectory, elapsed, stats
