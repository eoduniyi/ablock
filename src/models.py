import torch
import torch.nn as nn
import torch.nn.functional as F

class CliffordScoreNet(nn.Module):
    """
    Clifford Neural Score Network in Cl(2,0).
    Maps 2D conformational coordinate x -> (V_theta(x), sigma(x), kappa(x)):
      - V_theta(x): Learned scalar potential field (score s_theta(x) = -grad_x V_theta(x))
      - sigma(x): Bivector rotor chirality coefficient in [-1.5, 1.5]
      - kappa(x): Restoring dilator contraction coefficient in [0.1, inf)
    """
    def __init__(self, hidden_dim=64):
        super(CliffordScoreNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        
    def forward(self, x):
        out = self.net(x)
        V = out[:, 0:1]
        sigma = torch.tanh(out[:, 1:2]) * 1.5
        kappa = F.softplus(out[:, 2:3]) + 0.1
        return V, sigma, kappa


def apply_clifford_rotor_dilator(grad_v, sigma=1.2, kappa=0.6, chirality=1.0):
    """
    Applies the Geometric Algebra Cl(2,0) multivector rotor-dilator operator:
      f(x) = (sigma * I - kappa) * grad V(x)
    where I = e1 ^ e2 (I^2 = -1) rotates 2D vectors by 90 degrees counterclockwise:
      I * grad V = (-dV/dy, dV/dx)^T
    """
    rotor_grad_v = torch.stack([-grad_v[:, 1], grad_v[:, 0]], dim=1)
    if isinstance(chirality, torch.Tensor) and chirality.dim() == 1:
        chirality = chirality.unsqueeze(1)
    if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
        sigma = sigma.unsqueeze(1)
    if isinstance(kappa, torch.Tensor) and kappa.dim() == 1:
        kappa = kappa.unsqueeze(1)
    return (sigma * chirality) * rotor_grad_v - kappa * grad_v
