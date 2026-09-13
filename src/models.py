import torch
import torch.nn as nn

class CliffordScoreNet(nn.Module):
    """
    Clifford Neural Score Network in Cl(2,0).
    Maps 2D conformational state x -> (s_1, s_2, s_bivector).
    Trained via Denoising Score Matching (DSM).
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
        score_vector = out[:, :2]
        bivector_density = out[:, 2:3]
        return score_vector, bivector_density

def apply_clifford_rotor_dilator(score_vector, kappa=0.6, sigma=1.2, chirality=1.0):
    """
    Applies the Geometric Algebra Cl(2,0) rotor-dilator operator:
      f = (sigma * I - kappa) * grad V
    where:
      -grad V = score_vector
      -kappa * grad V = +kappa * score_vector  (dilator / contractive force)
      sigma * I * grad V = -sigma * I * score_vector (transverse rotor force)
    In Cl(2,0), for vector v = (vx, vy), the pseudoscalar action I * v is (-vy, vx).
    """
    sx = score_vector[:, 0]
    sy = score_vector[:, 1]
    
    if isinstance(chirality, torch.Tensor) and chirality.dim() > 1:
        chirality = chirality.squeeze(-1)

    # Contractive dilator term (-kappa * grad V)
    f_dilator_x = kappa * sx
    f_dilator_y = kappa * sy
    
    # Transverse rotor circulation (sigma * I * grad V)
    # With chirality = +/- 1
    f_rotor_x = -sigma * chirality * (-sy)
    f_rotor_y = -sigma * chirality * (sx)
    
    fx = f_dilator_x + f_rotor_x
    fy = f_dilator_y + f_rotor_y
    
    return torch.stack([fx, fy], dim=1)
