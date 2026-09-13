import pytest
import torch
import numpy as np

from src.potentials import MolecularMultiWellPotential, GlyphPotential
from src.models import CliffordScoreNet, apply_clifford_rotor_dilator
from src.samplers import simulate_ensemble
from src.metrics import compute_fes_rmse, compute_basin_coverage

def test_molecular_potential():
    pot = MolecularMultiWellPotential()
    x = torch.tensor([[0.0, 0.0], [-1.0, 1.0]], dtype=torch.float32)
    v = pot(x)
    assert v.shape == (2,)
    grad = pot.compute_gradient(x)
    assert grad.shape == (2, 2)

def test_clifford_rotor_dilator():
    # Test vector pointing along +x: v = (1.0, 0.0)
    score = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    # With kappa=0, sigma=1, chirality=1:
    # rotor force on score is -sigma * I * score
    # In Cl(2,0), I * (1,0) = (0,1), so rotor force is (0, -1) or orthogonal
    f_rotor = apply_clifford_rotor_dilator(score, kappa=0.0, sigma=1.0, chirality=1.0)
    assert f_rotor.shape == (1, 2)
    # Dot product between force and score should be 0 (purely orthogonal rotor)
    dot = torch.sum(f_rotor * score)
    assert torch.abs(dot) < 1e-6

def test_sampling_step():
    pot = MolecularMultiWellPotential()
    x_init = torch.randn(10, 2)
    def dummy_score(x):
        return -pot.compute_gradient(x)
        
    final_x, traj = simulate_ensemble(
        method='ablock',
        grad_or_score_fn=dummy_score,
        x_init=x_init,
        num_steps=100,
        dt=0.005
    )
    assert final_x.shape == (10, 2)
    assert len(traj) > 0

def test_metrics():
    pot = MolecularMultiWellPotential()
    samples = np.random.randn(100, 2)
    rmse = compute_fes_rmse(samples, pot, res=20)
    assert rmse > 0
    cov = compute_basin_coverage(samples, pot.centers.numpy())
    assert len(cov) == 3
    assert 0 <= sum(cov) <= 1.0
