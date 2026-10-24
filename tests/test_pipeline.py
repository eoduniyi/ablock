import pytest
import torch
import numpy as np

from src.potentials import MolecularMultiWellPotential, GlyphPotential
from src.models import CliffordScoreNet, apply_clifford_rotor_dilator
from src.samplers import simulate_multiwell_ensemble
from src.metrics import calc_fes_rmse, calc_basin_coverage

def test_molecular_potential():
    pot = MolecularMultiWellPotential()
    x = torch.tensor([[0.0, 0.0], [-1.0, 1.0]], dtype=torch.float32)
    v = pot(x)
    assert v.shape == (2,)
    grad = pot.compute_gradient(x)
    assert grad.shape == (2, 2)

def test_clifford_rotor_dilator():
    grad_v = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    # With kappa=0, sigma=1, chirality=1: f = I * grad_v = (0, 1), orthogonal to grad_v
    f_rotor = apply_clifford_rotor_dilator(grad_v, sigma=1.0, kappa=0.0, chirality=1.0)
    assert f_rotor.shape == (1, 2)
    dot = torch.sum(f_rotor * grad_v)
    assert torch.abs(dot) < 1e-6

def test_sampling_step():
    model = CliffordScoreNet(hidden_dim=16)
    x_init = torch.randn(10, 2)
    final_x, traj, elapsed, stats = simulate_multiwell_ensemble(
        method='ablock',
        model=model,
        x_init=x_init,
        num_steps=20,
        dt=0.015
    )
    assert final_x.shape == (10, 2)
    assert len(traj) > 0
    assert 'visited_bc_pct' in stats

def test_metrics():
    pot = MolecularMultiWellPotential()
    samples = np.random.randn(200, 2)
    rmse = calc_fes_rmse(samples, pot, bins=25)
    assert rmse > 0
    cov = calc_basin_coverage(torch.tensor(samples, dtype=torch.float32))
    assert 0.0 <= cov <= 100.0
