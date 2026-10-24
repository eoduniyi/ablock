import numpy as np
import torch

def calc_sliced_w1(pts_np, gt_particles, num_projections=64):
    """
    Computes Sliced 1-Wasserstein Distance (W1) across 64 random 1D projections.
    """
    projections = np.random.randn(num_projections, 2)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    proj_gt = np.sort(np.dot(gt_particles, projections.T), axis=0)
    proj_sample = np.sort(np.dot(pts_np, projections.T), axis=0)
    return float(np.mean(np.abs(proj_gt - proj_sample)))


def calc_fes_rmse(pts_np, potential_fn, kBT=0.6, bins=35):
    """
    Computes Free Energy Surface (FES) Reconstruction RMSE (kcal/mol)
    across metastable basins (V(x) < -1.2 kcal/mol).
    """
    hist, xedges, yedges = np.histogram2d(
        pts_np[:, 0], pts_np[:, 1],
        bins=bins, range=[[-2.2, 2.2], [-2.2, 2.2]]
    )
    prob = hist / (np.sum(hist) + 1e-10)
    fes_est = -kBT * np.log(prob + 1e-5)
    
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    XC, YC = np.meshgrid(xc, yc)
    grid = torch.tensor(np.stack([XC.ravel(), YC.ravel()], axis=1), dtype=torch.float32)
    v_exact = potential_fn(grid).numpy().reshape(bins, bins).T
    
    mask = v_exact < -1.2
    fes_est_norm = fes_est - np.min(fes_est[mask])
    v_exact_norm = v_exact - np.min(v_exact[mask])
    return float(np.sqrt(np.mean((fes_est_norm[mask] - v_exact_norm[mask])**2)))


def calc_basin_masks(pts_torch, r_sq=0.45):
    """Returns boolean masks for Basins A, B, and C."""
    in_A = torch.sum((pts_torch - torch.tensor([-1.0, 1.0]))**2, dim=1) < r_sq
    in_B = torch.sum((pts_torch - torch.tensor([1.0, -0.5]))**2, dim=1) < r_sq
    in_C = torch.sum((pts_torch - torch.tensor([-0.5, -1.0]))**2, dim=1) < r_sq
    return in_A, in_B, in_C


def calc_basin_coverage(pts_torch):
    """Percentage of terminal particles occupying non-initial metastable Basins B or C."""
    _, in_B, in_C = calc_basin_masks(pts_torch)
    coverage = torch.mean((in_B | in_C).float()) * 100.0
    return float(coverage.item())
