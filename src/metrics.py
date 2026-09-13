import numpy as np
import torch
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist

def compute_fes_rmse(samples_np, potential_fn, extent=2.0, res=60, k_B_T=1.0):
    """
    Computes Free Energy Surface (FES) Root Mean Square Error:
      FES_emp(x) = -k_B T * log(rho_emp(x))
      FES_true(x) = V(x)
    Calibrated by aligning median energies across low-energy regions.
    """
    lin = np.linspace(-extent, extent, res)
    X, Y = np.meshgrid(lin, lin)
    grid_pts = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Kernel Density Estimate of empirical distribution
    kde = gaussian_kde(samples_np.T, bw_method=0.20)
    rho_emp = kde(grid_pts.T) + 1e-12
    fes_emp = -k_B_T * np.log(rho_emp)
    
    # Ground truth potential
    grid_tensor = torch.tensor(grid_pts, dtype=torch.float32)
    with torch.no_grad():
        v_true = potential_fn(grid_tensor).cpu().numpy()
        
    # Restrict to low-energy basin region (v_true < 2.0 kcal/mol above minimum)
    v_min = v_true.min()
    mask = (v_true - v_min) < 3.5
    
    # Shift to align minimum
    fes_emp_shifted = fes_emp - np.median(fes_emp[mask])
    v_true_shifted = v_true - np.median(v_true[mask])
    
    rmse = np.sqrt(np.mean((fes_emp_shifted[mask] - v_true_shifted[mask])**2))
    return float(rmse)

def compute_basin_coverage(samples_np, basin_centers_np, radius=0.6):
    """
    Computes fraction of particles located inside each metastable basin.
    """
    N = samples_np.shape[0]
    basin_counts = []
    for center in basin_centers_np:
        dists = np.linalg.norm(samples_np - center, axis=1)
        count = np.sum(dists < radius)
        basin_counts.append(count / N)
    return basin_counts

def compute_glyph_precision_and_coverage(samples_np, glyph_mask, extent=2.4):
    """
    Computes:
      - In-Glyph Precision: percentage of particles located inside the glyph stroke.
      - Stroke Coverage: percentage of glyph pixels visited / occupied.
    """
    size = glyph_mask.shape[0]
    px = ((samples_np[:, 0] + extent) / (2 * extent) * (size - 1)).astype(int)
    py = ((-samples_np[:, 1] + extent) / (2 * extent) * (size - 1)).astype(int)
    
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    in_glyph = np.zeros(len(samples_np), dtype=bool)
    in_glyph[valid] = glyph_mask[py[valid], px[valid]]
    
    precision = np.mean(in_glyph) * 100.0
    
    # Occupied stroke pixels
    occupied_mask = np.zeros_like(glyph_mask, dtype=bool)
    valid_in = valid & in_glyph
    occupied_mask[py[valid_in], px[valid_in]] = True
    
    total_stroke_pixels = np.sum(glyph_mask)
    occupied_stroke_pixels = np.sum(occupied_mask & glyph_mask)
    coverage = (occupied_stroke_pixels / total_stroke_pixels) * 100.0
    
    return precision, coverage
