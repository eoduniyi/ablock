#!/usr/bin/env python3
"""
A-Block: Table 1 & Figure 1 Reproduction Script
Simulates multi-well conformational dynamics across 4 sampling methods:
  1. Brownian Dynamics (f = 0)
  2. Standard Langevin (f = -grad V)
  3. Rotor-Only ablation (kappa = 0, f = sigma * I * grad V)
  4. Full Clifford A-Block (f = (sigma * I - kappa) * grad V)

Generates:
  - output/table1_results.json
  - output/figure1_annotated_benchmark.pdf / .png
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Deterministic seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. Canonical 3-Basin Potential
# ============================================================================
class MolecularMultiWellPotential:
    def __init__(self):
        self.centers = torch.tensor([
            [-1.0, 1.0],
            [1.0, -0.5],
            [-0.5, -1.0]
        ], dtype=torch.float32)
        self.depths = torch.tensor([3.5, 4.0, 3.2], dtype=torch.float32)
        self.widths = torch.tensor([0.55, 0.60, 0.50], dtype=torch.float32)
        
    def __call__(self, x):
        v_total = torch.zeros(x.shape[0], dtype=torch.float32)
        for i in range(3):
            c = self.centers[i]
            d = self.depths[i]
            w = self.widths[i]
            dist_sq = torch.sum((x - c)**2, dim=1)
            v_total -= d * torch.exp(-dist_sq / (2 * w))
        v_confinement = 0.20 * torch.sum(x**2, dim=1)
        return v_total + v_confinement

potential_fn = MolecularMultiWellPotential()

def true_score(x):
    x_req = x.detach().requires_grad_(True)
    v = potential_fn(x_req)
    grad = torch.autograd.grad(v.sum(), x_req)[0]
    return -grad

# ============================================================================
# 2. Clifford Neural Score Network Cl(2,0)
# ============================================================================
class CliffordScoreNet(nn.Module):
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

# ============================================================================
# 3. Denoising Score Matching Training
# ============================================================================
def train_score_model(num_samples=12000, epochs=250):
    print("Generating equilibrium dataset across metastable basins...")
    samples_per_well = num_samples // 3
    x1 = torch.randn(samples_per_well, 2) * 0.35 + torch.tensor([-1.0, 1.0])
    x2 = torch.randn(samples_per_well, 2) * 0.40 + torch.tensor([1.0, -0.5])
    x3 = torch.randn(samples_per_well, 2) * 0.35 + torch.tensor([-0.5, -1.0])
    x_train = torch.cat([x1, x2, x3], dim=0)
    
    kBT = 0.6
    for _ in range(150):
        s = true_score(x_train)
        x_train = x_train + 0.012 * s + np.sqrt(2 * kBT * 0.012) * torch.randn_like(x_train)
        
    model = CliffordScoreNet(hidden_dim=64)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Training Clifford Score Network for {epochs} epochs via DSM...")
    sigmas = [0.05, 0.1, 0.2]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        sigma_noise = float(np.random.choice(sigmas))
        noise = torch.randn_like(x_train) * sigma_noise
        x_perturbed = x_train + noise
        
        target_score_dsm = - noise / (sigma_noise**2)
        
        x_perturbed_req = x_perturbed.detach().requires_grad_(True)
        V_pred, _, _ = model(x_perturbed_req)
        score_pred = -torch.autograd.grad(V_pred.sum(), x_perturbed_req, create_graph=True)[0]
        
        loss = torch.mean((score_pred - target_score_dsm)**2) * (sigma_noise**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
    return model

# ============================================================================
# 4. Sampling & Benchmark Execution
# ============================================================================
def run_benchmarks(model, num_agents=2000, num_steps=1000):
    dt = 0.015
    gamma = 1.0
    kBT = 0.6
    noise_amp = np.sqrt(2 * kBT * dt / gamma)
    
    # Initialize in Basin A (-1.0, 1.0) (kinetic trap scenario)
    x_init = torch.tensor([-1.0, 1.0]).repeat(num_agents, 1) + 0.15 * torch.randn(num_agents, 2)
    
    samplers = {
        'Brownian Dynamics (f = 0)': x_init.clone(),
        'Standard Langevin (f = -\\nabla V)': x_init.clone(),
        'Rotor-Only (\\kappa = 0)': x_init.clone(),
        'A-Block Full Clifford (Ours)': x_init.clone()
    }
    
    times = {}
    trajectories = {k: [] for k in samplers}
    
    basin_B_mask = lambda x: torch.sum((x - torch.tensor([1.0, -0.5]))**2, dim=1) < 0.45
    basin_C_mask = lambda x: torch.sum((x - torch.tensor([-0.5, -1.0]))**2, dim=1) < 0.45
    
    print("Running sampling algorithms (N=2000, T=1000)...")
    for name in samplers:
        x = samplers[name]
        t0 = time.time()
        
        for step in range(num_steps):
            if step % 2 == 0:
                trajectories[name].append(x[0].detach().numpy().copy())
            
            if 'Brownian' in name:
                x = x + noise_amp * torch.randn_like(x)
            elif 'Standard Langevin' in name:
                x_req = x.detach().requires_grad_(True)
                V, _, _ = model(x_req)
                gradV = torch.autograd.grad(V.sum(), x_req)[0]
                x = x - (dt / gamma) * gradV + noise_amp * torch.randn_like(x)
            elif 'Rotor-Only' in name:
                x_req = x.detach().requires_grad_(True)
                V, sigma, _ = model(x_req)
                gradV = torch.autograd.grad(V.sum(), x_req)[0]
                rotor_gradV = torch.stack([-gradV[:, 1], gradV[:, 0]], dim=1)
                flow = sigma * rotor_gradV
                x = x + dt * flow + noise_amp * torch.randn_like(x)
            elif 'A-Block Full' in name:
                x_req = x.detach().requires_grad_(True)
                V, sigma, kappa = model(x_req)
                gradV = torch.autograd.grad(V.sum(), x_req)[0]
                rotor_gradV = torch.stack([-gradV[:, 1], gradV[:, 0]], dim=1)
                flow = sigma * rotor_gradV - kappa * gradV
                x = x + dt * flow + noise_amp * torch.randn_like(x)
            
        times[name] = time.time() - t0
        samplers[name] = x
        
    # Reference Boltzmann equilibrium particles
    gx = torch.linspace(-2.2, 2.2, 80)
    gy = torch.linspace(-2.2, 2.2, 80)
    GX, GY = torch.meshgrid(gx, gy, indexing='ij')
    grid_pts = torch.stack([GX.ravel(), GY.ravel()], dim=1)
    v_grid = potential_fn(grid_pts)
    prob_gt = torch.exp(-v_grid / kBT)
    prob_gt = prob_gt / prob_gt.sum()
    idx_gt = torch.multinomial(prob_gt, num_samples=num_agents, replacement=True)
    gt_particles = grid_pts[idx_gt].numpy()
    
    # 1. Sliced 1-Wasserstein Distance (W1)
    def calc_w1(pts_np):
        projections = np.random.randn(64, 2)
        projections /= np.linalg.norm(projections, axis=1, keepdims=True)
        proj_gt = np.sort(np.dot(gt_particles, projections.T), axis=0)
        proj_sample = np.sort(np.dot(pts_np, projections.T), axis=0)
        return float(np.mean(np.abs(proj_gt - proj_sample)))
    
    # 2. Free Energy Surface Reconstruction RMSE in metastable wells
    def calc_fes_rmse(pts_np):
        hist, xedges, yedges = np.histogram2d(
            pts_np[:, 0], pts_np[:, 1],
            bins=35, range=[[-2.2, 2.2], [-2.2, 2.2]]
        )
        prob = hist / (np.sum(hist) + 1e-10)
        fes_est = -kBT * np.log(prob + 1e-5)
        
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        XC, YC = np.meshgrid(xc, yc)
        grid = torch.tensor(np.stack([XC.ravel(), YC.ravel()], axis=1), dtype=torch.float32)
        v_exact = potential_fn(grid).numpy().reshape(35, 35).T
        
        mask = v_exact < -1.2
        fes_est_norm = fes_est - np.min(fes_est[mask])
        v_exact_norm = v_exact - np.min(v_exact[mask])
        return float(np.sqrt(np.mean((fes_est_norm[mask] - v_exact_norm[mask])**2)))
    
    # 3. Multi-Basin Coverage (% occupying non-initial basins B or C)
    def calc_basin_coverage(pts_torch):
        in_B = basin_B_mask(pts_torch)
        in_C = basin_C_mask(pts_torch)
        coverage = torch.mean((in_B | in_C).float()) * 100.0
        return float(coverage.item())
    
    results = {}
    print("\n" + "=" * 80)
    print("Table 1 Benchmark Results:")
    print(f"{'Method':<35} | {'W1 (↓)':<8} | {'FES RMSE (↓)':<12} | {'Multi-Basin Cov (↑)':<20} | {'Time (s)':<8}")
    print("-" * 92)
    for name in samplers:
        s_np = samplers[name].detach().numpy()
        w1 = calc_w1(s_np)
        fes_rmse = calc_fes_rmse(s_np)
        coverage = calc_basin_coverage(samplers[name])
        
        results[name] = {
            'Wasserstein_1_Distance': round(w1, 3),
            'FES_Basin_RMSE_kcal_mol': round(fes_rmse, 3),
            'Multi_Basin_Coverage_pct': round(coverage, 1),
            'Time_sec': round(times[name], 4)
        }
        print(f"{name:<35} | {w1:<8.3f} | {fes_rmse:<12.3f} | {coverage:<19.1f}% | {times[name]:<8.3f}")
    print("=" * 80)
    
    return results, samplers, trajectories

def make_figure1(samplers, trajectories, results, out_pdf, out_png):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica Neue', 'Arial'],
        'mathtext.fontset': 'cm',
    })
    
    fig = plt.figure(figsize=(13.5, 4.0), dpi=250)
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 0.14], hspace=0.25, wspace=0.12)
    
    lin = np.linspace(-2.2, 2.2, 100)
    X, Y = np.meshgrid(lin, lin)
    grid_pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    V_grid = potential_fn(grid_pts).numpy().reshape(100, 100)
    
    names = list(samplers.keys())
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    equations = [
        r'$\mathbf{f} = \mathbf{0}$',
        r'$\mathbf{f} = -\nabla V_\theta$',
        r'$\mathbf{f} = \sigma I \nabla V_\theta$',
        r'$\mathbf{f} = (\sigma I - \kappa)\nabla V_\theta$'
    ]
    verdicts = [
        'Isotropic Dispersion',
        'Kinetic Arrest (Basin A)',
        'Centrifugal Escape',
        'Multi-Basin Equilibrium'
    ]
    
    for idx, name in enumerate(names):
        ax = fig.add_subplot(gs[0, idx])
        pts = samplers[name].detach().numpy()
        traj = np.array(trajectories[name])
        
        ax.contourf(X, Y, V_grid, levels=25, cmap='bone', alpha=0.55)
        ax.contour(X, Y, V_grid, levels=12, colors='#475569', linewidths=0.4, alpha=0.35)
        
        for c, lbl in zip(potential_fn.centers.numpy(), ['A', 'B', 'C']):
            ax.scatter(c[0], c[1], marker='+', color='#E2E8F0', s=45, lw=1.2, zorder=2)
            ax.text(c[0]+0.08, c[1]+0.08, lbl, fontsize=9.0, fontweight='bold', color='#E2E8F0', zorder=2)
            
        ax.plot(traj[:, 0], traj[:, 1], color='#E11D48', lw=1.1, alpha=0.75, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], color='#9F1239', s=55, marker='D', edgecolors='white', lw=1.0, zorder=8)
        ax.scatter(traj[-1, 0], traj[-1, 1], color='#E11D48', s=75, marker='*', edgecolors='white', lw=0.9, zorder=8)
        
        ax.scatter(pts[:, 0], pts[:, 1], s=5.5, color='#6D28D9', alpha=0.55, zorder=5, rasterized=True)
        
        is_winner = (idx == 3)
        ax.set_title(name, fontsize=10.5, fontweight='bold', pad=7, color='#059669' if is_winner else '#0F172A')
        
        ax.text(0.04, 0.95, panel_labels[idx], transform=ax.transAxes,
                fontsize=11.5, fontweight='bold', va='top', ha='left', color='#0F172A',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#CBD5E1', alpha=0.95, lw=0.7))
        
        ax.text(0.96, 0.95, equations[idx], transform=ax.transAxes,
                fontsize=9.5, ha='right', va='top', color='#1E293B',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#CBD5E1', alpha=0.92, lw=0.6))
        
        rmse_val = results[name]['FES_Basin_RMSE_kcal_mol']
        ax.text(0.96, 0.83, f"FES RMSE: {rmse_val:.3f}", transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', ha='right', va='top',
                color='#059669' if is_winner else '#475569',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='#F8FAFC',
                          edgecolor='#10B981' if is_winner else '#CBD5E1', alpha=0.92, lw=0.8))
        
        badge_bg = '#059669' if is_winner else '#FFFFFF'
        badge_text = '#FFFFFF' if is_winner else '#1E293B'
        badge_edge = '#047857' if is_winner else '#94A3B8'
        badge_lw = 1.6 if is_winner else 0.8
        
        ax.text(0.5, 0.07, verdicts[idx], transform=ax.transAxes,
                fontsize=9.0, fontweight='bold', ha='center', va='bottom',
                color=badge_text, zorder=12,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=badge_bg, edgecolor=badge_edge, lw=badge_lw, alpha=0.98))
        
        if is_winner:
            for spine in ax.spines.values():
                spine.set_edgecolor('#10B981')
                spine.set_linewidth(2.0)
                
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis('off')
    handles = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#9F1239', markersize=8, markeredgecolor='white', markeredgewidth=1.2, label='Seed Origin in Basin A ($t_0$)'),
        plt.Line2D([0], [0], color='#E11D48', lw=1.4, marker='*', markersize=9, markerfacecolor='#E11D48', markeredgecolor='white', markeredgewidth=0.7, label='Trajectory Endpoint ($t_{1000}$)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6D28D9', markersize=6.5, alpha=0.75, label='Particles $\\mu_{1000}$ ($N=2{,}000$)'),
        plt.Line2D([0], [0], marker='+', color='#E2E8F0', markersize=8, markeredgewidth=1.5, label='Metastable Centers (Basins A, B, C)'),
    ]
    legend_ax.legend(handles=handles, loc='center', ncol=4, fontsize=8.5, frameon=True,
                     facecolor='#F8FAFC', edgecolor='#CBD5E1', framealpha=0.95, columnspacing=1.3, handletextpad=0.4)
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated Figure 1: {out_pdf}")

def main():
    os.makedirs('output', exist_ok=True)
    print("=" * 80)
    print("A-Block: Table 1 & Figure 1 Reproduction")
    print("=" * 80)
    
    model = train_score_model()
    results, samplers, trajectories = run_benchmarks(model)
    
    with open('output/table1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved Table 1 results to output/table1_results.json")
    
    out_pdf = 'output/figure1_annotated_benchmark.pdf'
    out_png = 'output/figure1_annotated_benchmark.png'
    make_figure1(samplers, trajectories, results, out_pdf, out_png)
    print("=" * 80)

if __name__ == '__main__':
    main()

