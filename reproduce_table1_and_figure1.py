#!/usr/bin/env python3
"""
A-Block: Table 1, Multi-Seed Ablations & Figure 1 Reproduction Script
Simulates multi-well conformational dynamics across 5 sampling configurations:
  1. Brownian Dynamics (f = 0)
  2. Standard Langevin (f = -grad V_theta)
  3. Dilator-Only Ablation (f = -kappa * grad V_theta, sigma = 0) [Matched Contraction]
  4. Rotor-Only Ablation (f = sigma * I * grad V_theta, kappa = 0)
  5. Full Clifford A-Block (f = (sigma * I - kappa) * grad V_theta)

Generates:
  - output/table1_results.json (Canonical seed-42 & 5-seed mean ± std + sensitivity checks)
  - output/figure1_annotated_benchmark.pdf / .png
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.potentials import MolecularMultiWellPotential
from src.training import train_score_model
from src.samplers import simulate_multiwell_ensemble
from src.metrics import calc_sliced_w1, calc_fes_rmse, calc_basin_coverage

potential_fn = MolecularMultiWellPotential()


def run_canonical_and_multiseed(model, num_agents=2000, num_steps=1000, dt=0.015, kBT=0.6):
    # 1. Exact Canonical Run (seed=42 sequence matching original Table 1)
    gamma = 1.0
    x_init_canonical = torch.tensor([-1.0, 1.0]).repeat(num_agents, 1) + 0.15 * torch.randn(num_agents, 2)

    canonical_order = [
        ('Brownian Dynamics (f = 0)', 'brownian'),
        ('Standard Langevin (f = -\\nabla V)', 'langevin'),
        ('Rotor-Only (\\kappa = 0)', 'rotor_only'),
        ('A-Block Full Clifford (Ours)', 'ablock')
    ]

    samplers = {}
    trajectories = {}
    times = {}
    extra_stats = {}

    for name, m_code in canonical_order:
        x_f, traj, t_sec, st = simulate_multiwell_ensemble(
            m_code, model, x_init_canonical, num_steps=num_steps, dt=dt, kBT=kBT, gamma=gamma
        )
        samplers[name] = x_f
        trajectories[name] = traj
        times[name] = t_sec
        extra_stats[name] = st

    # Reference Boltzmann equilibrium particles (drawn right after the 4 canonical runs)
    gx = torch.linspace(-2.2, 2.2, 80)
    gy = torch.linspace(-2.2, 2.2, 80)
    GX, GY = torch.meshgrid(gx, gy, indexing='ij')
    grid_pts = torch.stack([GX.ravel(), GY.ravel()], dim=1)
    v_grid = potential_fn(grid_pts)
    prob_gt = torch.exp(-v_grid / kBT)
    prob_gt = prob_gt / prob_gt.sum()
    idx_gt = torch.multinomial(prob_gt, num_samples=num_agents, replacement=True)
    gt_particles = grid_pts[idx_gt].numpy()

    # Evaluate W1, FES RMSE, Coverage for the 4 canonical methods in exact order
    canonical_metrics = {}
    for name, _ in canonical_order:
        s_np = samplers[name].detach().numpy()
        w1 = calc_sliced_w1(s_np, gt_particles)
        fes_rmse = calc_fes_rmse(s_np, potential_fn, kBT=kBT, bins=35)
        cov = calc_basin_coverage(samplers[name])
        canonical_metrics[name] = (w1, fes_rmse, cov)

    # Now run the Dilator-Only (Matched Contraction: sigma=0, kappa) ablation
    d_name = 'Dilator-Only (\\sigma = 0, \\kappa)'
    x_f, traj, t_sec, st = simulate_multiwell_ensemble(
        'dilator_only', model, x_init_canonical, num_steps=num_steps, dt=dt, kBT=kBT, gamma=gamma
    )
    samplers[d_name] = x_f
    trajectories[d_name] = traj
    times[d_name] = t_sec
    extra_stats[d_name] = st
    s_np = x_f.detach().numpy()
    canonical_metrics[d_name] = (
        calc_sliced_w1(s_np, gt_particles),
        calc_fes_rmse(s_np, potential_fn, kBT=kBT, bins=35),
        calc_basin_coverage(x_f)
    )

    # 2. Multi-Seed Uncertainty Evaluation (5 independent seeds)
    all_methods = [
        ('Brownian Dynamics (f = 0)', 'brownian'),
        ('Standard Langevin (f = -\\nabla V)', 'langevin'),
        ('Dilator-Only (\\sigma = 0, \\kappa)', 'dilator_only'),
        ('Rotor-Only (\\kappa = 0)', 'rotor_only'),
        ('A-Block Full Clifford (Ours)', 'ablock')
    ]
    seeds = [42, 43, 44, 45, 46]
    multiseed = {name: {'w1': [], 'rmse35': [], 'rmse25': [], 'rmse50': [], 'cov': [], 'vis': [], 'trans': [], 'time': []} for name, _ in all_methods}

    for sd in seeds:
        torch.manual_seed(sd)
        np.random.seed(sd)
        x_init_s = torch.tensor([-1.0, 1.0]).repeat(num_agents, 1) + 0.15 * torch.randn(num_agents, 2)
        idx_s = torch.multinomial(prob_gt, num_samples=num_agents, replacement=True)
        gt_s = grid_pts[idx_s].numpy()
        for name, m_code in all_methods:
            torch.manual_seed(sd + 100)
            np.random.seed(sd + 100)
            xf_s, _, t_s, st_s = simulate_multiwell_ensemble(
                m_code, model, x_init_s, num_steps=num_steps, dt=dt, kBT=kBT, gamma=gamma
            )
            pts_s = xf_s.detach().numpy()
            multiseed[name]['w1'].append(calc_sliced_w1(pts_s, gt_s))
            multiseed[name]['rmse35'].append(calc_fes_rmse(pts_s, potential_fn, kBT=kBT, bins=35))
            multiseed[name]['rmse25'].append(calc_fes_rmse(pts_s, potential_fn, kBT=kBT, bins=25))
            multiseed[name]['rmse50'].append(calc_fes_rmse(pts_s, potential_fn, kBT=kBT, bins=50))
            multiseed[name]['cov'].append(calc_basin_coverage(xf_s))
            multiseed[name]['vis'].append(st_s['visited_bc_pct'])
            multiseed[name]['trans'].append(st_s['transitions_per_particle'])
            multiseed[name]['time'].append(t_s)

    results = {}
    print("\n" + "=" * 115)
    print("Table 1 Benchmark Results (Canonical Seed-42 & 5-Seed Mean ± Std):")
    print(f"{'Method':<34} | {'W1 (↓)':<13} | {'FES RMSE (↓)':<14} | {'Term B/C (%)':<14} | {'Visit B/C (%)':<14} | {'Trans/pt':<11}")
    print("-" * 115)
    for name, _ in all_methods:
        w1_c, rmse_c, cov_c = canonical_metrics[name]
        ms = multiseed[name]
        results[name] = {
            'Wasserstein_1_Distance': round(w1_c, 3),
            'Wasserstein_1_Mean_Std': f"{w1_c:.3f} ± {np.std(ms['w1']):.3f}",
            'FES_Basin_RMSE_kcal_mol': round(rmse_c, 3),
            'FES_Basin_RMSE_Mean_Std': f"{rmse_c:.3f} ± {np.std(ms['rmse35']):.3f}",
            'FES_RMSE_bins25': round(float(np.mean(ms['rmse25'])), 3),
            'FES_RMSE_bins50': round(float(np.mean(ms['rmse50'])), 3),
            'Multi_Basin_Coverage_pct': round(cov_c, 1),
            'Multi_Basin_Coverage_Mean_Std': f"{cov_c:.1f} ± {np.std(ms['cov']):.1f}%",
            'Trajectory_Visitation_BC_pct': round(float(extra_stats[name]['visited_bc_pct']), 1),
            'Transitions_per_particle': round(float(extra_stats[name]['transitions_per_particle']), 2),
            'Transitions_Mean_Std': f"{extra_stats[name]['transitions_per_particle']:.2f} ± {np.std(ms['trans']):.2f}",
            'Time_sec': round(times[name], 4)
        }
        print(
            f"{name:<34} | {w1_c:.3f}±{np.std(ms['w1']):.3f} | "
            f"{rmse_c:.3f}±{np.std(ms['rmse35']):.3f}  | "
            f"{cov_c:4.1f}±{np.std(ms['cov']):.1f}%    | "
            f"{extra_stats[name]['visited_bc_pct']:5.1f}±{np.std(ms['vis']):.1f}%   | "
            f"{extra_stats[name]['transitions_per_particle']:5.2f}±{np.std(ms['trans']):.2f}"
        )
    print("=" * 115)

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
    
    names = [
        'Brownian Dynamics (f = 0)',
        'Standard Langevin (f = -\\nabla V)',
        'Rotor-Only (\\kappa = 0)',
        'A-Block Full Clifford (Ours)'
    ]
    display_titles = [
        'Brownian Dynamics',
        'Standard Langevin',
        'Rotor-Only (κ = 0)',
        'A-Block Full Clifford (Ours)'
    ]
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    equations = [
        r'$\mathbf{f} = \mathbf{0}$',
        r'$\mathbf{f} = -\nabla V_\theta$',
        r'$\mathbf{f} = \sigma I \nabla V_\theta$',
        r'$\mathbf{f} = (\sigma I - \kappa)\nabla V_\theta$'
    ]
    verdicts = [
        'Isotropic Dispersion',
        'Mode Over-Concentration',
        'Centrifugal Escape',
        'Active Saddle Circulation'
    ]
    
    for idx, name in enumerate(names):
        ax = fig.add_subplot(gs[0, idx])
        pts = samplers[name].detach().numpy()
        traj = np.array(trajectories[name])
        
        ax.contourf(X, Y, V_grid, levels=25, cmap='bone', alpha=0.55)
        ax.contour(X, Y, V_grid, levels=12, colors='#475569', linewidths=0.4, alpha=0.35)
        ax.contour(X, Y, V_grid, levels=[-1.2], colors='#F59E0B', linewidths=0.9, linestyles='solid', alpha=0.75, zorder=2)
        
        r_basin = float(np.sqrt(0.45))
        for c, lbl in zip(potential_fn.centers.numpy(), ['A', 'B', 'C']):
            basin_circle = plt.Circle((c[0], c[1]), r_basin, color='#E2E8F0', fill=False, linestyle='--', linewidth=0.95, alpha=0.75, zorder=2)
            ax.add_patch(basin_circle)
            ax.scatter(c[0], c[1], marker='+', color='#E2E8F0', s=45, lw=1.2, zorder=2)
            ax.text(c[0]+0.08, c[1]+0.08, lbl, fontsize=9.0, fontweight='bold', color='#E2E8F0', zorder=2)
            
        ax.plot(traj[:, 0], traj[:, 1], color='#E11D48', lw=1.1, alpha=0.75, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], color='#9F1239', s=55, marker='D', edgecolors='white', lw=1.0, zorder=8)
        ax.scatter(traj[-1, 0], traj[-1, 1], color='#E11D48', s=75, marker='*', edgecolors='white', lw=0.9, zorder=8)
        
        ax.scatter(pts[:, 0], pts[:, 1], s=5.5, color='#6D28D9', alpha=0.55, zorder=5, rasterized=True)
        
        is_winner = (idx == 3)
        ax.set_title(display_titles[idx], fontsize=10.5, fontweight='bold', pad=7, color='#059669' if is_winner else '#0F172A')
        
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
        plt.Line2D([0], [0], color='#E2E8F0', lw=1.1, ls='--', marker='+', markersize=8, markeredgewidth=1.4, label='Basin Cores $\\mathcal{B}_{A,B,C}$ ($r^2<0.45$)'),
        plt.Line2D([0], [0], color='#F59E0B', lw=1.2, label='Saddle Envelope ($V=-1.2$)'),
    ]
    legend_ax.legend(handles=handles, loc='center', ncol=5, fontsize=8.2, frameon=True,
                     facecolor='#F8FAFC', edgecolor='#CBD5E1', framealpha=0.95, columnspacing=1.1, handletextpad=0.4)
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated Figure 1: {out_pdf}")


def main():
    os.makedirs('output', exist_ok=True)
    print("=" * 80)
    print("A-Block: Table 1, Multi-Seed Ablations & Figure 1 Reproduction")
    print("=" * 80)
    
    model = train_score_model(potential_fn, num_samples=12000, epochs=250, seed=42)
    torch.save(model.state_dict(), 'output/clifford_score_net.pt')
    
    results, samplers, trajectories = run_canonical_and_multiseed(model)
    
    with open('output/table1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved Table 1 results to output/table1_results.json")
    
    out_pdf = 'output/figure1_annotated_benchmark.pdf'
    out_png = 'output/figure1_annotated_benchmark.png'
    make_figure1(samplers, trajectories, results, out_pdf, out_png)
    print("=" * 80)

if __name__ == '__main__':
    main()
