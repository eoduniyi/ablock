#!/usr/bin/env python3
"""
Generate publication-quality 1x4 comparative benchmark figure and JSON metrics
for the Typography task on the upright non-convex letter 'G'
(Topological Navigation & Kinetic Escape).

Standardized across all methods:
  - Identical circular aperture boundary condition (r_dish = 0.44 * min(H, W))
  - Identical stochastic noise amplitude (noise_amp = 0.85, dt = 2.0)
  - Reports In-Glyph Precision (%), Exact Pixel Trajectory Visitation (%),
    and Terminal Footprint Coverage (%).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.potentials import build_glyph_potential

np.random.seed(42)

def run_typography_simulation(V, mask, dish_info, N=1200, T=4000):
    H, W = V.shape
    cx, cy, r_dish = dish_info
    gy, gx = np.gradient(V)

    # Inner spur of upright 'G' is in the center-right (x=230, y=205 for 400x400)
    seed_x, seed_y = int(W * 0.575), int(H * 0.512)

    configs = [
        ('Brownian Dynamics', 0.0, 0.0),
        ('Standard Langevin', 0.0, 1.0),
        ('Rotor-Only (κ = 0)', 11.0, 0.0),
        ('A-Block Full Clifford (Ours)', 11.0, 1.8)
    ]

    samplers = {}
    trajectories = {name: [] for name, _, _ in configs}
    metrics = {}

    dt = 2.0
    noise_amp = 0.85

    for name, sig, kap in configs:
        np.random.seed(42)
        p = np.random.randn(N, 2) * 3.0 + np.array([seed_x, seed_y])
        chir = np.random.choice([-1.0, 1.0], size=N)
        visited_exact = np.zeros_like(mask)

        for t in range(T):
            if t % 5 == 0:
                trajectories[name].append(p[0].copy())
            ix = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)
            iy = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
            visited_exact[iy, ix] = 1.0

            gvx = gx[iy, ix]
            gvy = gy[iy, ix]
            drift_x = sig * (-gvy * chir) - kap * gvx
            drift_y = sig * (gvx * chir) - kap * gvy
            p += dt * np.stack([drift_x, drift_y], axis=1) + np.random.randn(N, 2) * noise_amp

            # Standardized circular aperture confinement across all 4 methods
            dx = p[:, 0] - cx
            dy = p[:, 1] - cy
            d = np.sqrt(dx**2 + dy**2)
            out = d > (r_dish - 2.0)
            if np.any(out):
                p[out, 0] = cx + (dx[out] / d[out]) * (r_dish - 3.0)
                p[out, 1] = cy + (dy[out] / d[out]) * (r_dish - 3.0)

        samplers[name] = p

        ix = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
        prec = mask[iy, ix].mean() * 100.0

        # Terminal coverage (r=8px particle footprint)
        term_grid = np.zeros_like(mask)
        for px, py in zip(ix, iy):
            if mask[py, px] > 0.5:
                term_grid[max(0, py - 8):min(H, py + 9), max(0, px - 8):min(W, px + 9)] = 1.0
        term_cov = (term_grid * mask).sum() / mask.sum() * 100.0

        # Cumulative trajectory visitation (exact pixel visitation over t in [0, T])
        vis_cov = (visited_exact * mask).sum() / mask.sum() * 100.0

        metrics[name] = {
            'precision_pct': round(float(prec), 1),
            'trajectory_visitation_pct': round(float(vis_cov), 1),
            'terminal_coverage_pct': round(float(term_cov), 1)
        }
        print(
            f"[{name:<28}] Precision: {prec:5.1f}% | "
            f"Trajectory Visitation: {vis_cov:5.1f}% | Terminal Coverage: {term_cov:5.1f}%"
        )

    return samplers, trajectories, metrics, (seed_x, seed_y)


def make_typography_figure(V, mask, dish_info, samplers, trajectories, metrics, seed_pt, out_pdf, out_png):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica Neue', 'Arial'],
        'mathtext.fontset': 'cm',
    })

    H, W = V.shape
    cx, cy, r_dish = dish_info
    seed_x, seed_y = seed_pt

    fig = plt.figure(figsize=(13.5, 4.0), dpi=250)
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 0.14], hspace=0.25, wspace=0.12)

    names = [
        'Brownian Dynamics',
        'Standard Langevin',
        'Rotor-Only (κ = 0)',
        'A-Block Full Clifford (Ours)'
    ]
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    equations = [
        r'$\mathbf{f} = \mathbf{0}$',
        r'$\mathbf{f} = -\nabla V$',
        r'$\mathbf{f} = \sigma I \nabla V$',
        r'$\mathbf{f} = (\sigma I - \kappa)\nabla V$',
    ]
    verdicts = [
        'Isotropic Dispersion',
        'Kinetic Arrest (Spur)',
        'Centrifugal Leakage',
        'Active Confinement',
    ]

    for idx, name in enumerate(names):
        ax = fig.add_subplot(gs[0, idx])
        pts = samplers[name]
        traj = np.array(trajectories[name])

        ax.contourf(V, levels=25, cmap='bone', alpha=0.55)
        ax.contour(V, levels=12, colors='#475569', linewidths=0.4, alpha=0.35)
        ax.contour(mask, levels=[0.5], colors='#94A3B8', linewidths=1.2, alpha=0.85)
        dish_circle = plt.Circle((cx, cy), r_dish, fill=False, edgecolor='#64748B', lw=1.2, ls='--', alpha=0.7)
        ax.add_patch(dish_circle)

        ax.plot(traj[:, 0], traj[:, 1], color='#E11D48', lw=1.1, alpha=0.75, zorder=3)
        ax.scatter(seed_x, seed_y, color='#9F1239', s=55, marker='D', edgecolors='white', lw=1.0, zorder=8)
        ax.scatter(traj[-1, 0], traj[-1, 1], color='#E11D48', s=75, marker='*', edgecolors='white', lw=0.9, zorder=8)

        ax.scatter(pts[:, 0], pts[:, 1], s=5.5, color='#6D28D9', alpha=0.55, zorder=5, rasterized=True)

        is_winner = (idx == 3)
        ax.set_title(names[idx], fontsize=11, fontweight='bold', pad=7, color='#059669' if is_winner else '#0F172A')

        ax.text(0.04, 0.95, panel_labels[idx], transform=ax.transAxes,
                fontsize=11.5, fontweight='bold', va='top', ha='left', color='#0F172A',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#CBD5E1', alpha=0.95, lw=0.7))

        ax.text(0.96, 0.95, equations[idx], transform=ax.transAxes,
                fontsize=9.5, ha='right', va='top', color='#1E293B',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#CBD5E1', alpha=0.92, lw=0.6))

        vis_val = metrics[name]['trajectory_visitation_pct']
        prec_val = metrics[name]['precision_pct']
        ax.text(0.96, 0.83, f"Visit: {vis_val:.1f}% | Prec: {prec_val:.1f}%", transform=ax.transAxes,
                fontsize=7.8, fontweight='bold', ha='right', va='top',
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

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis('off')
    handles = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#9F1239', markersize=8, markeredgecolor='white', markeredgewidth=1.2, label=r'Seed Origin in Spur ($t_0$)'),
        plt.Line2D([0], [0], color='#E11D48', lw=1.4, marker='*', markersize=9, markerfacecolor='#E11D48', markeredgecolor='white', markeredgewidth=0.7, label=r'Trajectory Endpoint ($t_{4000}$)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6D28D9', markersize=6.5, alpha=0.75, label=r'Particles $\mu_{4000}$ ($N=1{,}200$)'),
        plt.Line2D([0], [0], color='#64748B', lw=1.2, ls='--', label='Aperture Boundary'),
        plt.Line2D([0], [0], color='#94A3B8', lw=1.2, label='Target Glyph Boundary'),
    ]
    legend_ax.legend(handles=handles, loc='center', ncol=5, fontsize=8.5, frameon=True,
                     facecolor='#F8FAFC', edgecolor='#CBD5E1', framealpha=0.95, columnspacing=1.3, handletextpad=0.4)

    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated Typography benchmark figure: {out_pdf}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)

    V, mask, dish_info = build_glyph_potential(resolution=400, char='G')
    samplers, trajectories, metrics, seed_pt = run_typography_simulation(V, mask, dish_info, N=1200, T=4000)

    json_path = os.path.join(out_dir, 'typography_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved Typography metrics to {json_path}")

    out_pdf = os.path.join(out_dir, 'figure3_typography_benchmark.pdf')
    out_png = os.path.join(out_dir, 'figure3_typography_benchmark.png')
    make_typography_figure(V, mask, dish_info, samplers, trajectories, metrics, seed_pt, out_pdf, out_png)

if __name__ == '__main__':
    main()
