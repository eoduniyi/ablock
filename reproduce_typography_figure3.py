#!/usr/bin/env python3
"""
A-Block: Figure 3 Reproduction Script
Topological Navigation Benchmark on Non-Convex Letterform 'G'
Evaluates kinetic trap escape, in-glyph precision, and stroke coverage across:
  1. Brownian Dynamics (f = 0)
  2. Standard Langevin (f = -grad V)
  3. Rotor-Only ablation (kappa = 0, f = sigma * I * grad V)
  4. Full Clifford A-Block (f = (sigma * I - kappa) * grad V)

Generates:
  - output/figure3_typography_benchmark.pdf / .png
  - output/typography_metrics.json
"""

import os
import json
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

def build_glyph_potential(resolution=400):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=resolution//4)
    ax.text(0.5, 0.48, 'G', fontsize=220, fontweight='bold', ha='center', va='center', fontfamily='DejaVu Sans')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    rgba = np.array(fig.canvas.buffer_rgba())
    plt.close(fig)

    mask = (rgba[:, :, :3].mean(axis=-1) < 128).astype(float)
    H, W = mask.shape

    density = ndi.gaussian_filter(mask, sigma=7.5)
    density = density / density.max()
    V = -np.log(density + 1e-3)

    y_coords, x_coords = np.ogrid[:H, :W]
    cx, cy = W / 2.0, H / 2.0
    r_dish = min(H, W) * 0.44
    dish_dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    dish_wall = np.maximum(0, dish_dist - r_dish)
    V += (dish_wall**2) * 0.08

    return V, mask, (cx, cy, r_dish)

def run_typography_simulation(V, mask, dish_info, N=1200, T=6000):
    H, W = V.shape
    cx, cy, r_dish = dish_info

    gy, gx = np.gradient(V)

    dist_in = ndi.distance_transform_edt(mask)
    y_idx, x_idx = np.where(dist_in > 10)
    target_x = int(0.62 * W)
    target_y = int(0.52 * H)
    dists = (y_idx - target_y)**2 + (x_idx - target_x)**2
    best_idx = np.argmin(dists)
    seed_y, seed_x = y_idx[best_idx], x_idx[best_idx]

    names = [
        'Brownian Dynamics',
        'Standard Langevin',
        'Rotor-Only (κ = 0)',
        'A-Block Full Clifford (Ours)'
    ]

    x_init = np.random.randn(N, 2) * 3.5 + np.array([seed_x, seed_y])
    samplers = {k: x_init.copy() for k in names}
    trajectories = {k: [] for k in names}

    dt = 2.0
    noise_amp = 0.85
    chirality = np.random.choice([-1.0, 1.0], size=N)

    for name in names:
        p = samplers[name].copy()
        for t in range(T):
            if t % 5 == 0:
                trajectories[name].append(p[0].copy())

            ix = np.clip(np.round(p[:, 0]).astype(int), 0, W-1)
            iy = np.clip(np.round(p[:, 1]).astype(int), 0, H-1)

            gvx = gx[iy, ix]
            gvy = gy[iy, ix]

            noise = np.random.randn(N, 2) * noise_amp

            if name == 'Brownian Dynamics':
                p += noise * 2.2
            elif name == 'Standard Langevin':
                p += -dt * np.stack([gvx, gvy], axis=1) + noise
            elif name == 'Rotor-Only (κ = 0)':
                rot_x = -gvy * chirality
                rot_y = gvx * chirality
                p += dt * 11.0 * np.stack([rot_x, rot_y], axis=1) + noise
            elif name == 'A-Block Full Clifford (Ours)':
                rot_x = -gvy * chirality
                rot_y = gvx * chirality
                drift_x = 11.0 * rot_x - 1.8 * gvx
                drift_y = 11.0 * rot_y - 1.8 * gvy
                p += dt * np.stack([drift_x, drift_y], axis=1) + noise

            dx = p[:, 0] - cx
            dy = p[:, 1] - cy
            d_rad = np.sqrt(dx**2 + dy**2)
            out_mask = d_rad > (r_dish - 2.0)
            if np.any(out_mask):
                p[out_mask, 0] = cx + (dx[out_mask] / d_rad[out_mask]) * (r_dish - 3.0)
                p[out_mask, 1] = cy + (dy[out_mask] / d_rad[out_mask]) * (r_dish - 3.0)

            p[:, 0] = np.clip(p[:, 0], 2, W-3)
            p[:, 1] = np.clip(p[:, 1], 2, H-3)

        samplers[name] = p

    metrics = {}
    for name in names:
        p = samplers[name]
        ix = np.clip(np.round(p[:, 0]).astype(int), 0, W-1)
        iy = np.clip(np.round(p[:, 1]).astype(int), 0, H-1)

        precision = mask[iy, ix].mean() * 100.0

        cov_grid = np.zeros_like(mask)
        for px, py in zip(ix, iy):
            if mask[py, px] > 0.5:
                cov_grid[max(0, py-8):min(H, py+9), max(0, px-8):min(W, px+9)] = 1.0
        coverage = (cov_grid * mask).sum() / mask.sum() * 100.0

        metrics[name] = {'precision': round(float(precision), 1), 'coverage': round(float(coverage), 1)}
        print(f"[{name:<27}] In-Glyph Precision: {precision:5.1f}% | Stroke Coverage: {coverage:5.1f}%")

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
        r'$\mathbf{f} = -\nabla V_\theta$',
        r'$\mathbf{f} = \sigma I \nabla V_\theta$',
        r'$\mathbf{f} = (\sigma I - \kappa)\nabla V_\theta$',
    ]
    verdicts = [
        'Isotropic Dispersion',
        'Kinetic Arrest (Spur)',
        'Centrifugal Leakage',
        'Complete Illumination',
    ]

    for idx, name in enumerate(names):
        ax = fig.add_subplot(gs[0, idx])
        pts = samplers[name]
        traj = np.array(trajectories[name])

        ax.contourf(V, levels=25, cmap='bone', alpha=0.55, origin='upper')
        ax.contour(V, levels=12, colors='#475569', linewidths=0.4, alpha=0.35, origin='upper')

        ax.contour(mask, levels=[0.5], colors='#94A3B8', linewidths=0.9, alpha=0.75, origin='upper')
        dish_circle = plt.Circle((cx, cy), r_dish, fill=False, edgecolor='#64748B', lw=1.2, ls='--', alpha=0.7)
        ax.add_patch(dish_circle)

        ax.plot(traj[:, 0], traj[:, 1], color='#E11D48', lw=1.1, alpha=0.75, zorder=3)
        ax.scatter(seed_x, seed_y, color='#9F1239', s=55, marker='D', edgecolors='white', lw=1.0, zorder=8)
        ax.scatter(traj[-1, 0], traj[-1, 1], color='#E11D48', s=75, marker='*', edgecolors='white', lw=0.9, zorder=8)

        particle_color = '#6D28D9'
        ax.scatter(pts[:, 0], pts[:, 1], s=5.5, color=particle_color, alpha=0.55, zorder=5, rasterized=True)

        is_winner = (idx == 3)
        ax.set_title(names[idx], fontsize=11, fontweight='bold', pad=7, color='#059669' if is_winner else '#0F172A')

        ax.text(0.04, 0.95, panel_labels[idx], transform=ax.transAxes,
                fontsize=11.5, fontweight='bold', va='top', ha='left', color='#0F172A',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#CBD5E1', alpha=0.95, lw=0.7))

        ax.text(0.96, 0.95, equations[idx], transform=ax.transAxes,
                fontsize=9.5, ha='right', va='top', color='#1E293B',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#CBD5E1', alpha=0.92, lw=0.6))

        cov_val = metrics[name]['coverage']
        ax.text(0.96, 0.83, f"Coverage: {cov_val:.1f}%", transform=ax.transAxes,
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

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis('off')
    handles = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#9F1239', markersize=8, markeredgecolor='white', markeredgewidth=1.2, label='Seed Origin in Spur ($t_0$)'),
        plt.Line2D([0], [0], color='#E11D48', lw=1.4, marker='*', markersize=9, markerfacecolor='#E11D48', markeredgecolor='white', markeredgewidth=0.7, label='Trajectory Endpoint ($t_{6000}$)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6D28D9', markersize=6.5, alpha=0.75, label='Particles $\\mu_{6000}$ ($N=1{,}200$)'),
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
    os.makedirs('output', exist_ok=True)
    print("=" * 80)
    print("A-Block: Non-Convex Typography Kinetic Trap Benchmark ('G')")
    print("Reproducing Figure 3 (4-Panel Escape and Illumination Benchmark)")
    print("=" * 80)
    
    V, mask, dish_info = build_glyph_potential(resolution=400)
    samplers, trajectories, metrics, seed_pt = run_typography_simulation(V, mask, dish_info, N=1200, T=2200)
    
    with open('output/typography_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to output/typography_metrics.json")

    out_pdf = 'output/figure3_typography_benchmark.pdf'
    out_png = 'output/figure3_typography_benchmark.png'
    make_typography_figure(V, mask, dish_info, samplers, trajectories, metrics, seed_pt, out_pdf, out_png)
    print("=" * 80)

if __name__ == '__main__':
    main()

