#!/usr/bin/env python3
"""
Generate publication-quality 1x4 comparative benchmark figure for the Typography task
on the upright non-convex letter 'G' (Topological Navigation & Kinetic Escape).
"""

import os
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

def build_glyph_potential(resolution=400):
    img = Image.new('L', (resolution, resolution), color=0)
    draw = ImageDraw.Draw(img)
    font = None
    for p in ['/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/Supplemental/Arial Bold.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']:
        try:
            font = ImageFont.truetype(p, int(resolution * 0.70))
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), 'G', font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((resolution - w)/2 - bbox[0], (resolution - h)/2 - bbox[1]), 'G', fill=255, font=font)
    mask = (np.array(img) > 128).astype(float)
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

def run_typography_simulation(V, mask, dish_info, N=1200, T=4000):
    H, W = V.shape
    cx, cy, r_dish = dish_info

    gy, gx = np.gradient(V)

    # Inner spur of upright 'G' is in the center-right (approx x=230, y=205 for 400x400)
    seed_x, seed_y = int(W * 0.575), int(H * 0.512)

    names = [
        'Brownian Dynamics',
        'Standard Langevin',
        'Rotor-Only (κ = 0)',
        'A-Block Full Clifford (Ours)'
    ]

    samplers = {}
    trajectories = {k: [] for k in names}
    metrics = {}

    dt = 2.0
    noise_amp = 0.85

    # 1. Brownian Dynamics
    p_b = np.random.randn(N, 2) * 3.0 + np.array([seed_x, seed_y])
    for t in range(T):
        if t % 5 == 0: trajectories['Brownian Dynamics'].append(p_b[0].copy())
        p_b += np.random.randn(N, 2) * noise_amp * 2.2
        dx = p_b[:, 0] - cx; dy = p_b[:, 1] - cy; d = np.sqrt(dx**2 + dy**2)
        out = d > (r_dish - 2.0)
        if np.any(out):
            p_b[out, 0] = cx + (dx[out]/d[out]) * (r_dish - 3.0)
            p_b[out, 1] = cy + (dy[out]/d[out]) * (r_dish - 3.0)
    samplers['Brownian Dynamics'] = p_b

    # 2. Standard Langevin
    p_l = np.random.randn(N, 2) * 3.0 + np.array([seed_x, seed_y])
    for t in range(T):
        if t % 5 == 0: trajectories['Standard Langevin'].append(p_l[0].copy())
        ix = np.clip(np.round(p_l[:, 0]).astype(int), 0, W-1)
        iy = np.clip(np.round(p_l[:, 1]).astype(int), 0, H-1)
        p_l += -dt * np.stack([gx[iy, ix], gy[iy, ix]], axis=1) + np.random.randn(N, 2) * noise_amp
    samplers['Standard Langevin'] = p_l

    # 3. Rotor-Only (kappa = 0)
    p_r = np.random.randn(N, 2) * 3.0 + np.array([seed_x, seed_y])
    chir_r = np.random.choice([-1.0, 1.0], size=N)
    for t in range(T):
        if t % 5 == 0: trajectories['Rotor-Only (κ = 0)'].append(p_r[0].copy())
        ix = np.clip(np.round(p_r[:, 0]).astype(int), 0, W-1)
        iy = np.clip(np.round(p_r[:, 1]).astype(int), 0, H-1)
        gvx = gx[iy, ix]; gvy = gy[iy, ix]
        p_r += dt * 11.0 * np.stack([-gvy * chir_r, gvx * chir_r], axis=1) + np.random.randn(N, 2) * noise_amp
        dx = p_r[:, 0] - cx; dy = p_r[:, 1] - cy; d = np.sqrt(dx**2 + dy**2)
        out = d > (r_dish - 2.0)
        if np.any(out):
            p_r[out, 0] = cx + (dx[out]/d[out]) * (r_dish - 3.0)
            p_r[out, 1] = cy + (dy[out]/d[out]) * (r_dish - 3.0)
    samplers['Rotor-Only (κ = 0)'] = p_r

    # 4. A-Block Full Clifford
    p_a = np.random.randn(N, 2) * 3.0 + np.array([seed_x, seed_y])
    chir_a = np.random.choice([-1.0, 1.0], size=N)
    for t in range(T):
        if t % 5 == 0: trajectories['A-Block Full Clifford (Ours)'].append(p_a[0].copy())
        ix = np.clip(np.round(p_a[:, 0]).astype(int), 0, W-1)
        iy = np.clip(np.round(p_a[:, 1]).astype(int), 0, H-1)
        gvx = gx[iy, ix]; gvy = gy[iy, ix]
        drift_x = 11.0 * (-gvy * chir_a) - 1.8 * gvx
        drift_y = 11.0 * (gvx * chir_a) - 1.8 * gvy
        p_a += dt * np.stack([drift_x, drift_y], axis=1) + np.random.randn(N, 2) * noise_amp
        dx = p_a[:, 0] - cx; dy = p_a[:, 1] - cy; d = np.sqrt(dx**2 + dy**2)
        out = d > (r_dish - 2.0)
        if np.any(out):
            p_a[out, 0] = cx + (dx[out]/d[out]) * (r_dish - 3.0)
            p_a[out, 1] = cy + (dy[out]/d[out]) * (r_dish - 3.0)
    samplers['A-Block Full Clifford (Ours)'] = p_a

    # Metrics
    for name in names:
        pts = samplers[name]
        ix = np.clip(np.round(pts[:, 0]).astype(int), 0, W-1)
        iy = np.clip(np.round(pts[:, 1]).astype(int), 0, H-1)
        prec = mask[iy, ix].mean() * 100.0
        cov_grid = np.zeros_like(mask)
        for px, py in zip(ix, iy):
            if mask[py, px] > 0.5:
                cov_grid[max(0, py-8):min(H, py+9), max(0, px-8):min(W, px+9)] = 1.0
        cov = (cov_grid * mask).sum() / mask.sum() * 100.0
        metrics[name] = {'precision': round(float(prec), 1), 'coverage': round(float(cov), 1)}
        print(f"[{name:<27}] In-Glyph Precision: {prec:5.1f}% | Stroke Coverage: {cov:5.1f}%")

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
    V, mask, dish_info = build_glyph_potential(resolution=400)
    samplers, trajectories, metrics, seed_pt = run_typography_simulation(V, mask, dish_info, N=1200, T=4000)
    
    out_pdf = os.path.join(base_dir, 'typography_kinetic_trap_benchmark_1x4.pdf')
    out_png = os.path.join(base_dir, 'typography_kinetic_trap_benchmark_1x4.png')
    make_typography_figure(V, mask, dish_info, samplers, trajectories, metrics, seed_pt, out_pdf, out_png)

if __name__ == '__main__':
    main()
