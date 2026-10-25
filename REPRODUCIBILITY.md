# Reproducing the A-Block Camera-Ready Benchmark (ML4PS @ NeurIPS 2026)

This document provides a complete walkthrough and exact instructions for reproducing the theoretical and empirical results reported in:

> **"A-Block: Active Form Dynamics via Clifford Neural Score Matching"**  
> Erick Oduniyi (Massachusetts Institute of Technology)  
> *Machine Learning and the Physical Sciences (ML4PS) Workshop at NeurIPS 2026*.

---

## 1. Mathematical & Algorithmic Formulation

### 1.1 The Ground-Truth Physical Landscape
The conformational benchmark evaluates transition sampling on a continuous 2D triple-well potential (in kcal/mol) representing a canonical multi-state conformational landscape at thermal energy $k_B T = 0.6\text{ kcal/mol}$:
* **Basin A (Initial kinetic trap):** $\mathbf{c}_A = (-1.0, 1.0)$, depth $d_A = 3.5$, width $w_A = 0.55$
* **Basin B:** $\mathbf{c}_B = (1.0, -0.5)$, depth $d_B = 4.0$, width $w_B = 0.60$
* **Basin C:** $\mathbf{c}_C = (-0.5, -1.0)$, depth $d_C = 3.2$, width $w_C = 0.50$

The analytical potential includes harmonic confinement:

$$
V(\mathbf{x}) = -\sum_{k=1}^3 d_k \exp\left(-\frac{\|\mathbf{x} - \mathbf{c}_k\|^2}{2 w_k}\right) + 0.20 \|\mathbf{x}\|^2
$$

---

### 1.2 The Neural Network (`CliffordScoreNet`) & Energy Scale Calibration
The network learns the potential landscape and outputs geometric flow operators via a 4-layer MLP ($\mathbb{R}^2 \to 64 \to 64 \to 64 \to \mathbb{R}^3$) with SiLU activations:

$$
\mathbf{x} \in \mathbb{R}^2 \xrightarrow{\quad \text{MLP with SiLU} \quad} \bigl( V_\theta(\mathbf{x}), \; \sigma(\mathbf{x}), \; \kappa(\mathbf{x}) \bigr)^\top
$$

1. **Scalar Potential** ($V_\theta(\mathbf{x}) \in \mathbb{R}$): Continuous scalar potential landscape. Because Denoising Score Matching (DSM) regresses against $-\boldsymbol{\epsilon}/\sigma_n^2$, it learns the dimensionless score $\nabla \log p(\mathbf{x}) = -\nabla V(\mathbf{x}) / (k_B T)$, yielding an empirical inverse-temperature scale factor $\alpha_E \approx 1.678\text{ (kcal/mol)}^{-1}$ (matching $(k_B T)^{-1}$).
2. **Rotor Chirality** ($\sigma(\mathbf{x}) = 1.5 \tanh(\cdot) \in [-1.5, 1.5]$): Dictates transverse contour circulation along equipotential isolines.
3. **Restoration Dilator** ($\kappa(\mathbf{x}) = \text{softplus}(\cdot) + 0.1 \in [0.1, \infty)$): Controls contractive descent ($\bar{\kappa} \approx 0.33\text{--}0.55$), which naturally counterbalances $\alpha_E$ ($\bar{\kappa}\alpha_E \approx 0.56\text{--}0.92 \approx 1$).

---

### 1.3 Fokker–Planck Stationarity vs. Non-Equilibrium Active Confinement
Under the overdamped Ito SDE with diffusion $D = k_B T / \gamma$:

$$
d\mathbf{x} = \gamma^{-1}(\sigma(\mathbf{x}) I - \kappa(\mathbf{x}))\nabla V_\theta(\mathbf{x})dt + \sqrt{2 D}\,d\mathbf{W}_t
$$

the Fokker–Planck equation decomposes into reversible and circulatory currents:

$$
\frac{\partial \rho}{\partial t} = \nabla \cdot \left(\gamma^{-1}\kappa(\mathbf{x})\rho\nabla V_\theta + D\nabla\rho\right) - \nabla \cdot \left(\gamma^{-1}\sigma(\mathbf{x}) I \nabla V_\theta\,\rho\right)
$$

* **Measure-Preserving Non-Reversible Regime:** When $\kappa$ is constant (with calibrated potential $\kappa V_\theta = V$) and $\sigma = \sigma(V_\theta(\mathbf{x}))$, the circulatory current $J_{\text{irr}}$ is strictly divergence-free ($\nabla \cdot J_{\text{irr}} = 0$) at the target Boltzmann distribution $\rho_{\text{B}}(\mathbf{x})$, preserving exact Boltzmann stationarity while accelerating barrier crossing.
* **Non-Equilibrium Steady State (NESS) Regime:** Driving $\kappa$ and $\sigma$ outside fluctuation-dissipation creates an active NESS with persistent boundary currents ($\nabla \times J_{\text{NESS}} \neq \mathbf{0}$) and sharp topological confinement (used in the non-convex `'G'` glyph navigation benchmark).

---

## 2. Multi-Well Benchmark & Component Ablations (Table 1 & Figure 1)

* **Ensemble:** $N = 2,000$ particles initialized in **Basin A** ($\mathcal{N}(\mathbf{c}_A, 0.15^2 \mathbf{I})$).
* **Integration:** $T = 1,000$ steps ($\Delta t = 0.015, \gamma = 1.0, k_B T = 0.6$), evaluated across 5 random seeds.
* **Small-Timestep Verification:** Verified at $3\times$ smaller timestep ($\Delta t = 0.005, T = 3,000$, identical physical horizon $T \cdot \Delta t = 15.0$).

### Verified Results (Table 1 — Seed 42 & 5-Seed Mean ± Std)

| Sampling Method / Ablation | FES RMSE (kcal/mol) $\downarrow$ | $\mathcal{W}_1$ Distance $\downarrow$ | Term. B/C Occ. (%) | Cumul. B/C Visit. (%) $\uparrow$ | Transitions / Pt. $\uparrow$ | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Brownian Dynamics** ($\mathbf{f} = \mathbf{0}$) | 2.572 ± 0.136 | 3.127 ± 0.035 | 2.0 ± 0.4% | 56.4 ± 1.0% | 1.74 ± 0.06 | 0.13 |
| **Standard Langevin** ($\kappa = 1, \sigma = 0$) | 1.476 ± 0.069 | **0.066 ± 0.008** | **60.9 ± 1.6%** | **99.7 ± 0.1%** | 21.47 ± 0.13 | 2.31 |
| **Dilator-Only Ablation** ($\sigma = 0, \kappa$) | 1.064 ± 0.044 | 0.266 ± 0.026 | 44.2 ± 1.5% | 96.9 ± 0.3% | 13.86 ± 0.15 | 2.20 |
| **A-Block Rotor-Only** ($\kappa = 0, \sigma$) | 3.028 ± 0.194 | 4.243 ± 0.017 | 0.4 ± 0.2% | 70.1 ± 0.8% | 2.82 ± 0.04 | 2.38 |
| **A-Block Full Clifford (Ours)** | **1.059 ± 0.031** | 0.385 ± 0.016 | 37.4 ± 0.7% | **99.7 ± 0.1%** | **22.48 ± 0.15** | 2.23 |

---

## 3. Standardized Non-Convex Typography Benchmark (Figure 3)

All four dynamics are evaluated on the $400 \times 400$ non-convex letterform `'G'` from the exact same initial seed in the inner horizontal spur ($N = 1,200$, $T = 4,000$, $\Delta t = 2.0$), with identical stochastic noise (`noise_amp = 0.85`) and identical circular aperture boundary clamping ($r_{\text{dish}} = 176\text{ px}$).

| Method | In-Glyph Precision (%) $\uparrow$ | Cumulative Trajectory Visitation (%) $\uparrow$ | Terminal Coverage (%) $\uparrow$ |
| :--- | :---: | :---: | :---: |
| **Brownian Dynamics** ($\mathbf{f} = \mathbf{0}$) | 26.1% | 99.3% | 90.5% |
| **Standard Langevin** ($\sigma = 0, \kappa = 1.0$) | 98.9% | 33.2% | 33.7% |
| **Rotor-Only** ($\sigma = 11.0, \kappa = 0$) | 0.1% | 100.0% | 1.4% |
| **A-Block Full Clifford** ($\sigma = 11.0, \kappa = 1.8$) | **99.9%** | **97.8%** | **99.2%** |

---

## 4. Quickstart Commands

```bash
# 1. Run unit tests
pytest tests/ -v

# 2. Reproduce Table 1, multi-seed ablations, and Figure 1
python3 reproduce_table1_and_figure1.py

# 3. Reproduce Figure 3 (standardized non-convex 'G' benchmark)
python3 reproduce_typography_figure3.py
```
