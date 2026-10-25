xw# Parameter & Artifact Provenance (`PROVENANCE.md`)

This document records the complete provenance of every physical parameter, neural network hyperparameter, integration setting, trained weight checkpoint, and visual figure in **"A-Block: Active Form Dynamics via Clifford Neural Score Matching"** (ML4PS @ NeurIPS 2026).

---

## 1. Complete Artifact & Figure Provenance

| Paper Element | File in Repository | Generating Script / Source | Deterministic Seed / Checkpoint |
| :--- | :--- | :--- | :--- |
| **Trained DSM Network** (`CliffordScoreNet`) | `output/clifford_score_net.pt` | `reproduce_table1_and_figure1.py` | `torch.manual_seed(42)`, `np.random.seed(42)` |
| **Table 1 & Multi-Seed Ablations** | `output/table1_results.json` | `reproduce_table1_and_figure1.py` | Canonical Seed `42`; 5-Seed Eval `{42, 100, 2026, 7, 99}` |
| **Figure 1** (3-Basin Kinetic Trap Benchmark) | `output/figure1_annotated_benchmark.pdf` | `reproduce_table1_and_figure1.py` | Seed `42` ($N=2,000, T=1,000, \Delta t = 0.015$) |
| **Figure 2** (`A1`–`A5` `"MAGIC"` Apertures) | `paper/d.png` | `interactive/magic_apertures/sketch.js` | p5.js multi-aperture bioluminescent simulation |
| **Figure 3 & Metrics** (`'G'` Glyph Benchmark) | `output/figure3_typography_benchmark.pdf`, `output/typography_metrics.json` | `reproduce_typography_figure3.py` | Seed `42` ($N=1,200, T=4,000, \Delta t = 2.0$) |
| **Interactive Proposition 1 Explorer** | `interactive/proposition1_visualizer/` | `interactive/proposition1_visualizer/sketch.js` | Real-time $O(N^2)$ graph vs. $O(N)$ point-measure comparison |

---

## 2. Parameter Provenance: Multi-Well Conformational Benchmark (Section 4)

### 2.1 Ground-Truth Analytical Potential (`src/potentials.py`)
The 2D three-basin conformational potential $V(\mathbf{x})$ (in kcal/mol) models three metastable conformational states separated by saddle barriers:

$$
V(\mathbf{x}) = -\sum_{k \in \{A, B, C\}} d_k \exp\left(-\frac{\|\mathbf{x} - \mathbf{c}_k\|^2}{2 w_k}\right) + c_{\text{conf}} \|\mathbf{x}\|^2
$$

| Parameter | Symbol | Value | Physical Role / Provenance |
| :--- | :---: | :---: | :--- |
| Basin A Center, Depth, Width | $\mathbf{c}_A$, $d_A$, $w_A$ | $(-1.0, 1.0), \; 3.5\text{ kcal/mol}, \; 0.55$ | Initial kinetic trap state |
| Basin B Center, Depth, Width | $\mathbf{c}_B$, $d_B$, $w_B$ | $(1.0, -0.5), \; 4.0\text{ kcal/mol}, \; 0.60$ | Global minimum conformational basin |
| Basin C Center, Depth, Width | $\mathbf{c}_C$, $d_C$, $w_C$ | $(-0.5, -1.0), \; 3.2\text{ kcal/mol}, \; 0.50$ | Intermediate metastable state |
| Harmonic Confinement | $c_{\text{conf}}$ | $0.20\text{ kcal/mol}\cdot\text{unit}^{-2}$ | Prevents infinite spatial escape at large $\|\mathbf{x}\|$ |
| Thermal Energy | $k_B T$ | $0.6\text{ kcal/mol}$ | Room-temperature biomolecular energy scale ($T \approx 300\text{ K}$) |
| Basin Core Indicator Radius | $r_{\text{basin}}^2$ | $\|\mathbf{x} - \mathbf{c}_k\|^2 < 0.45$ | Defines core basin masks $\mathcal{B}_A$, $\mathcal{B}_B$, and $\mathcal{B}_C$ |
| Low-Energy FES Evaluation Mask | $\mathcal{M}$ | $V(\mathbf{x}) < -1.2\text{ kcal/mol}$ | Restricts FES RMSE to populated wells & transition saddles |

> **Boltzmann Reference Partitioning:** Under the exact Boltzmann density $p_{\text{B}}(\mathbf{x})$ on $[-2.2, 2.2]^2$, 8.6% of probability mass lies in $\mathcal{B}_A$, 40.5% in $\mathcal{B}_B$, 23.1% in $\mathcal{B}_C$ (63.6% total in $\mathcal{B}_B$ and $\mathcal{B}_C$), and 27.8% in the surrounding basin skirts and saddle regions.

---

### 2.2 `CliffordScoreNet` & Denoising Score Matching Calibration (`src/models.py`, `src/training.py`)

| Parameter | Value | Provenance & Physical Meaning |
| :--- | :---: | :--- |
| MLP Architecture | $2 \to 64 \to 64 \to 64 \to 3$ (SiLU) | Smooth $C^\infty$ differentiability for $\nabla V_\theta(\mathbf{x})$ |
| Training Corpus | 12,000 samples (4,000 per basin) | Relaxed via 150 unadjusted Langevin steps ($\Delta t = 0.012, k_B T = 0.6$) |
| DSM Noise Levels | $\sigma_n \in \{0.05, 0.10, 0.20\}$ | Multi-scale perturbation covering core wells and saddle regions |
| Optimizer & Schedule | AdamW ($\text{lr}=2\times 10^{-3}$, wd $=10^{-5}$) | 250 epochs, cosine annealing, gradient norm clipping at 1.0 |
| **Energy Calibration Factor** ($\alpha_E$) | $\alpha_E \approx 1.678\text{ (kcal/mol)}^{-1}$ | DSM target $-\boldsymbol{\epsilon}/\sigma_n^2$ estimates dimensionless score $-\nabla V / (k_B T)$ |
| Rotor Chirality Output | $\sigma(\mathbf{x}) = 1.5\tanh(z_1) \in [-1.5, 1.5]$ | Unconstrained random-feature field ($\bar{\sigma} = -1.31 \pm 0.26$) |
| Dilator Contraction Output | $\kappa(\mathbf{x}) = \text{softplus}(z_2) + 0.1$ | Unconstrained random-feature field ($\bar{\kappa} = 0.33 \pm 0.17$), which naturally counterbalances $\alpha_E$ ($\bar{\kappa}\alpha_E \approx 0.56\text{--}0.92 \approx 1$) |

---

## 3. Parameter Provenance: Non-Convex Typography Benchmark (Appendix A / Figure 3)

| Parameter | Symbol | Value | Provenance & Role |
| :--- | :---: | :---: | :--- |
| Spatial Grid & Font | $H \times W$ | $400 \times 400\text{ px}$ | Sans-serif bold `'G'` (`size=270`), centered in circular aperture |
| Gaussian Smoothing Width | $\sigma_b$ | $7.5\text{ px}$ | Smooths binary stroke $\mathbf{1}_{\text{glyph}}$ into differentiable log-density |
| Circular Dish Radius | $r_{\text{dish}}$ | $176.0\text{ px}$ ($0.44 \times 400$) | Identical hard/harmonic boundary clamping across all 4 baselines |
| Initial Seed Location | $\mathbf{x}_0$ | $(268, 224) + \mathcal{N}(\mathbf{0}, 2.5^2\mathbf{I})$ | Localized inside the sharp inner horizontal spur of the `'G'` |
| Ensemble & Horizon | $N, T, \Delta t$ | $N=1,200, \; T=4,000, \; \Delta t = 2.0$ | Standardized across all 4 methods |
| Stochastic Noise Amplitude | `noise_amp` | $0.85\text{ px}\cdot\text{step}^{-1/2}$ | Identical thermal agitation across all 4 methods |
| A-Block Rotor / Dilator Gains | $\sigma, \kappa$ | $\sigma = 11.0, \; \kappa = 1.8$ | Non-equilibrium active confinement regime (Proposition 1, Case 2) |
